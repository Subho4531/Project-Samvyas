"""Modern Transformer Decoder Backbone with RMSNorm, SwiGLU, GQA, and KV Cache."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from voicellm.models.backbones.rope import RotaryEmbedding, apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w1(x)) * w3(x)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    """Grouped-Query Multi-Head Causal Attention with RoPE and dynamic KV-Cache."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = rope(x, seq_len=T, offset=offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v)

        # Repeat KV heads for GQA if needed
        if self.num_queries_per_kv > 1:
            k_expanded = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
            v_expanded = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)
        else:
            k_expanded, v_expanded = k, v

        # Scaled Dot-Product Attention with causal masking
        # Supports PyTorch 2.0+ native FlashAttention/SDPA
        is_causal = (mask is None) and (T > 1) and (kv_cache is None)
        out = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training and isinstance(self.dropout, nn.Dropout) else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv_cache


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm and SwiGLU."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim=dim, hidden_dim=intermediate_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with pre-norm & residual
        normed_x = self.attn_norm(x)
        attn_out, new_kv = self.attn(normed_x, rope=rope, kv_cache=kv_cache, mask=mask, offset=offset)
        x = x + attn_out

        # FFN with pre-norm & residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class TransformerDecoder(nn.Module):
    """Stack of Transformer Decoder blocks."""

    def __init__(
        self,
        hidden_dim: int = 768,
        intermediate_dim: int = 2048,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        max_seq_len: int = 4096,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[list[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        mask: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_kv_caches = []
        h = x
        for layer, cache in zip(self.layers, kv_caches):
            h, new_cache = layer(h, rope=self.rope, kv_cache=cache, mask=mask, offset=offset)
            new_kv_caches.append(new_cache)

        h = self.final_norm(h)
        return h, new_kv_caches
