"""Unified VoiceLLM Model with Multi-Codebook Autoregressive Audio Modeling."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from voicellm.core.config import ModelConfig
from voicellm.models.backbones.transformer import TransformerDecoder
from voicellm.models.conditioning.emotion_encoder import EmotionConditioner
from voicellm.models.conditioning.speaker_encoder import SpeakerProjector


class VoiceLLM(nn.Module):
    """End-to-End Multimodal Voice LLM.
    
    Generates multi-codebook acoustic tokens conditioned on:
    1. Text prompt tokens (phonemes or subwords)
    2. Zero-shot speaker conditioning embeddings
    3. Emotion / Prosody control latents
    4. Causal history of acoustic codebooks (delay pattern or hierarchical RVQ)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_codebooks = config.num_codebooks
        self.codebook_vocab_size = config.codebook_vocab_size

        # 1. Text Token Embeddings
        self.text_embed = nn.Embedding(config.vocab_size_text, config.hidden_dim)

        # 2. Multi-Codebook Acoustic Embeddings (One per RVQ level + 1 for Special Tokens / BOS / EOS / PAD)
        # Extra tokens: [PAD=0, BOS=1, EOS=2, AUDIO_START=3, AUDIO_END=4]
        self.special_token_offset = 5
        self.total_acoustic_vocab = config.codebook_vocab_size + self.special_token_offset

        self.audio_embeds = nn.ModuleList([
            nn.Embedding(self.total_acoustic_vocab, config.hidden_dim)
            for _ in range(config.num_codebooks)
        ])

        # 3. Conditioning Modules
        self.speaker_proj = SpeakerProjector(
            embedding_dim=config.speaker_encoder.embedding_dim,
            hidden_dim=config.hidden_dim,
        )
        self.emotion_conditioner = EmotionConditioner(
            num_emotions=config.emotion.num_emotions,
            hidden_dim=config.hidden_dim,
            use_continuous_latents=True,
        )

        # 4. Transformer Decoder Backbone
        self.backbone = TransformerDecoder(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        # 5. Multi-Codebook Prediction Heads (One linear head per RVQ level)
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, self.total_acoustic_vocab, bias=False)
            for _ in range(config.num_codebooks)
        ])

    def embed_acoustic_tokens(self, acoustic_tokens: torch.Tensor, already_shifted: bool = False) -> torch.Tensor:
        """Sums embeddings across all active RVQ codebooks.
        
        Args:
            acoustic_tokens: (B, num_codebooks, T_audio)
            already_shifted: bool indicating if special_token_offset has already been applied
        Returns:
            acoustic_emb: (B, T_audio, hidden_dim)
        """
        B, K, T = acoustic_tokens.shape
        shifted_tokens = acoustic_tokens if already_shifted else (acoustic_tokens + self.special_token_offset)
        
        emb = sum(self.audio_embeds[k](shifted_tokens[:, k, :]) for k in range(K))
        return emb

    def forward(
        self,
        text_tokens: torch.Tensor,
        acoustic_tokens: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        emotion_id: Optional[torch.Tensor] = None,
        continuous_vad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training forward pass.
        
        Args:
            text_tokens: (B, T_text)
            acoustic_tokens: (B, num_codebooks, T_audio)
            speaker_emb: Optional (B, speaker_dim)
            emotion_id: Optional (B,)
            continuous_vad: Optional (B, 3)
            
        Returns:
            Dict containing 'logits' of shape (B, num_codebooks, T_audio, vocab_size)
        """
        prefix_embs = []

        # Speaker conditioning token
        if speaker_emb is not None:
            spk_token = self.speaker_proj(speaker_emb)  # (B, 1, hidden_dim)
            prefix_embs.append(spk_token)

        # Emotion conditioning token
        if emotion_id is not None or continuous_vad is not None:
            emo_token = self.emotion_conditioner(emotion_id=emotion_id, continuous_vad=continuous_vad)
            if emo_token is not None:
                prefix_embs.append(emo_token)

        # Text prompt embeddings
        text_emb = self.text_embed(text_tokens)  # (B, T_text, hidden_dim)
        prefix_embs.append(text_emb)

        # Concatenate prefix sequence
        prefix = torch.cat(prefix_embs, dim=1)  # (B, T_prefix, hidden_dim)
        prefix_len = prefix.shape[1]

        if acoustic_tokens is not None:
            acoustic_emb = self.embed_acoustic_tokens(acoustic_tokens)  # (B, T_audio, hidden_dim)
            full_input = torch.cat([prefix, acoustic_emb], dim=1)
        else:
            full_input = prefix

        # Causal transformer backbone forward
        h, _ = self.backbone(full_input)

        # Slice out the acoustic prediction slice
        # The model predicts the next acoustic token at every step
        h_audio = h[:, prefix_len - 1 : -1, :]  # (B, T_audio, hidden_dim)

        # Project through each codebook head
        logits = torch.stack([head(h_audio) for head in self.lm_heads], dim=1)  # (B, K, T_audio, vocab)

        return {"logits": logits, "hidden_states": h}

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        emotion_id: Optional[torch.Tensor] = None,
        continuous_vad: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Autoregressive acoustic token generation with KV-caching.
        
        Returns:
            generated_tokens: (B, num_codebooks, generated_len)
        """
        self.eval()
        device = text_tokens.device
        B = text_tokens.shape[0]

        # 1. Build Prefix sequence
        prefix_embs = []
        if speaker_emb is not None:
            prefix_embs.append(self.speaker_proj(speaker_emb))
        if emotion_id is not None or continuous_vad is not None:
            emo = self.emotion_conditioner(emotion_id=emotion_id, continuous_vad=continuous_vad)
            if emo is not None:
                prefix_embs.append(emo)
        prefix_embs.append(self.text_embed(text_tokens))
        prefix = torch.cat(prefix_embs, dim=1)

        # 2. Prefill KV Cache with prefix
        h, kv_caches = self.backbone(prefix)
        cur_h = h[:, -1:, :]  # (B, 1, hidden_dim)
        offset = prefix.shape[1]

        generated = [[] for _ in range(self.num_codebooks)]

        for step in range(max_new_tokens):
            step_tokens = []
            for k in range(self.num_codebooks):
                logits = self.lm_heads[k](cur_h).squeeze(1)  # (B, vocab)
                logits = logits / max(temperature, 1e-4)

                # Repetition penalty
                if repetition_penalty > 1.0 and len(generated[k]) > 0:
                    past_tokens = torch.cat(generated[k], dim=1)
                    for b in range(B):
                        for t in past_tokens[b]:
                            logits[b, t] /= repetition_penalty

                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                step_tokens.append(next_token)
                generated[k].append(next_token)

            # Stack step tokens across codebooks: (B, num_codebooks, 1)
            stacked_step = torch.stack(step_tokens, dim=1)
            # Embed for next step
            next_emb = self.embed_acoustic_tokens(stacked_step)

            # Step KV cache forward
            cur_h, kv_caches = self.backbone(next_emb, kv_caches=kv_caches, offset=offset)
            offset += 1

        # Concatenate tokens across time
        final_tokens = torch.cat([
            torch.cat(generated[k], dim=1).unsqueeze(1)
            for k in range(self.num_codebooks)
        ], dim=1)  # (B, num_codebooks, max_new_tokens)

        return final_tokens
