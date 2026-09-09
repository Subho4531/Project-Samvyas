"""Text and Acoustic Tokenizer Interfaces."""

from typing import List, Union
import torch


class VoiceTextTokenizer:
    """Wrapper for text tokenization (supporting HuggingFace tokenizers or fallback character/byte tokenizer)."""

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self.hf_tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        except Exception:
            # Fallback basic ASCII / UTF-8 byte tokenizer
            self.hf_tokenizer = None

    def encode(self, text: Union[str, List[str]], max_length: int = 512) -> torch.Tensor:
        if self.hf_tokenizer is not None:
            if isinstance(text, str):
                text = [text]
            tokens = self.hf_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return tokens.input_ids
        else:
            # Fallback byte encoding
            if isinstance(text, str):
                text = [text]
            encoded = []
            for t in text:
                bytes_list = list(t.encode("utf-8"))[:max_length]
                encoded.append(torch.tensor(bytes_list, dtype=torch.long))
            return torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)

    def decode(self, tokens: torch.Tensor) -> List[str]:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.batch_decode(tokens, skip_special_tokens=True)
        else:
            decoded = []
            for seq in tokens:
                valid_bytes = [b.item() for b in seq if b.item() != 0]
                decoded.append(bytes(valid_bytes).decode("utf-8", errors="ignore"))
            return decoded
