"""Bengali text normalization, Grapheme-to-Phoneme (G2P), and IPA tokenization.

Designed for StyleTTS 2 and expressive neural speech synthesis in Samvyas.
Handles Bengali punctuation, numerals, inherent vowel insertion, conjuncts,
and maps text to IPA token sequences.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import ClassVar

# Bengali digit to spoken Bengali word mapping
BENGALI_DIGITS: dict[str, str] = {
    "০": "শূন্য",
    "১": "এক",
    "২": "দুই",
    "৩": "তিন",
    "৪": "চার",
    "৫": "পাঁচ",
    "৬": "ছয়",
    "৭": "সাত",
    "৮": "আট",
    "৯": "নয়",
}

# Standard Bengali conjuncts (juktakkhor) to IPA
BENGALI_CONJUNCTS_IPA: dict[str, str] = {
    "ক্ষ": "kʰ kʰ",
    "জ্ঞ": "ɡ̃ ɡ̃",
    "ঙ্ক": "ŋ k",
    "ঙ্গ": "ŋ ɡ",
    "ঞ্চ": "n t͡ʃ",
    "ঞ্ছ": "n t͡ʃʰ",
    "ঞ্জ": "n d͡ʒ",
    "ট্ট": "ʈ ʈ",
    "ট্ঠ": "ʈ ʈʰ",
    "ড্ড": "ɖ ɖ",
    "ণ্ট": "ɳ ʈ",
    "ণ্ঠ": "ɳ ʈʰ",
    "ণ্ড": "ɳ ɖ",
    "ত্ত": "t̪ t̪",
    "ত্থ": "t̪ t̪ʰ",
    "দ্দ": "d̪ d̪",
    "দ্ধ": "d̪ d̪ʱ",
    "ন্ত": "n t̪",
    "ন্থ": "n t̪ʰ",
    "ন্দ": "n d̪",
    "ন্ধ": "n d̪ʱ",
    "ন্ন": "n n",
    "প্ত": "p t̪",
    "প্স": "p ʃ",
    "ব্দ": "b d̪",
    "ব্ধ": "b d̪ʱ",
    "ব্ব": "b b",
    "ম্প": "m p",
    "ম্ফ": "m pʰ",
    "ম্ব": "m b",
    "ম্ভ": "m bʱ",
    "ম্ম": "m m",
    "ল্ক": "l k",
    "ল্গ": "l ɡ",
    "ল্প": "l p",
    "ল্ল": "l l",
    "ষ্ক": "ʃ k",
    "ষ্ট": "ʃ ʈ",
    "ষ্ঠ": "ʃ ʈʰ",
    "স্প": "ʃ p",
    "স্ফ": "ʃ pʰ",
    "স্ত": "s t̪",
    "স্থ": "s t̪ʰ",
    "স্নে": "s n e",
    "স্প্ল": "s p l",
}

# Standalone Bengali vowels to IPA
BENGALI_VOWELS_IPA: dict[str, str] = {
    "অ": "ɔ",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi̯",
    "ও": "o",
    "ঔ": "ou̯",
}

# Bengali vowel diacritics (kar) to IPA
BENGALI_KARS_IPA: dict[str, str] = {
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi̯",
    "ো": "o",
    "ৌ": "ou̯",
}

# Standalone Bengali consonants to base IPA (without inherent vowel)
BENGALI_CONSONANTS_IPA: dict[str, str] = {
    "ক": "k",
    "খ": "kʰ",
    "গ": "ɡ",
    "ঘ": "ɡʱ",
    "ঙ": "ŋ",
    "চ": "t͡ʃ",
    "ছ": "t͡ʃʰ",
    "জ": "d͡ʒ",
    "ঝ": "d͡ʒʱ",
    "ঞ": "n",
    "ট": "ʈ",
    "ঠ": "ʈʰ",
    "ড": "ɖ",
    "ঢ": "ɖʱ",
    "ণ": "n",
    "ত": "t̪",
    "থ": "t̪ʰ",
    "দ": "d̪",
    "ধ": "d̪ʱ",
    "ন": "n",
    "প": "p",
    "ফ": "pʰ",
    "ব": "b",
    "ভ": "bʱ",
    "ম": "m",
    "য": "d͡ʒ",
    "র": "r",
    "ল": "l",
    "শ": "ʃ",
    "ষ": "ʃ",
    "স": "ʃ",
    "হ": "h",
    "ড়": "ɽ",
    "ঢ়": "ɽʱ",
    "য়": "e̯",
    "ৎ": "t̪",
}

VIRAMA = "্"
CHANDRABINDU = "ঁ"
ANUSVARA = "ং"
VISARGA = "ঃ"


class BengaliTextNormalizer:
    """Cleans and standardizes raw Bengali text for speech synthesis."""

    @staticmethod
    def normalize(text: str) -> str:
        """Applies canonical NFC normalization, expands digits, and cleans punctuation.

        Parameters
        ----------
        text : str
            Raw input Bengali text.

        Returns
        -------
        str
            Sanitized, speakable Bengali text.
        """
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Expand Bengali digits
        for digit, word in BENGALI_DIGITS.items():
            text = text.replace(digit, f" {word} ")

        # Expand ASCII digits if any
        ascii_digits = {
            "0": "শূন্য",
            "1": "এক",
            "2": "দুই",
            "3": "তিন",
            "4": "চার",
            "5": "পাঁচ",
            "6": "ছয়",
            "7": "সাত",
            "8": "আট",
            "9": "নয়",
        }
        for digit, word in ascii_digits.items():
            text = text.replace(digit, f" {word} ")

        # Normalize punctuation: keep Bengali Dari '।' and standard delimiters
        text = re.sub(r"[\r\t\n]+", " ", text)
        text = re.sub(r"[।!?]+", " । ", text)
        text = re.sub(r"[,;]+", " , ", text)
        # Remove unsupported symbols
        text = re.sub(r"[^\u0980-\u09FF\s।,!?a-zA-Z]", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


class BengaliPhonemizer:
    """Converts normalized Bengali text into an International Phonetic Alphabet (IPA) sequence."""

    def __init__(self, default_inherent_vowel: str = "ɔ") -> None:
        self.default_inherent = default_inherent_vowel

    def word_to_ipa(self, word: str) -> str:
        """Converts a single Bengali word to IPA representation."""
        if not word:
            return ""

        # Check for punctuation markers
        if word in ("।", "!", "?"):
            return "‖"
        if word == ",":
            return "|"

        # Check for known conjunct prefixes first
        for conjunct, ipa in BENGALI_CONJUNCTS_IPA.items():
            if conjunct in word:
                word = word.replace(conjunct, f"§{ipa}§")

        result: list[str] = []
        chars = list(word)
        n = len(chars)
        i = 0

        while i < n:
            ch = chars[i]

            # Preserved conjunct replacement token
            if ch == "§":
                end = word.find("§", i + 1)
                if end != -1:
                    result.append(word[i + 1 : end])
                    i = end + 1
                    continue

            # Vowels
            if ch in BENGALI_VOWELS_IPA:
                result.append(BENGALI_VOWELS_IPA[ch])
                i += 1
                continue

            # Consonants
            if ch in BENGALI_CONSONANTS_IPA:
                base_c = BENGALI_CONSONANTS_IPA[ch]
                has_next = i + 1 < n
                next_ch = chars[i + 1] if has_next else None

                # Hasanta / Virama -> suppress vowel
                if next_ch == VIRAMA:
                    result.append(base_c)
                    i += 2
                    continue

                # Kar sign attaches
                if next_ch in BENGALI_KARS_IPA:
                    result.append(base_c)
                    result.append(BENGALI_KARS_IPA[next_ch])
                    i += 2
                    continue

                # Modifier characters
                if next_ch == CHANDRABINDU:
                    result.append(base_c)
                    result.append(self.default_inherent)
                    result.append("̃")
                    i += 2
                    continue
                if next_ch == ANUSVARA:
                    result.append(base_c)
                    result.append(self.default_inherent)
                    result.append("ŋ")
                    i += 2
                    continue
                if next_ch == VISARGA:
                    result.append(base_c)
                    result.append(self.default_inherent)
                    result.append("h")
                    i += 2
                    continue

                # Inherent vowel rule: word-final consonants usually drop inherent vowel in Bengali,
                # except in monosyllabic or specific morphological stems.
                result.append(base_c)
                if has_next:
                    result.append(self.default_inherent)

                i += 1
                continue

            # Modifiers standalone
            if ch == ANUSVARA:
                result.append("ŋ")
            elif ch == VISARGA:
                result.append("h")
            elif ch == CHANDRABINDU:
                result.append("̃")

            i += 1

        return " ".join(filter(None, result))

    def phonemize(self, text: str) -> str:
        """Phonemizes full Bengali text into space-separated IPA tokens."""
        norm_text = BengaliTextNormalizer.normalize(text)
        words = norm_text.split()
        ipa_words = [self.word_to_ipa(w) for w in words]
        return "  ".join(filter(None, ipa_words))


@dataclass
class BengaliPhonemeTokenizer:
    """Maps IPA phonemes to discrete token IDs for the neural embedding table."""

    PAD_TOKEN: ClassVar[str] = "<pad>"
    UNK_TOKEN: ClassVar[str] = "<unk>"
    BLANK_TOKEN: ClassVar[str] = "<blank>"
    PAUSE_TOKEN: ClassVar[str] = "‖"
    SHORT_PAUSE_TOKEN: ClassVar[str] = "|"

    vocab: dict[str, int] = field(default_factory=dict)
    inverse_vocab: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            self._build_default_vocab()

    def _build_default_vocab(self) -> None:
        """Builds standard Bengali IPA vocabulary."""
        special = [
            self.PAD_TOKEN,
            self.BLANK_TOKEN,
            self.UNK_TOKEN,
            self.PAUSE_TOKEN,
            self.SHORT_PAUSE_TOKEN,
        ]
        symbols = [
            "a",
            "i",
            "u",
            "e",
            "o",
            "ɔ",
            "ɛ",
            "oi̯",
            "ou̯",
            "ri",
            "k",
            "kʰ",
            "ɡ",
            "ɡʱ",
            "ɡ̃",
            "ŋ",
            "t͡ʃ",
            "t͡ʃʰ",
            "d͡ʒ",
            "d͡ʒʱ",
            "n",
            "ʈ",
            "ʈʰ",
            "ɖ",
            "ɖʱ",
            "ɳ",
            "t̪",
            "t̪ʰ",
            "d̪",
            "d̪ʱ",
            "p",
            "pʰ",
            "b",
            "bʱ",
            "m",
            "r",
            "l",
            "ʃ",
            "s",
            "h",
            "ɽ",
            "ɽʱ",
            "e̯",
            "̃",
            " ",
        ]
        all_tokens = special + symbols
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, phoneme_str: str) -> list[int]:
        """Encodes space-separated IPA tokens into vocabulary IDs."""
        tokens = phoneme_str.split()
        encoded: list[int] = []
        unk_id = self.vocab.get(self.UNK_TOKEN, 2)
        for tok in tokens:
            encoded.append(self.vocab.get(tok, unk_id))
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Decodes token IDs back into an IPA string."""
        tokens = [self.inverse_vocab.get(tid, self.UNK_TOKEN) for tid in token_ids]
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
