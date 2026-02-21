"""
encoder.py — Input Encoders

Bridges the gap between raw input (strings, numbers, sequences)
and the SDR space the pattern matcher operates in.

Key design: encoders are deterministic and structure-preserving.
  - Similar inputs → overlapping SDRs (so similarity matching works)
  - Same input → same SDR always (stable representation)
  - Different modalities can be encoded into the same space and composed

Encoders provided:
  - TokenEncoder:  text tokens / words
  - ScalarEncoder: numbers (nearby values share bits)
  - SequenceEncoder: ordered sequences (captures position + content)
  - CompositeEncoder: combine multiple encoders
"""

import numpy as np
import hashlib
from abc import ABC, abstractmethod
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY


# Stop words — down-weighted in sequence encoding so content words dominate
STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "on","in","at","to","of","and","or","but","not","no",
    "it","its","he","she","they","i","you","we","my","your",
    "this","that","these","those","with","for","from","by","as",
    "do","does","did","have","has","had","will","would","could",
    "should","may","might","can","about","which","who","what",
    "how","when","where","if","than","then","so","all","just",
    "their","there","here","also","very","more","some","any",
}


# ------------------------------------------------------------------ #
#  Base                                                                #
# ------------------------------------------------------------------ #

class Encoder(ABC):
    @abstractmethod
    def encode(self, value) -> SDR:
        ...

    def encode_batch(self, values: list) -> list[SDR]:
        return [self.encode(v) for v in values]


# ------------------------------------------------------------------ #
#  Token Encoder (text / symbols)                                     #
# ------------------------------------------------------------------ #

class TokenEncoder(Encoder):
    """
    Encodes string tokens into SDRs.
    
    Uses hash-based encoding so:
    - No vocabulary needed upfront
    - Same token always maps to same SDR
    - Related tokens can be made to overlap via subword sharing
    
    subword_overlap: if True, tokens sharing character n-grams will
    have overlapping SDRs — so "running" and "runner" will be similar.
    This is the symbolic equivalent of subword tokenization.
    """

    def __init__(self, subword_overlap: bool = True, ngram_size: int = 3):
        self.subword_overlap = subword_overlap
        self.ngram_size = ngram_size
        self._cache: dict[str, SDR] = {}

    def encode(self, token: str) -> SDR:
        token = str(token).lower().strip()
        if token in self._cache:
            return self._cache[token]

        if self.subword_overlap:
            sdr = self._encode_subword(token)
        else:
            sdr = SDR.from_hash(token, label=token)

        self._cache[token] = sdr
        return sdr

    def _encode_subword(self, token: str) -> SDR:
        """
        Build SDR as union of character n-gram SDRs.
        "cat" → SDR("#ca") ∪ SDR("cat") ∪ SDR("at#")
        So tokens sharing substrings share bits → similar SDRs.
        """
        padded = f"#{token}#"
        ngrams = [
            padded[i:i+self.ngram_size]
            for i in range(len(padded) - self.ngram_size + 1)
        ]
        if not ngrams:
            ngrams = [token]

        # Each ngram contributes bits; we take the top-sparsity bits by vote
        n_active = int(SDR_SIZE * SDR_SPARSITY)
        votes = np.zeros(SDR_SIZE, dtype=np.float32)

        for ng in ngrams:
            ng_sdr = SDR.from_hash(ng)
            votes += ng_sdr.bits.astype(np.float32)

        # Pick most-voted bits
        top_indices = np.argsort(votes)[-n_active:]
        return SDR.from_indices(top_indices.tolist(), label=token)


# ------------------------------------------------------------------ #
#  Scalar Encoder (numbers)                                           #
# ------------------------------------------------------------------ #

class ScalarEncoder(Encoder):
    """
    Encodes a number into an SDR where nearby numbers share bits.
    
    Works by mapping the number onto a circular bit range and
    activating a window of bits around that position.
    Nearby numbers → overlapping windows → similar SDRs.
    
    This is how HTM/NuPIC encodes scalars and it works beautifully.
    """

    def __init__(self, min_val: float, max_val: float, n_active: int = None):
        self.min_val = min_val
        self.max_val = max_val
        self.n_active = n_active or int(SDR_SIZE * SDR_SPARSITY)

    def encode(self, value: float) -> SDR:
        value = float(value)
        # Clamp to range
        value = max(self.min_val, min(self.max_val, value))
        # Map to position in SDR
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        center = int(ratio * (SDR_SIZE - self.n_active))
        indices = list(range(center, center + self.n_active))
        return SDR.from_indices(indices, label=str(value))


# ------------------------------------------------------------------ #
#  Sequence Encoder                                                   #
# ------------------------------------------------------------------ #

class SequenceEncoder(Encoder):
    """
    Encodes an ordered sequence of tokens as a single SDR.
    
    Captures BOTH content and order — unlike a simple bag of words.
    Uses position-shifted encoding: each token's SDR is shifted by
    its position before combining, so "AB" ≠ "BA".
    
    This is the key thing flat vector approaches get wrong —
    word order matters and this captures it symbolically.
    """

    def __init__(self, token_encoder: TokenEncoder = None, max_len: int = 32):
        self.token_encoder = token_encoder or TokenEncoder()
        self.max_len = max_len

    def encode(self, sequence: list[str]) -> SDR:
        if not sequence:
            return SDR(bits=np.zeros(SDR_SIZE, dtype=bool))

        n_active = int(SDR_SIZE * SDR_SPARSITY)
        votes = np.zeros(SDR_SIZE, dtype=np.float32)

        for pos, token in enumerate(sequence[:self.max_len]):
            token_sdr = self.token_encoder.encode(token)
            # Rotate bits by position — encodes position structurally
            shift = (pos * 17) % SDR_SIZE   # 17 is coprime with SDR_SIZE, spreads well
            rotated = np.roll(token_sdr.bits, shift)
            # Weight earlier positions slightly more (like attention to start)
            weight = 1.0 / (1.0 + pos * 0.1)
            votes += rotated.astype(np.float32) * weight

        top_indices = np.argsort(votes)[-n_active:]
        label = " ".join(sequence[:4]) + ("..." if len(sequence) > 4 else "")
        return SDR.from_indices(top_indices.tolist(), label=label)

    def encode(self, sequence) -> SDR:
        # Handle both string (tokenize) and list input
        if isinstance(sequence, str):
            tokens = sequence.lower().split()
        else:
            tokens = [str(t) for t in sequence]
        return self._encode_tokens(tokens)

    def _encode_tokens(self, tokens: list[str]) -> SDR:
        if not tokens:
            return SDR(bits=np.zeros(SDR_SIZE, dtype=bool))

        n_active = int(SDR_SIZE * SDR_SPARSITY)
        votes = np.zeros(SDR_SIZE, dtype=np.float32)

        for pos, token in enumerate(tokens[:self.max_len]):
            token_sdr = self.token_encoder.encode(token)
            shift = (pos * 3) % SDR_SIZE
            rotated = np.roll(token_sdr.bits, shift)
            # Stop words get 10% weight — still present but don't dominate
            stop_factor = 0.1 if token in STOP_WORDS else 1.0
            weight = stop_factor / (1.0 + pos * 0.15)
            votes += rotated.astype(np.float32) * weight

        top_indices = np.argsort(votes)[-n_active:]
        label = " ".join(tokens[:4]) + ("..." if len(tokens) > 4 else "")
        return SDR.from_indices(top_indices.tolist(), label=label)


# ------------------------------------------------------------------ #
#  Composite Encoder                                                  #
# ------------------------------------------------------------------ #

class CompositeEncoder(Encoder):
    """
    Combine multiple encoders into one SDR by bundling their outputs.
    
    Example: encode (word, part-of-speech, position) together
    so the pattern matcher sees the full context, not just the word.
    """

    def __init__(self, encoders: list[tuple[str, Encoder, float]]):
        """
        encoders: list of (name, encoder, weight) tuples
        weight controls how much each encoder contributes to the bundle
        """
        self.encoders = encoders

    def encode(self, values: dict) -> SDR:
        """values: dict mapping encoder name → value"""
        n_active = int(SDR_SIZE * SDR_SPARSITY)
        votes = np.zeros(SDR_SIZE, dtype=np.float32)
        total_weight = 0.0

        for name, encoder, weight in self.encoders:
            if name in values:
                sdr = encoder.encode(values[name])
                votes += sdr.bits.astype(np.float32) * weight
                total_weight += weight

        if total_weight == 0:
            return SDR(bits=np.zeros(SDR_SIZE, dtype=bool))

        top_indices = np.argsort(votes)[-n_active:]
        return SDR.from_indices(top_indices.tolist())