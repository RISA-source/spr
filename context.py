"""
context.py — Context-Sensitive Encoding (v2.2)

The problem: in flat encoding, "bank" always has the same SDR whether
you say "river bank" or "bank account". The system can't tell them apart.

Solution: Partitioned SDR space.
  - Bits 0..BASE_END   → encode WORD IDENTITY (same word = same bits here)
  - Bits BASE_END..END → encode LOCAL CONTEXT (neighbors determine these bits)

This way:
  "bank" near "river"   → shared base bits + river-tinted context bits
  "bank" near "account" → shared base bits + account-tinted context bits
  Same base identity, different context signature.

Similarity between the two "bank"s drops from 1.0 to ~0.68
while maintaining enough overlap to know it's still the same word.

This is NOT full attention. It's closest to:
  - 1D convolution with a local receptive field
  - Factored word + context embedding
It handles local disambiguation, modifier scope, phrase coherence.
It does NOT handle long-range dependencies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY
from .encoder import TokenEncoder, SequenceEncoder

# ------------------------------------------------------------------ #
#  Partitioned SDR dimensions                                          #
# ------------------------------------------------------------------ #

BASE_RATIO   = 0.65    # fraction of SDR_SIZE used for word identity
CTX_RATIO    = 0.35    # fraction for local context
BASE_END     = int(SDR_SIZE * BASE_RATIO)
CTX_START    = BASE_END
BASE_ACTIVE  = int(BASE_END * SDR_SPARSITY)
CTX_ACTIVE   = int((SDR_SIZE - BASE_END) * SDR_SPARSITY)
CTX_WINDOW   = 2       # neighbors on each side


class ContextualTokenEncoder:
    """
    Encodes tokens with awareness of local context.
    Uses partitioned SDR space: base identity bits + context bits.
    Same word + different context = different SDR, partially overlapping.
    """

    def __init__(
        self,
        base_encoder: TokenEncoder = None,
        window: int = CTX_WINDOW,
        blend: float = 0.25,    # kept for API compat, not used in partition approach
    ):
        self.base_encoder = base_encoder or TokenEncoder(subword_overlap=True)
        self.window = window

    def encode_in_context(self, tokens: list[str], position: int) -> SDR:
        """Encode token at position with local context."""
        if not tokens or position >= len(tokens):
            return SDR(bits=np.zeros(SDR_SIZE, dtype=bool))

        target = tokens[position]

        # --- Base bits: word identity (mapped to base region) ---
        base_sdr = self.base_encoder.encode(target)
        # Remap base SDR's active indices into the base region [0, BASE_END)
        base_indices_raw = base_sdr.active_indices()
        base_indices = list(set(
            int(idx * BASE_END / SDR_SIZE)
            for idx in base_indices_raw
        ))[:BASE_ACTIVE]

        # --- Context bits: aggregate neighbors into context region ---
        ctx_votes = np.zeros(SDR_SIZE - BASE_END, dtype=np.float32)

        for offset in range(-self.window, self.window + 1):
            if offset == 0:
                continue
            nbr_pos = position + offset
            if nbr_pos < 0 or nbr_pos >= len(tokens):
                continue
            nbr_sdr = self.base_encoder.encode(tokens[nbr_pos])
            weight = 1.0 / abs(offset)
            # Map neighbor bits into context region
            for idx in nbr_sdr.active_indices():
                ctx_idx = int(idx * (SDR_SIZE - BASE_END) / SDR_SIZE)
                if ctx_idx < len(ctx_votes):
                    ctx_votes[ctx_idx] += weight

        top_ctx_local = np.argsort(ctx_votes)[-CTX_ACTIVE:]
        ctx_indices = [CTX_START + int(i) for i in top_ctx_local]

        all_indices = base_indices + ctx_indices
        return SDR.from_indices(all_indices, label=target)

    def encode_sequence_contextual(self, tokens: list[str]) -> list[SDR]:
        """Encode all tokens in a sequence with context. One SDR per token."""
        return [self.encode_in_context(tokens, i) for i in range(len(tokens))]


class ContextualSequenceEncoder:
    """
    Encodes a full sequence as one SDR using context-aware token SDRs.
    Drop-in replacement for SequenceEncoder — same interface, richer output.
    """

    def __init__(self, window: int = CTX_WINDOW, blend: float = 0.25):
        self.ctx_encoder = ContextualTokenEncoder(window=window)
        self.max_len = 64

    def encode(self, text) -> SDR:
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = [str(t) for t in text]

        if not tokens:
            return SDR(bits=np.zeros(SDR_SIZE, dtype=bool))

        token_sdrs = self.ctx_encoder.encode_sequence_contextual(tokens)

        n_active = int(SDR_SIZE * SDR_SPARSITY)
        votes = np.zeros(SDR_SIZE, dtype=np.float32)

        for pos, sdr in enumerate(token_sdrs[:self.max_len]):
            shift = (pos * 3) % SDR_SIZE
            rotated = np.roll(sdr.bits, shift)
            weight = 1.0 / (1.0 + pos * 0.15)
            votes += rotated.astype(np.float32) * weight

        top_indices = np.argsort(votes)[-n_active:]
        label = " ".join(tokens[:4]) + ("..." if len(tokens) > 4 else "")
        return SDR.from_indices(top_indices.tolist(), label=label)

    def similarity(self, a: str, b: str) -> float:
        return self.encode(a).overlap_score(self.encode(b))


def show_context_effect(word: str, context_a: list[str], context_b: list[str]) -> dict:
    """
    Show how context changes a word's SDR.
    Example:
        show_context_effect("bank", ["river","bank","water"], ["bank","account","money"])
    """
    enc = ContextualTokenEncoder()
    base_enc = TokenEncoder(subword_overlap=True)

    try:
        pos_a = context_a.index(word)
        pos_b = context_b.index(word)
    except ValueError:
        return {"error": f"{word!r} not found in one of the context lists"}

    ctx_sdr_a = enc.encode_in_context(context_a, pos_a)
    ctx_sdr_b = enc.encode_in_context(context_b, pos_b)
    base_sdr  = base_enc.encode(word)

    return {
        "word": word,
        "context_a": " ".join(context_a),
        "context_b": " ".join(context_b),
        "ctx_a_vs_ctx_b_similarity": round(ctx_sdr_a.overlap_score(ctx_sdr_b), 3),
        "base_vs_ctx_a": round(base_sdr.overlap_score(ctx_sdr_a), 3),
        "base_vs_ctx_b": round(base_sdr.overlap_score(ctx_sdr_b), 3),
        "context_separates": ctx_sdr_a.overlap_score(ctx_sdr_b) < 0.85,
    }