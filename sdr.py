"""
sdr.py — Sparse Distributed Representations

The foundation of sym-pattern. Every symbol, token, pattern is encoded
as a sparse binary vector across a large dimension space.

Why sparse? Because:
  - overlap = similarity (naturally)
  - robust to noise (a few wrong bits don't tank the match)
  - composable (union/intersection of bit sets = combine/constrain)
  - inspectable (you can literally look at what bits are active)

This is how Numenta's HTM works, and it's a big reason biological
neurons use sparse coding too.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Global SDR config — tune these for your domain
SDR_SIZE = 1024       # total dimensions
SDR_SPARSITY = 0.05   # fraction of bits active (5% = ~51 bits on)


@dataclass
class SDR:
    """
    A Sparse Distributed Representation.
    
    Internally stored as a dense bool array for fast numpy ops,
    but logically it's a set of active indices in a large space.
    """
    bits: np.ndarray          # shape: (SDR_SIZE,), dtype bool
    label: Optional[str] = None   # human-readable name, optional

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_indices(cls, indices: list[int], label=None) -> "SDR":
        bits = np.zeros(SDR_SIZE, dtype=bool)
        bits[indices] = True
        return cls(bits=bits, label=label)

    @classmethod
    def random(cls, label=None, rng=None) -> "SDR":
        """Create a random SDR with the global sparsity level."""
        rng = rng or np.random.default_rng()
        n_active = int(SDR_SIZE * SDR_SPARSITY)
        indices = rng.choice(SDR_SIZE, size=n_active, replace=False)
        return cls.from_indices(indices.tolist(), label=label)

    @classmethod
    def from_hash(cls, value: str, label=None) -> "SDR":
        """
        Deterministic SDR from a string — same string always gives same SDR.
        Uses hash-based index selection so no training needed for basic symbols.
        """
        import hashlib
        n_active = int(SDR_SIZE * SDR_SPARSITY)
        indices = set()
        seed = 0
        while len(indices) < n_active:
            h = hashlib.sha256(f"{value}:{seed}".encode()).digest()
            for i in range(0, len(h) - 1, 2):
                idx = int.from_bytes(h[i:i+2], "big") % SDR_SIZE
                indices.add(idx)
                if len(indices) >= n_active:
                    break
            seed += 1
        return cls.from_indices(list(indices)[:n_active], label=label)

    # ------------------------------------------------------------------ #
    #  Core operations                                                     #
    # ------------------------------------------------------------------ #

    def active_indices(self) -> np.ndarray:
        return np.where(self.bits)[0]

    def n_active(self) -> int:
        return int(self.bits.sum())

    def overlap(self, other: "SDR") -> int:
        """Number of bits both SDRs share. The raw similarity measure."""
        return int(np.logical_and(self.bits, other.bits).sum())

    def overlap_score(self, other: "SDR") -> float:
        """
        Normalized overlap: 0.0 = nothing in common, 1.0 = identical.
        Normalizes by the smaller (more specific) SDR — so a focused query
        matching a broad prototype gets a fair score, not penalized for the
        prototype being wide. This is the right metric for recognition:
        "what fraction of my query bits does the pattern cover?"
        """
        intersection = int(np.logical_and(self.bits, other.bits).sum())
        denom = min(int(self.bits.sum()), int(other.bits.sum()))
        if denom == 0:
            return 0.0
        return float(intersection / denom)

    def subsumes(self, other: "SDR", threshold: float = 0.8) -> bool:
        """Does self 'contain' other? Useful for hierarchical matching."""
        if other.n_active() == 0:
            return True
        overlap = np.logical_and(self.bits, other.bits).sum()
        return float(overlap / other.n_active()) >= threshold

    # ------------------------------------------------------------------ #
    #  Composition                                                         #
    # ------------------------------------------------------------------ #

    def union(self, other: "SDR", label=None) -> "SDR":
        """OR — combines two patterns. More general."""
        return SDR(bits=np.logical_or(self.bits, other.bits), label=label)

    def intersection(self, other: "SDR", label=None) -> "SDR":
        """AND — what two patterns share. More specific."""
        return SDR(bits=np.logical_and(self.bits, other.bits), label=label)

    def add_noise(self, flip_rate: float = 0.05, rng=None) -> "SDR":
        """
        Corrupt this SDR slightly. Used to test robustness of matching.
        flip_rate: fraction of active bits to randomly move.
        """
        rng = rng or np.random.default_rng()
        new_bits = self.bits.copy()
        active = np.where(self.bits)[0]
        n_flip = max(1, int(len(active) * flip_rate))
        # turn off some active bits
        to_off = rng.choice(active, size=n_flip, replace=False)
        new_bits[to_off] = False
        # turn on random inactive bits instead
        inactive = np.where(~self.bits)[0]
        to_on = rng.choice(inactive, size=n_flip, replace=False)
        new_bits[to_on] = True
        return SDR(bits=new_bits, label=self.label)

    def bundle(self, *others: "SDR", label=None) -> "SDR":
        """
        Bundle multiple SDRs into one representative SDR.
        Majority vote per bit — like an average but stays binary/sparse.
        This is the symbolic equivalent of averaging embeddings.
        """
        all_sdrs = [self] + list(others)
        stack = np.stack([s.bits for s in all_sdrs], axis=0)
        votes = stack.sum(axis=0)
        threshold = len(all_sdrs) / 2
        # keep top SDR_SPARSITY% of most-voted bits
        n_active = int(SDR_SIZE * SDR_SPARSITY)
        top_indices = np.argsort(votes)[-n_active:]
        new_bits = np.zeros(SDR_SIZE, dtype=bool)
        new_bits[top_indices] = True
        return SDR(bits=new_bits, label=label)

    # ------------------------------------------------------------------ #
    #  Utils                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        name = f'"{self.label}" ' if self.label else ""
        return f"SDR({name}active={self.n_active()}/{SDR_SIZE})"

    def __eq__(self, other):
        return np.array_equal(self.bits, other.bits)

    def __hash__(self):
        return hash(self.bits.tobytes())
