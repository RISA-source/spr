"""
memory.py — Pattern Memory

This is the "learned weights" equivalent in our symbolic system.
Stores SDR patterns and does fast approximate matching against them.

Key idea: instead of a weight matrix, we have a store of SDRs.
Recognition = find the stored pattern with highest overlap to the input.

Supports:
  - Online learning (show it examples, it updates)
  - Fuzzy recall (noisy input still finds the right pattern)
  - Confidence scoring (how sure is the match)
  - Prototype formation (multiple examples → one representative SDR)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY


@dataclass
class MatchResult:
    """What the matcher returns — not just a label but a full confidence picture."""
    label: str
    score: float              # 0.0 - 1.0 Jaccard overlap
    raw_overlap: int          # actual bit overlap count
    pattern: SDR              # the stored pattern that matched
    input_sdr: SDR            # what was matched against

    @property
    def confident(self) -> bool:
        return self.score >= 0.5

    def __repr__(self):
        bar = "█" * int(self.score * 20) + "░" * (20 - int(self.score * 20))
        return f"Match({self.label!r} |{bar}| {self.score:.2f})"


class PatternMemory:
    """
    Core pattern store. Learns SDR prototypes and matches against them.
    
    Each "concept" is stored as one prototype SDR, built by bundling
    all examples seen for that concept. New examples update the prototype.
    
    This gives us:
    - Generalization (prototype = central tendency of examples)
    - Robustness (noisy input still overlaps with prototype)
    - Online learning (just call learn() with new examples)
    """

    def __init__(self, match_threshold: float = 0.1):
        """
        match_threshold: minimum score to count as a real match.
                        0.1 is deliberately low — let the caller decide
                        what's "good enough" for their use case.
        """
        self.prototypes: dict[str, SDR] = {}       # label → prototype SDR
        self.example_counts: dict[str, int] = {}   # label → n examples seen
        self._example_bank: dict[str, list[SDR]] = {}  # raw examples (for recompute)
        self.match_threshold = match_threshold

    # ------------------------------------------------------------------ #
    #  Learning                                                            #
    # ------------------------------------------------------------------ #

    def learn(self, label: str, sdr: SDR):
        """
        Learn one example. Updates the prototype for this label.
        Incremental — can call many times, prototype keeps improving.
        """
        if label not in self._example_bank:
            self._example_bank[label] = []
            self.example_counts[label] = 0

        self._example_bank[label].append(sdr)
        self.example_counts[label] += 1

        # Recompute prototype by bundling all examples
        # For large counts we could do online bundling but exact is fine for v1
        examples = self._example_bank[label]
        if len(examples) == 1:
            self.prototypes[label] = SDR(bits=examples[0].bits.copy(), label=label)
        else:
            self.prototypes[label] = examples[0].bundle(*examples[1:], label=label)

    def learn_batch(self, label: str, sdrs: list[SDR]):
        for sdr in sdrs:
            self.learn(label, sdr)

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def match(self, input_sdr: SDR) -> Optional[MatchResult]:
        """
        Find best matching pattern for this input.
        Returns None if nothing clears the threshold.
        """
        results = self.match_all(input_sdr)
        if not results:
            return None
        best = results[0]
        return best if best.score >= self.match_threshold else None

    def match_all(self, input_sdr: SDR, top_k: int = 5) -> list[MatchResult]:
        """
        Return top_k matches ranked by score. Useful for:
          - seeing runner-up candidates
          - ambiguity detection (two high scores = uncertain)
          - soft classification (weighted vote over top matches)
        """
        results = []
        for label, proto in self.prototypes.items():
            score = input_sdr.overlap_score(proto)
            raw = input_sdr.overlap(proto)
            results.append(MatchResult(
                label=label,
                score=score,
                raw_overlap=raw,
                pattern=proto,
                input_sdr=input_sdr
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def is_novel(self, input_sdr: SDR, novelty_threshold: float = 0.3) -> bool:
        """
        Does this input NOT match anything well?
        This is how the system knows when to learn a new pattern vs
        update an existing one. Crucial for unsupervised learning.
        """
        best = self.match(input_sdr)
        if best is None:
            return True
        return best.score < novelty_threshold

    def ambiguous(self, input_sdr: SDR, gap_threshold: float = 0.1) -> bool:
        """
        Are the top two matches too close to call?
        Useful for detecting inputs that sit between categories.
        """
        results = self.match_all(input_sdr, top_k=2)
        if len(results) < 2:
            return False
        return (results[0].score - results[1].score) < gap_threshold

    # ------------------------------------------------------------------ #
    #  Inspection                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            label: {
                "examples": self.example_counts[label],
                "active_bits": self.prototypes[label].n_active()
            }
            for label in self.prototypes
        }

    def __len__(self):
        return len(self.prototypes)

    def __repr__(self):
        return f"PatternMemory({len(self)} patterns, threshold={self.match_threshold})"
