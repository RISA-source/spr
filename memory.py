"""
memory.py — Pattern Memory with Structural Description (v3.1)

The fundamental upgrade from v1-v3.0:

Before: each pattern was one opaque prototype SDR.
  - You could ask "does this match?" (score 0-1)
  - You could NOT ask "why does this match?" or "what defines this pattern?"

After: each pattern is a StructuredPrototype with four bit strata:
  - core_bits:       present in ≥80% of examples  → necessary features
  - typical_bits:    present in 40-79%             → characteristic but not required
  - peripheral_bits: present in 10-39%             → occasional/contextual
  - forbidden_bits:  never in this pattern, common in others → exclusion features

This makes matching EXPLAINABLE:
  "Matched 'cat' because:
    core features present (0.94)
    typical features present (0.87)
    no forbidden features triggered (0.00)
    → explanation: 'cat_token', 'mat_token', 'sat_token'"

And it makes patterns INSPECTABLE:
  "What defines 'greeting'?
    Always: hello_ngrams, how_ngrams (core)
    Usually: friend_token, morning_token (typical)
    Never: technical_jargon (forbidden)"

The structured prototype is computed once from all examples and updated
incrementally as new examples arrive. Backward compatible — all existing
matching APIs still work, they just use the richer prototype internally.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY


# ------------------------------------------------------------------ #
#  Bit frequency thresholds                                           #
# ------------------------------------------------------------------ #

# Adaptive thresholds — scale with number of examples.
# With few examples, a "core" bit needs to appear in most of them.
# Formula: core_threshold = max(0.50, 1.0 - 1.0/sqrt(n_examples))
# Typical and peripheral are fractions of the core threshold.
FORBIDDEN_CROSS_MIN  = 0.35   # bit must appear in ≥35% of OTHER patterns to be forbidden


def _compute_strata(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Compute (core, typical, peripheral, is_grounded) bit index arrays from per-bit frequencies.

    Two modes:
      Frequency-grounded (max_freq >= 0.50): uses hard frequency thresholds.
        Core     = freq >= 0.75  (appears in 75%+ of examples — genuinely necessary)
        Typical  = freq in [0.40, 0.75) (characteristic but not always present)
        Peripheral = freq in [0.15, 0.40) (occasional or contextual)

      Rank-based (max_freq < 0.50): examples too diverse for hard thresholds.
        Splits active bits into proportional buckets by rank.
        Marked is_grounded=False — caller knows these are weak claims.

    The distinction is important: a pattern with core bits freq=0.90 is genuinely
    well-defined. A pattern with "core" bits freq=0.39 is just the least-bad bits.
    """
    max_freq = freqs.max()

    if max_freq >= 0.50:
        # Hard frequency bands — meaningful claims
        core    = np.where(freqs >= 0.75)[0]
        typical = np.where((freqs >= 0.40) & (freqs < 0.75))[0]
        periph  = np.where((freqs >= 0.15) & (freqs < 0.40))[0]
        return core, typical, periph, True
    else:
        # Rank-based fallback — low-confidence structural claims
        active_mask    = freqs > 0
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, empty, False
        active_freqs = freqs[active_mask]
        sorted_idx   = np.argsort(active_freqs)[::-1]
        n_active     = len(active_indices)
        n_core    = max(1, n_active // 5)
        n_typical = max(1, n_active // 4)
        n_periph  = max(1, n_active // 3)
        core    = active_indices[sorted_idx[:n_core]]
        typical = active_indices[sorted_idx[n_core:n_core + n_typical]]
        periph  = active_indices[sorted_idx[n_core + n_typical:n_core + n_typical + n_periph]]
        return core, typical, periph, False


# ------------------------------------------------------------------ #
#  Structured Prototype                                                #
# ------------------------------------------------------------------ #

@dataclass
class StructuredPrototype:
    """
    A pattern's full structural description.

    Not just one SDR — a layered view of what the pattern consists of,
    derived from statistical analysis of all examples seen.

    The matching_sdr is the majority-vote bundle used for fast matching.
    The strata are used for explanation and refined scoring.
    """
    label: str
    matching_sdr: SDR                    # majority-vote bundle (backward compat)
    bit_frequencies: np.ndarray          # per-bit frequency across all examples [0.0-1.0]

    core_bits:       np.ndarray          # indices of core bits (≥80% frequency)
    typical_bits:    np.ndarray          # indices of typical bits (40-79%)
    peripheral_bits: np.ndarray          # indices of peripheral bits (10-39%)
    forbidden_bits:  np.ndarray          # indices of forbidden bits (0% here, common elsewhere)

    example_count: int = 0
    is_grounded: bool = True    # False = low-confidence strata (diverse examples, small n)

    @property
    def bits(self) -> np.ndarray:
        """Backward compat: expose matching SDR's bits."""
        return self.matching_sdr.bits

    def n_active(self) -> int:
        return self.matching_sdr.n_active()

    def active_indices(self) -> np.ndarray:
        return self.matching_sdr.active_indices()

    def explain(self) -> dict:
        """Human-readable structural summary."""
        return {
            "label":           self.label,
            "examples":        self.example_count,
            "core_bits":       len(self.core_bits),
            "typical_bits":    len(self.typical_bits),
            "peripheral_bits": len(self.peripheral_bits),
            "forbidden_bits":  len(self.forbidden_bits),
            "sparsity":        f"{self.n_active()}/{SDR_SIZE}",
            "grounded":        self.is_grounded,
        }

    def __repr__(self):
        return (
            f"StructuredPrototype({self.label!r}, "
            f"core={len(self.core_bits)}, "
            f"typical={len(self.typical_bits)}, "
            f"peripheral={len(self.peripheral_bits)}, "
            f"forbidden={len(self.forbidden_bits)}, "
            f"n={self.example_count})"
        )


def _build_prototype(label: str, example_sdrs: list[SDR]) -> StructuredPrototype:
    """
    Build a StructuredPrototype from a list of example SDRs.
    Called whenever the example bank changes.
    """
    n = len(example_sdrs)
    assert n > 0

    # Per-bit frequency
    freqs = np.zeros(SDR_SIZE, dtype=np.float32)
    for sdr in example_sdrs:
        freqs += sdr.bits.astype(np.float32)
    freqs /= n

    # Compute strata using hybrid frequency/rank approach
    core, typical, peripheral, is_grounded = _compute_strata(freqs)
    # forbidden computed later when we know other patterns' distributions
    forbidden  = np.array([], dtype=np.int64)

    # Majority-vote matching SDR: bits active in >50% of examples
    majority_bits = freqs >= 0.5
    # Ensure we have roughly SDR_SPARSITY active bits
    n_target = int(SDR_SIZE * SDR_SPARSITY)
    if majority_bits.sum() < n_target:
        # Not enough majority bits — take top-N by frequency
        top_indices = np.argsort(freqs)[-n_target:]
        majority_bits = np.zeros(SDR_SIZE, dtype=bool)
        majority_bits[top_indices] = True

    matching_sdr = SDR(bits=majority_bits, label=label)

    return StructuredPrototype(
        label=label,
        matching_sdr=matching_sdr,
        bit_frequencies=freqs,
        core_bits=core,
        typical_bits=typical,
        peripheral_bits=peripheral,
        forbidden_bits=forbidden,
        example_count=n,
        is_grounded=is_grounded,
    )


# ------------------------------------------------------------------ #
#  Explained Match Result                                             #
# ------------------------------------------------------------------ #

@dataclass
class MatchExplanation:
    """
    Why did a match happen?
    Breaks the score down by feature stratum.
    """
    core_hit:       float    # fraction of core bits present in input
    typical_hit:    float    # fraction of typical bits present in input
    peripheral_hit: float    # fraction of peripheral bits present in input
    forbidden_hit:  float    # fraction of forbidden bits present (LOWER = better)
    n_core:         int
    n_typical:      int
    n_forbidden:    int

    @property
    def clean(self) -> bool:
        """No forbidden features fired — this is a clean match."""
        return self.forbidden_hit < 0.05

    @property
    def strength(self) -> str:
        if self.core_hit >= 0.8 and self.clean:
            return "strong"
        if self.core_hit >= 0.5:
            return "moderate"
        return "weak"

    def summary(self) -> str:
        bars = {
            "core":     "█" * int(self.core_hit * 10),
            "typical":  "█" * int(self.typical_hit * 10),
            "forbidden":"█" * int(self.forbidden_hit * 10),
        }
        return (
            f"core={self.core_hit:.2f}|{bars['core']:<10}  "
            f"typical={self.typical_hit:.2f}|{bars['typical']:<10}  "
            f"forbidden={self.forbidden_hit:.2f}|{bars['forbidden']:<10}  "
            f"[{self.strength}]"
        )

    def __repr__(self):
        return f"Explanation(core={self.core_hit:.2f}, typical={self.typical_hit:.2f}, forbidden={self.forbidden_hit:.2f}, {self.strength})"


@dataclass
class MatchResult:
    """
    What the matcher returns.

    v3.1 upgrade: now includes a MatchExplanation — the WHY behind the score.
    Backward compatible: score, label, raw_overlap all still present.
    """
    label: str
    score: float              # overall overlap score (backward compat)
    raw_overlap: int          # actual bit overlap count
    pattern: SDR              # the matching SDR (backward compat)
    input_sdr: SDR            # what was matched against
    explanation: Optional[MatchExplanation] = None   # v3.1: WHY it matched

    @property
    def confident(self) -> bool:
        return self.score >= 0.5

    @property
    def explained(self) -> bool:
        return self.explanation is not None

    def why(self) -> str:
        """Human-readable explanation of the match."""
        if self.explanation is None:
            return f"Match({self.label!r}, score={self.score:.2f}) [no explanation]"
        return (
            f"Match({self.label!r}, score={self.score:.2f})\n"
            f"  {self.explanation.summary()}"
        )

    def __repr__(self):
        bar = "█" * int(self.score * 20) + "░" * (20 - int(self.score * 20))
        strength = f" [{self.explanation.strength}]" if self.explanation else ""
        return f"Match({self.label!r} |{bar}| {self.score:.2f}{strength})"


# ------------------------------------------------------------------ #
#  Pattern Memory                                                      #
# ------------------------------------------------------------------ #

class PatternMemory:
    """
    Core pattern store with structural prototype descriptions.

    v3.1 upgrade: each pattern is a StructuredPrototype, not a flat SDR.
    Matching is now explainable — you can ask WHY something matched.
    Patterns are inspectable — you can ask WHAT DEFINES each pattern.

    Fully backward compatible: all v1-v3.0 APIs still work.
    """

    def __init__(self, match_threshold: float = 0.1):
        self.match_threshold = match_threshold

        # v3.1: structured prototypes replace flat SDR prototypes
        self._structured: dict[str, StructuredPrototype] = {}

        # Backward compat shim: .prototypes[label] returns the matching SDR
        # (not the full StructuredPrototype) so old code still works
        self.prototypes: _PrototypeProxy = _PrototypeProxy(self._structured)

        self.example_counts: dict[str, int] = {}
        self._example_bank: dict[str, list[SDR]] = {}

    # ------------------------------------------------------------------ #
    #  Learning                                                            #
    # ------------------------------------------------------------------ #

    def learn(self, label: str, sdr: SDR):
        """
        Learn one example. Rebuilds the StructuredPrototype for this label.
        Also recomputes forbidden bits across all patterns.
        """
        if label not in self._example_bank:
            self._example_bank[label] = []
            self.example_counts[label] = 0

        self._example_bank[label].append(sdr)
        self.example_counts[label] += 1

        # Rebuild this pattern's structured prototype
        self._structured[label] = _build_prototype(label, self._example_bank[label])

        # Recompute forbidden bits across all patterns
        # (only if we have 2+ patterns — forbidden needs cross-pattern comparison)
        if len(self._structured) >= 2:
            self._recompute_forbidden()

    def learn_batch(self, label: str, sdrs: list[SDR]):
        for sdr in sdrs:
            self.learn(label, sdr)

    def _recompute_forbidden(self):
        """
        Forbidden bits: bits that NEVER appear in pattern P
        but appear frequently across other patterns.

        These are the most diagnostically useful bits for rejection —
        if a forbidden bit fires, the input probably isn't this pattern.
        """
        # For each label: compute the "other" frequency = mean freq across all other patterns
        labels = list(self._structured.keys())

        for label in labels:
            proto = self._structured[label]
            this_freq = proto.bit_frequencies

            # Mean frequency across all other patterns
            other_freqs = [
                self._structured[l].bit_frequencies
                for l in labels if l != label
            ]
            if not other_freqs:
                continue
            other_mean = np.mean(other_freqs, axis=0)

            # Forbidden: never in this pattern (freq=0) AND common elsewhere (≥threshold)
            forbidden = np.where(
                (this_freq == 0.0) & (other_mean >= FORBIDDEN_CROSS_MIN)
            )[0]
            proto.forbidden_bits = forbidden

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def match(self, input_sdr: SDR) -> Optional[MatchResult]:
        """Best match, or None if below threshold."""
        results = self.match_all(input_sdr)
        if not results:
            return None
        best = results[0]
        return best if best.score >= self.match_threshold else None

    def match_all(self, input_sdr: SDR, top_k: int = 5) -> list[MatchResult]:
        """
        Return top_k matches ranked by score.
        Each result now includes a MatchExplanation.
        """
        results = []
        for label, proto in self._structured.items():
            score = input_sdr.overlap_score(proto.matching_sdr)
            raw   = input_sdr.overlap(proto.matching_sdr)
            explanation = self._explain(input_sdr, proto)
            results.append(MatchResult(
                label=label,
                score=score,
                raw_overlap=raw,
                pattern=proto.matching_sdr,
                input_sdr=input_sdr,
                explanation=explanation,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _explain(self, input_sdr: SDR, proto: StructuredPrototype) -> MatchExplanation:
        """
        Compute a MatchExplanation: how well each feature stratum was hit.
        """
        inp = input_sdr.bits

        def hit_rate(indices):
            if len(indices) == 0:
                return 0.0
            return float(np.sum(inp[indices])) / len(indices)

        return MatchExplanation(
            core_hit=       hit_rate(proto.core_bits),
            typical_hit=    hit_rate(proto.typical_bits),
            peripheral_hit= hit_rate(proto.peripheral_bits),
            forbidden_hit=  hit_rate(proto.forbidden_bits),
            n_core=         len(proto.core_bits),
            n_typical=      len(proto.typical_bits),
            n_forbidden=    len(proto.forbidden_bits),
        )

    # ------------------------------------------------------------------ #
    #  Introspection — what defines each pattern?                         #
    # ------------------------------------------------------------------ #

    def describe(self, label: str) -> Optional[dict]:
        """
        Full structural description of a pattern.
        This is what makes the system symbolic — you can inspect what defines
        any learned pattern, not just match against it.
        """
        if label not in self._structured:
            return None
        proto = self._structured[label]
        return {
            "label":           label,
            "examples":        proto.example_count,
            "core_bits":       len(proto.core_bits),
            "typical_bits":    len(proto.typical_bits),
            "peripheral_bits": len(proto.peripheral_bits),
            "forbidden_bits":  len(proto.forbidden_bits),
            "core_frequency":  float(np.mean(proto.bit_frequencies[proto.core_bits])) if len(proto.core_bits) else 0.0,
            "defining_ratio":  (len(proto.core_bits) + len(proto.typical_bits)) / max(1, int(SDR_SIZE * SDR_SPARSITY)),
            "grounded":        proto.is_grounded,
        }

    def describe_all(self) -> dict[str, dict]:
        """Structural description for every known pattern."""
        return {label: self.describe(label) for label in self._structured}

    def contrast(self, label_a: str, label_b: str) -> Optional[dict]:
        """
        What distinguishes pattern A from pattern B?
        Returns bits that are core to A but forbidden in B (and vice versa).
        These are the diagnostic features — the ones that definitively separate the two.
        """
        if label_a not in self._structured or label_b not in self._structured:
            return None
        a = self._structured[label_a]
        b = self._structured[label_b]

        # Bits that are core to A AND forbidden in B → definitively A not B
        a_core_set = set(a.core_bits.tolist())
        b_forbidden_set = set(b.forbidden_bits.tolist())
        a_distinguishing = list(a_core_set & b_forbidden_set)

        b_core_set = set(b.core_bits.tolist())
        a_forbidden_set = set(a.forbidden_bits.tolist())
        b_distinguishing = list(b_core_set & a_forbidden_set)

        # Overlap: bits core to both → shared features
        shared_core = list(a_core_set & b_core_set)

        return {
            f"{label_a}_distinguishing": len(a_distinguishing),
            f"{label_b}_distinguishing": len(b_distinguishing),
            "shared_core":               len(shared_core),
            "separability":              (len(a_distinguishing) + len(b_distinguishing)) / max(1, len(shared_core) + 1),
        }

    def get_structured(self, label: str) -> Optional[StructuredPrototype]:
        """Get the full StructuredPrototype for a label."""
        return self._structured.get(label)

    # ------------------------------------------------------------------ #
    #  Novelty / ambiguity (unchanged)                                     #
    # ------------------------------------------------------------------ #

    def is_novel(self, input_sdr: SDR, novelty_threshold: float = 0.3) -> bool:
        best = self.match(input_sdr)
        if best is None:
            return True
        return best.score < novelty_threshold

    def ambiguous(self, input_sdr: SDR, gap_threshold: float = 0.1) -> bool:
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
                "examples":   self.example_counts[label],
                "active_bits": self._structured[label].n_active(),
                "core_bits":   len(self._structured[label].core_bits),
                "typical_bits": len(self._structured[label].typical_bits),
                "forbidden_bits": len(self._structured[label].forbidden_bits),
            }
            for label in self._structured
        }

    def __len__(self):
        return len(self._structured)

    def __repr__(self):
        return f"PatternMemory({len(self)} patterns, threshold={self.match_threshold})"


# ------------------------------------------------------------------ #
#  Backward compatibility shim                                         #
# ------------------------------------------------------------------ #

class _PrototypeProxy:
    """
    Makes memory.prototypes[label] return the matching SDR (not StructuredPrototype).
    All code that did: proto = memory.prototypes[label]; proto.bits / proto.overlap(...)
    continues to work unchanged.
    """
    def __init__(self, structured: dict):
        self._structured = structured

    def __getitem__(self, label: str) -> SDR:
        return self._structured[label].matching_sdr

    def __setitem__(self, label: str, value):
        # Support direct assignment (used in persistence.py load)
        if isinstance(value, SDR):
            if label in self._structured:
                self._structured[label].matching_sdr = value
            # else: will be created properly by learn()
        elif isinstance(value, StructuredPrototype):
            self._structured[label] = value

    def __contains__(self, label: str) -> bool:
        return label in self._structured

    def __iter__(self):
        return iter(self._structured)

    def items(self):
        return {k: self._structured[k].matching_sdr for k in self._structured}.items()

    def keys(self):
        return self._structured.keys()

    def values(self):
        return [v.matching_sdr for v in self._structured.values()]

    def get(self, label: str, default=None):
        if label in self._structured:
            return self._structured[label].matching_sdr
        return default