"""
sym_pattern — Symbolic Pattern Recognition System v2.2

Changelog:
  v1.0 — SDR core, fuzzy matching, subword similarity, novelty detection
  v1.1 — Save/load patterns to .npz / .json
  v1.2 — File learning, file similarity comparison
  v1.3 — Continuous strengthening, candidate decay, confidence tracking, on_the_go mode
  v2.0 — Hierarchical composition: stacked pattern layers (the big one)
  v2.1 — Temporal memory: transition graph, sequence patterns, predictions, surprise detection
  v2.2 — Context-sensitive encoding: same word → different SDR in different contexts
"""

from .sdr import SDR, SDR_SIZE, SDR_SPARSITY
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, ScalarEncoder, SequenceEncoder, CompositeEncoder
from .learner import UnsupervisedLearner
from .hierarchy import PatternLayer, HierarchicalSystem, HierarchicalResult, LayerActivation
from .temporal import TemporalMemory, TransitionGraph, SequencePatternMemory, TemporalResult
from .context import ContextualTokenEncoder, ContextualSequenceEncoder, show_context_effect
from . import persistence
from . import fileops


class SymPattern:
    """
    High-level API. One object, all capabilities.

    mode="flat"         — v1 behavior: single layer, fast, simple
    mode="hierarchical" — v2.0: stacked layers, builds abstraction
    mode="temporal"     — v2.1: flat + time-aware, tracks sequences
    mode="full"         — v2.2: hierarchy + temporal + context-sensitive encoding
    """

    def __init__(
        self,
        mode: str = "flat",
        unsupervised: bool = False,
        subword_overlap: bool = True,
        novelty_threshold: float = 0.20,  # lowered from 0.25 — better recall without sacrificing rejection
        promotion_threshold: int = 3,
        on_the_go: bool = False,
        n_layers: int = 3,
        context_window: int = 2,
        context_blend: float = 0.25,
    ):
        self.mode = mode

        # Choose encoder based on mode
        if mode in ("full", "contextual"):
            from .context import ContextualSequenceEncoder
            self.encoder = ContextualSequenceEncoder(
                window=context_window, blend=context_blend
            )
        else:
            self.encoder = SequenceEncoder(
                token_encoder=TokenEncoder(subword_overlap=subword_overlap)
            )

        # Core flat memory (always present)
        self.memory = PatternMemory(match_threshold=novelty_threshold)
        self.learner = UnsupervisedLearner(
            memory=self.memory,
            novelty_threshold=novelty_threshold,
            promotion_threshold=promotion_threshold,
            on_the_go=on_the_go,
        )

        # Hierarchical system (v2.0)
        # Layer 0 shares the flat PatternMemory so teach() works for both.
        self.hierarchy: HierarchicalSystem | None = None
        if mode in ("hierarchical", "full"):
            self.hierarchy = HierarchicalSystem(
                n_layers=n_layers,
                on_the_go=on_the_go,
            )
            # Wire layer 0 to use the same memory as the flat system.
            self.hierarchy.layers[0].memory = self.memory
            self.hierarchy.layers[0].learner.memory = self.memory

        # Temporal memory (v2.1)
        self.temporal: TemporalMemory | None = None
        if mode in ("temporal", "full"):
            self.temporal = TemporalMemory(
                pattern_memory=self.memory,
            )

        self.unsupervised = unsupervised

    # ------------------------------------------------------------------ #
    #  Core API (all modes)                                               #
    # ------------------------------------------------------------------ #

    def teach(self, label: str, examples: list[str]):
        """Teach labeled examples. Works in all modes."""
        for example in examples:
            sdr = self.encoder.encode(example)
            self.learner.teach(label, sdr)
            if self.hierarchy:
                self.hierarchy.layers[0].teach(label, sdr)
        return self

    def observe(self, text: str, label: str = None) -> MatchResult | None:
        sdr = self.encoder.encode(text)
        result = self.learner.observe(sdr, label=label)
        if self.temporal:
            self.temporal.step(sdr)
        if self.hierarchy:
            self.hierarchy.process(sdr)
        return result

    def observe_corpus(self, texts: list[str]) -> list[MatchResult | None]:
        return [self.observe(t) for t in texts]

    def recognize(self, text: str, top_k: int = 3, threshold: float = None) -> list[MatchResult]:
        """
        Recognize text against all learned patterns.
        threshold: minimum score to include in results. Defaults to memory.match_threshold.
                   Set to 0.0 to get all results regardless of score.
        """
        sdr = self.encoder.encode(text)
        results = self.learner.recognize(sdr, top_k=top_k)
        floor = threshold if threshold is not None else self.memory.match_threshold
        return [r for r in results if r.score >= floor]

    def best_match(self, text: str) -> MatchResult | None:
        results = self.recognize(text, top_k=1)
        return results[0] if results and results[0].score >= self.memory.match_threshold else None

    def is_novel(self, text: str) -> bool:
        return self.memory.is_novel(self.encoder.encode(text))

    def is_ambiguous(self, text: str) -> bool:
        return self.memory.ambiguous(self.encoder.encode(text))

    def encode(self, text: str) -> SDR:
        return self.encoder.encode(text)

    def similarity(self, a: str, b: str) -> float:
        return self.encoder.encode(a).overlap_score(self.encoder.encode(b))

    # ------------------------------------------------------------------ #
    #  v2.0 — Hierarchical API                                            #
    # ------------------------------------------------------------------ #

    def process_hierarchical(self, text: str, learn: bool = True) -> HierarchicalResult | None:
        """Process text through all hierarchy layers. Requires mode='hierarchical' or 'full'."""
        if not self.hierarchy:
            raise RuntimeError("Hierarchical mode not enabled. Use mode='hierarchical' or 'full'.")
        sdr = self.encoder.encode(text)
        return self.hierarchy.process(sdr, learn=learn)

    def hierarchy_summary(self) -> str:
        if not self.hierarchy:
            return "No hierarchy (use mode='hierarchical' or 'full')"
        return self.hierarchy.summary()

    # ------------------------------------------------------------------ #
    #  v2.1 — Temporal API                                                #
    # ------------------------------------------------------------------ #

    def predict_next(self, top_k: int = 3) -> list[tuple[str, float]]:
        """What pattern does the system expect next?"""
        if not self.temporal:
            raise RuntimeError("Temporal mode not enabled. Use mode='temporal' or 'full'.")
        return self.temporal.predict_next(top_k=top_k)

    def recent_surprises(self) -> list[tuple[str, str, float]]:
        """Get surprisingly unexpected transitions seen recently."""
        if not self.temporal:
            return []
        return self.temporal.recent_surprises()

    def teach_sequence(self, seq_label: str, label_sequences: list[list[str]]):
        """Teach named multi-step sequence patterns."""
        if not self.temporal:
            raise RuntimeError("Temporal mode not enabled.")
        self.temporal.teach_sequence(seq_label, label_sequences)

    # ------------------------------------------------------------------ #
    #  v1.1 — Persistence                                                 #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        persistence.save(self.memory, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "SymPattern":
        sp = cls(**kwargs)
        sp.memory = persistence.load(path)
        sp.learner.memory = sp.memory
        if sp.temporal:
            sp.temporal.pattern_memory = sp.memory
        return sp

    # ------------------------------------------------------------------ #
    #  v1.2 — File ops                                                    #
    # ------------------------------------------------------------------ #

    def learn_file(self, path: str, label: str = None, chunk_size: int = 50) -> dict:
        return fileops.learn_from_file(self, path, label=label, chunk_size=chunk_size)

    def compare_files(self, path_a: str, path_b: str, chunk_size: int = 50):
        return fileops.compare_files(self, path_a, path_b, chunk_size=chunk_size)

    # ------------------------------------------------------------------ #
    #  v1.3 — Confidence                                                  #
    # ------------------------------------------------------------------ #

    def confidence_report(self) -> dict:
        return self.learner.confidence_report()

    # ------------------------------------------------------------------ #
    #  Inspection                                                         #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        s = {
            "mode": self.mode,
            "learner": self.learner.stats(),
            "memory": self.memory.stats(),
        }
        if self.hierarchy:
            s["hierarchy"] = self.hierarchy.stats()
        if self.temporal:
            s["temporal"] = self.temporal.stats()
        return s

    def __repr__(self):
        return f"SymPattern(mode={self.mode!r}, patterns={len(self.memory)}, {self.learner})"


__all__ = [
    "SymPattern",
    "SDR", "SDR_SIZE", "SDR_SPARSITY",
    "PatternMemory", "MatchResult",
    "TokenEncoder", "ScalarEncoder", "SequenceEncoder", "CompositeEncoder",
    "UnsupervisedLearner",
    "PatternLayer", "HierarchicalSystem", "HierarchicalResult", "LayerActivation",
    "TemporalMemory", "TransitionGraph", "SequencePatternMemory", "TemporalResult",
    "ContextualTokenEncoder", "ContextualSequenceEncoder", "show_context_effect",
    "persistence", "fileops",
]