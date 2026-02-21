"""
sym_pattern — Symbolic Pattern Recognition System v1

A symbolic AI pattern recognition system with NN-grade matching capability.

Core components:
  - SDR:               Sparse Distributed Representation (the atom)
  - TokenEncoder:      text → SDR
  - ScalarEncoder:     numbers → SDR  
  - SequenceEncoder:   ordered sequences → SDR
  - CompositeEncoder:  combine multiple encoders
  - PatternMemory:     stores and matches patterns with confidence scores
  - UnsupervisedLearner: learns patterns from raw input automatically

Quickstart:

    from sym_pattern import SymPattern

    sp = SymPattern()

    # Teach it some patterns
    sp.teach("greeting", ["hello world", "hi there", "hey how are you"])
    sp.teach("farewell", ["goodbye", "see you later", "bye bye"])

    # Recognize
    result = sp.recognize("hello!")
    print(result)   # Match('greeting' | confidence 0.73)

    # Or let it learn unsupervised
    sp2 = SymPattern(unsupervised=True)
    for text in my_corpus:
        sp2.observe(text)
    # Patterns emerge automatically from recurring structures
"""

from .sdr import SDR, SDR_SIZE, SDR_SPARSITY
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, ScalarEncoder, SequenceEncoder, CompositeEncoder
from .learner import UnsupervisedLearner


class SymPattern:
    """
    High-level API. One object to rule them all.
    
    Wraps encoder + memory + learner into a simple interface.
    Works in two modes:
      - supervised: you provide labels (sp.teach("cat", examples))
      - unsupervised: it discovers patterns on its own (sp.observe(input))
      - mixed: both at once, which is often most powerful
    """

    def __init__(
        self,
        unsupervised: bool = False,
        subword_overlap: bool = True,
        novelty_threshold: float = 0.25,
        promotion_threshold: int = 3,
    ):
        self.encoder = SequenceEncoder(
            token_encoder=TokenEncoder(subword_overlap=subword_overlap)
        )
        self.memory = PatternMemory(match_threshold=novelty_threshold)
        self.learner = UnsupervisedLearner(
            memory=self.memory,
            novelty_threshold=novelty_threshold,
            promotion_threshold=promotion_threshold,
        )
        self.unsupervised = unsupervised

    # ------------------------------------------------------------------ #
    #  Supervised API                                                      #
    # ------------------------------------------------------------------ #

    def teach(self, label: str, examples: list[str]):
        """
        Teach the system a named pattern with one or more text examples.
        Multiple examples → more robust prototype.
        """
        for example in examples:
            sdr = self.encoder.encode(example)
            self.learner.teach(label, sdr)
        return self

    # ------------------------------------------------------------------ #
    #  Unsupervised API                                                    #
    # ------------------------------------------------------------------ #

    def observe(self, text: str, label: str = None) -> MatchResult | None:
        """
        Show the system one input. Let it decide if it's familiar or novel.
        In unsupervised mode, patterns crystallize automatically.
        """
        sdr = self.encoder.encode(text)
        return self.learner.observe(sdr, label=label)

    def observe_corpus(self, texts: list[str]) -> list[MatchResult | None]:
        """Feed a whole corpus. Good for bootstrapping pattern discovery."""
        return [self.observe(t) for t in texts]

    # ------------------------------------------------------------------ #
    #  Recognition API                                                     #
    # ------------------------------------------------------------------ #

    def recognize(self, text: str, top_k: int = 3) -> list[MatchResult]:
        """
        Recognize text against all learned patterns.
        Returns top_k matches ranked by confidence.
        """
        sdr = self.encoder.encode(text)
        return self.learner.recognize(sdr, top_k=top_k)

    def best_match(self, text: str) -> MatchResult | None:
        """Single best match, or None if nothing fits."""
        results = self.recognize(text, top_k=1)
        return results[0] if results and results[0].score >= self.memory.match_threshold else None

    def is_novel(self, text: str) -> bool:
        """Has the system seen anything like this before?"""
        sdr = self.encoder.encode(text)
        return self.memory.is_novel(sdr)

    def is_ambiguous(self, text: str) -> bool:
        """Does this input sit between two known patterns?"""
        sdr = self.encoder.encode(text)
        return self.memory.ambiguous(sdr)

    # ------------------------------------------------------------------ #
    #  SDR-level API (for power users)                                     #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> SDR:
        """Get the raw SDR for any text."""
        return self.encoder.encode(text)

    def similarity(self, a: str, b: str) -> float:
        """How similar are two texts? 0.0 - 1.0."""
        return self.encoder.encode(a).overlap_score(self.encoder.encode(b))

    # ------------------------------------------------------------------ #
    #  Inspection                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "learner": self.learner.stats(),
            "memory": self.memory.stats(),
        }

    def __repr__(self):
        return f"SymPattern(patterns={len(self.memory)}, {self.learner})"


__all__ = [
    "SymPattern",
    "SDR", "SDR_SIZE", "SDR_SPARSITY",
    "PatternMemory", "MatchResult",
    "TokenEncoder", "ScalarEncoder", "SequenceEncoder", "CompositeEncoder",
    "UnsupervisedLearner",
]
