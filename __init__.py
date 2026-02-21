"""
sym_pattern — Symbolic Pattern Recognition System v1.3

Core components:
  - SDR:                Sparse Distributed Representation (the atom)
  - TokenEncoder:       text → SDR (with subword overlap)
  - ScalarEncoder:      numbers → SDR (nearby values share bits)
  - SequenceEncoder:    ordered sequences → SDR (position-aware)
  - CompositeEncoder:   combine multiple encoders
  - PatternMemory:      stores and matches patterns with confidence scores
  - UnsupervisedLearner: learns patterns from raw input automatically (v1.3: +decay, +confidence)
  - persistence:        save/load learned patterns to disk (v1.1)
  - fileops:            learn from files, compare files for similarity (v1.2)

Quickstart:

    from sym_pattern import SymPattern
    from sym_pattern.persistence import save, load
    from sym_pattern.fileops import learn_from_file, compare_files

    sp = SymPattern()
    sp.teach("greeting", ["hello world", "hi there"])
    sp.teach("farewell", ["goodbye", "see you later"])

    result = sp.best_match("hey what's up")
    print(result)

    save(sp.memory, "my_patterns.npz")          # v1.1: persist
    sp2 = SymPattern()
    sp2.memory = load("my_patterns.npz")        # v1.1: restore

    learn_from_file(sp, "corpus.txt")           # v1.2: learn from file
    report = compare_files(sp, "a.txt", "b.txt") # v1.2: compare files
    print(report.summary())
"""

from .sdr import SDR, SDR_SIZE, SDR_SPARSITY
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, ScalarEncoder, SequenceEncoder, CompositeEncoder
from .learner import UnsupervisedLearner
from . import persistence
from . import fileops


class SymPattern:
    """
    High-level API. One object to rule them all.
    Works supervised, unsupervised, or mixed.
    v1.1: save/load via sp.save() / SymPattern.load()
    v1.2: file learning via sp.learn_file() / sp.compare_files()
    v1.3: on_the_go mode, confidence_report()
    """

    def __init__(
        self,
        unsupervised: bool = False,
        subword_overlap: bool = True,
        novelty_threshold: float = 0.25,
        promotion_threshold: int = 3,
        on_the_go: bool = False,
    ):
        self.encoder = SequenceEncoder(
            token_encoder=TokenEncoder(subword_overlap=subword_overlap)
        )
        self.memory = PatternMemory(match_threshold=novelty_threshold)
        self.learner = UnsupervisedLearner(
            memory=self.memory,
            novelty_threshold=novelty_threshold,
            promotion_threshold=promotion_threshold,
            on_the_go=on_the_go,
        )
        self.unsupervised = unsupervised

    # ------------------------------------------------------------------ #
    #  Supervised                                                          #
    # ------------------------------------------------------------------ #

    def teach(self, label: str, examples: list[str]):
        for example in examples:
            sdr = self.encoder.encode(example)
            self.learner.teach(label, sdr)
        return self

    # ------------------------------------------------------------------ #
    #  Unsupervised                                                        #
    # ------------------------------------------------------------------ #

    def observe(self, text: str, label: str = None) -> MatchResult | None:
        sdr = self.encoder.encode(text)
        return self.learner.observe(sdr, label=label)

    def observe_corpus(self, texts: list[str]) -> list[MatchResult | None]:
        return [self.observe(t) for t in texts]

    # ------------------------------------------------------------------ #
    #  Recognition                                                         #
    # ------------------------------------------------------------------ #

    def recognize(self, text: str, top_k: int = 3) -> list[MatchResult]:
        sdr = self.encoder.encode(text)
        return self.learner.recognize(sdr, top_k=top_k)

    def best_match(self, text: str) -> MatchResult | None:
        results = self.recognize(text, top_k=1)
        return results[0] if results and results[0].score >= self.memory.match_threshold else None

    def is_novel(self, text: str) -> bool:
        sdr = self.encoder.encode(text)
        return self.memory.is_novel(sdr)

    def is_ambiguous(self, text: str) -> bool:
        sdr = self.encoder.encode(text)
        return self.memory.ambiguous(sdr)

    # ------------------------------------------------------------------ #
    #  SDR primitives                                                      #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> SDR:
        return self.encoder.encode(text)

    def similarity(self, a: str, b: str) -> float:
        return self.encoder.encode(a).overlap_score(self.encoder.encode(b))

    # ------------------------------------------------------------------ #
    #  v1.1 — Persistence                                                  #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save learned patterns to disk (.npz or .json)."""
        persistence.save(self.memory, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "SymPattern":
        """Load a SymPattern from a saved file. kwargs passed to __init__."""
        sp = cls(**kwargs)
        sp.memory = persistence.load(path)
        sp.learner.memory = sp.memory   # keep in sync
        return sp

    # ------------------------------------------------------------------ #
    #  v1.2 — File operations                                              #
    # ------------------------------------------------------------------ #

    def learn_file(self, path: str, label: str = None, chunk_size: int = 50) -> dict:
        """Learn patterns from a text file."""
        return fileops.learn_from_file(self, path, label=label, chunk_size=chunk_size)

    def compare_files(self, path_a: str, path_b: str, chunk_size: int = 50) -> fileops.FileSimilarityReport:
        """Compare two files for similarity. Returns a FileSimilarityReport."""
        return fileops.compare_files(self, path_a, path_b, chunk_size=chunk_size)

    # ------------------------------------------------------------------ #
    #  v1.3 — Confidence                                                   #
    # ------------------------------------------------------------------ #

    def confidence_report(self) -> dict:
        """How confident is the system in each learned pattern?"""
        return self.learner.confidence_report()

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
    "persistence", "fileops",
]