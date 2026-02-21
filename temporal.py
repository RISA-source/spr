"""
temporal.py — Temporal Sequence Memory (v2.1)

Tracks what patterns tend to follow other patterns over time.

Why this matters:
  A flat pattern matcher sees each input in isolation.
  "machine learning" and "deep learning" might both match "ML_topic".
  But it has no idea that "ML_topic → result → conclusion" is a typical
  sequence structure in papers, while "ML_topic → ML_topic → ML_topic"
  might mean someone is going in circles.

  Temporal memory adds the TIME dimension to recognition.

What it does:
  - Maintains a transition graph: pattern_A → pattern_B with a count/weight
  - After seeing pattern_A, predicts what pattern is likely to come next
  - Flags SURPRISES when what comes next has low transition probability
  - Detects recurring SEQUENCE PATTERNS (not just single patterns)
  - Can recognize narrative/discourse structure: intro → body → conclusion

This is the symbolic analog of:
  - RNN hidden state (we use explicit graph instead of dense vector)
  - n-gram language model (we use pattern labels instead of raw words)
  - HMM (but we learn the transitions from data, not specify them)

Implementation:
  Weighted directed graph where nodes are pattern labels and edges
  are transition counts. Normalized → transition probabilities.
  Sequences of labels → SDR-encoded for sequence pattern matching.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from .sdr import SDR
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, SequenceEncoder
from .learner import UnsupervisedLearner


# ------------------------------------------------------------------ #
#  Transition graph                                                    #
# ------------------------------------------------------------------ #

class TransitionGraph:
    """
    Directed weighted graph of pattern-to-pattern transitions.
    
    After seeing label A then label B, edge A→B gets stronger.
    After enough data, the graph tells us:
      - What usually follows A (predictions)
      - What rarely follows A (surprises)
      - What chains of patterns tend to occur together (sequence patterns)
    """

    def __init__(self, smoothing: float = 0.1):
        # counts[from_label][to_label] = raw transition count
        self.counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.smoothing = smoothing   # Laplace smoothing — prevents zero probabilities
        self._total_transitions = 0

    def observe_transition(self, from_label: str, to_label: str, weight: float = 1.0):
        """Record that from_label was followed by to_label."""
        self.counts[from_label][to_label] += weight
        self._total_transitions += 1

    def transition_probability(self, from_label: str, to_label: str) -> float:
        """P(to_label | from_label)"""
        if from_label not in self.counts:
            return self.smoothing  # unseen state → small prob
        row = self.counts[from_label]
        total = sum(row.values()) + self.smoothing * len(row)
        return (row.get(to_label, 0) + self.smoothing) / (total + self.smoothing)

    def top_successors(self, from_label: str, k: int = 3) -> list[tuple[str, float]]:
        """What labels most commonly follow from_label?"""
        if from_label not in self.counts:
            return []
        row = self.counts[from_label]
        total = sum(row.values())
        if total == 0:
            return []
        ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)
        return [(lbl, cnt / total) for lbl, cnt in ranked[:k]]

    def surprise(self, from_label: str, to_label: str) -> float:
        """
        How surprising is this transition? 0.0 = expected, 1.0 = very surprising.
        Based on negative log probability, normalized.
        """
        prob = self.transition_probability(from_label, to_label)
        # -log2(prob) ranges from 0 (certain) to inf (impossible)
        # Normalize with a soft cap at 10 bits of surprise
        raw_surprise = -np.log2(max(prob, 1e-10))
        return min(1.0, raw_surprise / 10.0)

    def known_patterns(self) -> list[str]:
        return list(self.counts.keys())

    def stats(self) -> dict:
        return {
            "nodes": len(self.counts),
            "total_transitions": self._total_transitions,
            "edges": sum(len(v) for v in self.counts.values()),
        }


# ------------------------------------------------------------------ #
#  Sequence pattern memory                                             #
# ------------------------------------------------------------------ #

class SequencePatternMemory:
    """
    Recognizes recurring SEQUENCES of patterns, not just single patterns.
    
    Uses a sliding window over recent pattern labels, encodes the window
    as an SDR, and matches it against known sequence prototypes.
    
    Example:
      Labels seen: ["greeting", "question", "answer", "farewell"]
      Window of 3: ["greeting", "question", "answer"] → SDR → matches "conversation"
      
    This is what lets the system recognize discourse structure, narrative shape,
    argument structure, etc — things that only emerge over multiple steps.
    """

    def __init__(self, window_size: int = 4, min_sequence_count: int = 3):
        self.window_size = window_size
        self.min_sequence_count = min_sequence_count
        self._seq_encoder = SequenceEncoder(
            token_encoder=TokenEncoder(subword_overlap=False)  # exact label matching
        )
        self._memory = PatternMemory(match_threshold=0.35)
        self._learner = UnsupervisedLearner(
            memory=self._memory,
            novelty_threshold=0.35,
            promotion_threshold=min_sequence_count,
        )

    def observe_window(self, label_window: list[str]) -> Optional[MatchResult]:
        """
        Learn from a window of consecutive pattern labels.
        Returns a match if this sequence is a known sequence pattern.
        """
        if len(label_window) < 2:
            return None
        window_sdr = self._seq_encoder.encode(" ".join(label_window))
        return self._learner.observe(window_sdr)

    def recognize_sequence(self, label_window: list[str]) -> list[MatchResult]:
        """Recognize a label sequence against known sequence patterns."""
        window_sdr = self._seq_encoder.encode(" ".join(label_window))
        return self._memory.match_all(window_sdr, top_k=3)

    def teach_sequence(self, seq_label: str, label_sequences: list[list[str]]):
        """Teach named sequence patterns."""
        for seq in label_sequences:
            sdr = self._seq_encoder.encode(" ".join(seq))
            self._learner.teach(seq_label, sdr)

    @property
    def n_patterns(self) -> int:
        return len(self._memory)


# ------------------------------------------------------------------ #
#  Temporal memory — ties it all together                             #
# ------------------------------------------------------------------ #

@dataclass
class TemporalResult:
    """What temporal memory produces for one step."""
    current_label: Optional[str]         # what was just recognized
    current_score: float
    predictions: list[tuple[str, float]] # (label, probability) for next step
    surprise: float                       # how surprising was this transition
    sequence_match: Optional[MatchResult] # did recent sequence match anything?
    from_label: Optional[str]            # what came before

    def __repr__(self):
        preds = ", ".join(f"{l}({p:.2f})" for l, p in self.predictions[:2])
        return (
            f"Temporal(current={self.current_label!r}, "
            f"surprise={self.surprise:.2f}, "
            f"predicts=[{preds}])"
        )


class TemporalMemory:
    """
    Adds time awareness to the pattern recognition system.
    
    Wraps a PatternMemory and tracks the sequence of what gets recognized.
    Builds a transition graph from observations.
    Recognizes recurring multi-step sequences.
    Predicts what pattern comes next.
    Flags transitions that are unexpectedly surprising.
    """

    def __init__(
        self,
        pattern_memory: PatternMemory,
        window_size: int = 4,
        surprise_threshold: float = 0.7,
    ):
        self.pattern_memory = pattern_memory
        self.transition_graph = TransitionGraph()
        self.seq_memory = SequencePatternMemory(window_size=window_size)
        self.surprise_threshold = surprise_threshold
        self.window_size = window_size

        self._history: list[str] = []          # recent recognized labels
        self._prev_label: Optional[str] = None
        self._n_steps = 0
        self._surprises: list[tuple[str, str, float]] = []  # (from, to, score)

    def step(self, input_sdr: SDR) -> TemporalResult:
        """
        Process one input through pattern memory + temporal tracking.
        Updates transition graph and sequence memory.
        """
        self._n_steps += 1

        # Recognize current input
        matches = self.pattern_memory.match_all(input_sdr, top_k=1)
        current_label = matches[0].label if matches and matches[0].score >= self.pattern_memory.match_threshold else None
        current_score = matches[0].score if matches else 0.0

        # Compute surprise and predictions
        surprise = 0.0
        predictions = []

        if self._prev_label and current_label:
            surprise = self.transition_graph.surprise(self._prev_label, current_label)
            self.transition_graph.observe_transition(self._prev_label, current_label)
            if surprise >= self.surprise_threshold:
                self._surprises.append((self._prev_label, current_label, surprise))

        if current_label:
            predictions = self.transition_graph.top_successors(current_label, k=3)

        # Update history window
        if current_label:
            self._history.append(current_label)
            if len(self._history) > self.window_size * 2:
                self._history = self._history[-self.window_size * 2:]

        # Check for sequence patterns
        seq_match = None
        if len(self._history) >= 2:
            window = self._history[-self.window_size:]
            seq_match = self.seq_memory.observe_window(window)

        result = TemporalResult(
            current_label=current_label,
            current_score=current_score,
            predictions=predictions,
            surprise=surprise,
            sequence_match=seq_match,
            from_label=self._prev_label,
        )

        self._prev_label = current_label
        return result

    def teach_sequence(self, seq_label: str, label_sequences: list[list[str]]):
        """Teach named multi-step sequence patterns."""
        self.seq_memory.teach_sequence(seq_label, label_sequences)

    def predict_next(self, top_k: int = 3) -> list[tuple[str, float]]:
        """What does the system expect to see next?"""
        if not self._prev_label:
            return []
        return self.transition_graph.top_successors(self._prev_label, k=top_k)

    def recent_surprises(self) -> list[tuple[str, str, float]]:
        """Get transitions that were surprising."""
        return sorted(self._surprises, key=lambda x: x[2], reverse=True)

    def history(self) -> list[str]:
        return list(self._history)

    def stats(self) -> dict:
        return {
            "steps": self._n_steps,
            "surprises": len(self._surprises),
            "transition_graph": self.transition_graph.stats(),
            "sequence_patterns": self.seq_memory.n_patterns,
            "history_len": len(self._history),
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"TemporalMemory(steps={s['steps']}, "
            f"transitions={s['transition_graph']['edges']}, "
            f"seq_patterns={s['sequence_patterns']})"
        )