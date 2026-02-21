"""
learner.py — Unsupervised Pattern Learner (v1.3)

v1.3 changes vs v1.0:
  - Continuous strengthening: every match strengthens the prototype,
    not just promotion events. More like Hebbian learning.
  - Candidate decay: candidates that aren't seen again lose strength
    over time and eventually get pruned. Prevents stale memory buildup.
  - Confidence tracking: each pattern has a running confidence score
    based on how consistently it's been matched.
  - on_the_go mode: lowers thresholds so the system learns faster from live streams.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .sdr import SDR
from .memory import PatternMemory, MatchResult


@dataclass
class CandidatePattern:
    sdr: SDR
    count: int = 1
    strength: float = 1.0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"pattern_{id(self)}"

    def decay(self, rate: float = 0.85):
        self.strength *= rate

    @property
    def alive(self) -> bool:
        return self.strength > 0.1


@dataclass
class PatternConfidence:
    label: str
    match_count: int = 0
    total_score: float = 0.0

    @property
    def avg_score(self) -> float:
        if self.match_count == 0:
            return 0.0
        return self.total_score / self.match_count

    def update(self, score: float):
        self.match_count += 1
        self.total_score += score


class UnsupervisedLearner:
    """
    Watches input, discovers patterns on its own.
    v1.3: continuous strengthening, candidate decay, confidence tracking, on_the_go mode.
    """

    def __init__(
        self,
        memory: PatternMemory = None,
        novelty_threshold: float = 0.25,
        promotion_threshold: int = 3,
        candidate_match_threshold: float = 0.35,
        max_candidates: int = 500,
        auto_label: bool = True,
        on_the_go: bool = False,
        decay_every: int = 50,
        decay_rate: float = 0.85,
    ):
        self.memory = memory if memory is not None else PatternMemory(match_threshold=novelty_threshold)
        self.auto_label = auto_label
        self.max_candidates = max_candidates
        self.decay_every = decay_every
        self.decay_rate = decay_rate

        if on_the_go:
            self.novelty_threshold = max(0.15, novelty_threshold - 0.05)
            self.promotion_threshold = max(2, promotion_threshold - 1)
            self.candidate_match_threshold = max(0.25, candidate_match_threshold - 0.05)
        else:
            self.novelty_threshold = novelty_threshold
            self.promotion_threshold = promotion_threshold
            self.candidate_match_threshold = candidate_match_threshold

        self._candidates: list[CandidatePattern] = []
        self._confidence: dict[str, PatternConfidence] = {}
        self._n_seen = 0
        self._n_promoted = 0
        self._n_novel = 0
        self._n_strengthened = 0

    def observe(self, sdr: SDR, label: Optional[str] = None) -> Optional[MatchResult]:
        self._n_seen += 1

        if self._n_seen % self.decay_every == 0:
            self._decay_candidates()

        result = self.memory.match(sdr)
        if result and result.score >= self.novelty_threshold:
            self.memory.learn(result.label, sdr)
            self._n_strengthened += 1
            if result.label not in self._confidence:
                self._confidence[result.label] = PatternConfidence(result.label)
            self._confidence[result.label].update(result.score)
            return result

        self._n_novel += 1
        matched_candidate = self._find_candidate_match(sdr)

        if matched_candidate:
            matched_candidate.sdr = matched_candidate.sdr.bundle(sdr)
            matched_candidate.count += 1
            matched_candidate.strength = min(1.0, matched_candidate.strength + 0.3)

            if matched_candidate.count >= self.promotion_threshold:
                lbl = label or matched_candidate.label or f"p{self._n_promoted}"
                self.memory.learn(lbl, matched_candidate.sdr)
                self._candidates.remove(matched_candidate)
                self._n_promoted += 1
                self._confidence[lbl] = PatternConfidence(
                    lbl, match_count=matched_candidate.count,
                    total_score=float(matched_candidate.count) * 0.5
                )
                return MatchResult(
                    label=lbl, score=1.0,
                    raw_overlap=matched_candidate.sdr.n_active(),
                    pattern=matched_candidate.sdr, input_sdr=sdr,
                )
        else:
            lbl = label or f"candidate_{self._n_seen}"
            self._candidates.append(CandidatePattern(sdr=sdr, count=1, label=lbl))
            self._prune_candidates()

        return None

    def observe_batch(self, sdrs: list[SDR], labels: list[str] = None) -> list[Optional[MatchResult]]:
        labels = labels or [None] * len(sdrs)
        return [self.observe(sdr, lbl) for sdr, lbl in zip(sdrs, labels)]

    def teach(self, label: str, sdr: SDR):
        self.memory.learn(label, sdr)
        if label not in self._confidence:
            self._confidence[label] = PatternConfidence(label)

    def recognize(self, sdr: SDR, top_k: int = 3) -> list[MatchResult]:
        return self.memory.match_all(sdr, top_k=top_k)

    def is_known(self, sdr: SDR) -> bool:
        result = self.memory.match(sdr)
        return result is not None and result.score >= self.novelty_threshold

    def _find_candidate_match(self, sdr: SDR) -> Optional[CandidatePattern]:
        best = None
        best_score = self.candidate_match_threshold
        for cand in self._candidates:
            score = sdr.overlap_score(cand.sdr)
            if score > best_score:
                best_score = score
                best = cand
        return best

    def _decay_candidates(self):
        for cand in self._candidates:
            cand.decay(self.decay_rate)
        self._candidates = [c for c in self._candidates if c.alive]

    def _prune_candidates(self):
        if len(self._candidates) > self.max_candidates:
            self._candidates.sort(key=lambda c: c.strength * c.count, reverse=True)
            self._candidates = self._candidates[:self.max_candidates]

    def confidence_report(self) -> dict:
        return {
            label: {"matches": c.match_count, "avg_score": round(c.avg_score, 3)}
            for label, c in sorted(
                self._confidence.items(), key=lambda x: x[1].avg_score, reverse=True
            )
        }

    def stats(self) -> dict:
        return {
            "seen": self._n_seen,
            "novel": self._n_novel,
            "promoted": self._n_promoted,
            "strengthened": self._n_strengthened,
            "candidates": len(self._candidates),
            "known_patterns": len(self.memory),
            "novelty_rate": round(self._n_novel / max(1, self._n_seen), 3),
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"UnsupervisedLearner("
            f"seen={s['seen']}, patterns={s['known_patterns']}, "
            f"candidates={s['candidates']}, promoted={s['promoted']}, "
            f"strengthened={s['strengthened']})"
        )