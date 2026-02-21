"""
learner.py — Unsupervised Pattern Learner

This is where "learning without being told" happens.

The system watches an input stream, finds what recurs, and
promotes recurring structures into named patterns automatically.

Algorithm:
  1. Encode each input as an SDR
  2. Check if it matches anything in memory well enough
  3. If yes: strengthen that pattern (it's recurring)
  4. If no (novel): tentatively store it
  5. Tentative patterns that recur enough → promoted to real patterns
  6. Patterns that never recur → forgotten

This is analogous to:
  - Hebbian learning ("fire together wire together")
  - Competitive learning / self-organizing maps
  - Online k-means clustering
  
But all symbolic, all interpretable.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .sdr import SDR
from .memory import PatternMemory, MatchResult


@dataclass
class CandidatePattern:
    """A pattern that's been seen but not promoted yet."""
    sdr: SDR
    count: int = 1
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"pattern_{id(self)}"


class UnsupervisedLearner:
    """
    Watches input, discovers patterns on its own.
    
    Two-tier memory:
      - candidates: SDRs seen but not yet frequent enough to trust
      - memory: promoted patterns (the PatternMemory)
    
    When a candidate matches a candidate already in the pool,
    we bundle them and increment the count.
    When count hits promotion_threshold, it becomes a real pattern.
    """

    def __init__(
        self,
        memory: PatternMemory = None,
        novelty_threshold: float = 0.25,   # below this → it's novel
        promotion_threshold: int = 3,       # seen this many times → promote
        candidate_match_threshold: float = 0.35,  # for matching within candidates
        max_candidates: int = 500,
        auto_label: bool = True,
    ):
        self.memory = memory if memory is not None else PatternMemory(match_threshold=novelty_threshold)
        self.novelty_threshold = novelty_threshold
        self.promotion_threshold = promotion_threshold
        self.candidate_match_threshold = candidate_match_threshold
        self.max_candidates = max_candidates
        self.auto_label = auto_label

        self._candidates: list[CandidatePattern] = []
        self._n_seen = 0
        self._n_promoted = 0
        self._n_novel = 0

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def observe(self, sdr: SDR, label: Optional[str] = None) -> Optional[MatchResult]:
        """
        Show the learner one input SDR.
        
        If it matches a known pattern → returns MatchResult, strengthens pattern.
        If it's novel → adds to candidates, may promote if seen enough.
        Returns None if truly novel and not yet promoted.
        """
        self._n_seen += 1

        # 1. Check against known patterns
        result = self.memory.match(sdr)
        if result and result.score >= self.novelty_threshold:
            # Strengthen the prototype by including this example
            self.memory.learn(result.label, sdr)
            return result

        # 2. Novel — check candidates
        self._n_novel += 1
        matched_candidate = self._find_candidate_match(sdr)

        if matched_candidate:
            # Merge with existing candidate
            matched_candidate.sdr = matched_candidate.sdr.bundle(sdr)
            matched_candidate.count += 1

            if matched_candidate.count >= self.promotion_threshold:
                # Promote to real pattern!
                lbl = label or matched_candidate.label or f"p{self._n_promoted}"
                if self.auto_label and not label:
                    lbl = self._auto_label(matched_candidate.sdr, lbl)
                self.memory.learn(lbl, matched_candidate.sdr)
                self._candidates.remove(matched_candidate)
                self._n_promoted += 1
                # Return a synthetic match result
                return MatchResult(
                    label=lbl,
                    score=1.0,
                    raw_overlap=matched_candidate.sdr.n_active(),
                    pattern=matched_candidate.sdr,
                    input_sdr=sdr
                )
        else:
            # Brand new — add as candidate
            lbl = label or f"candidate_{self._n_seen}"
            self._candidates.append(CandidatePattern(sdr=sdr, count=1, label=lbl))
            self._prune_candidates()

        return None

    def observe_batch(self, sdrs: list[SDR], labels: list[str] = None) -> list[Optional[MatchResult]]:
        labels = labels or [None] * len(sdrs)
        return [self.observe(sdr, lbl) for sdr, lbl in zip(sdrs, labels)]

    # ------------------------------------------------------------------ #
    #  Supervised shortcut                                                 #
    # ------------------------------------------------------------------ #

    def teach(self, label: str, sdr: SDR):
        """
        Supervised: directly teach a labeled pattern.
        Bypasses the candidate stage — goes straight to memory.
        Can mix supervised and unsupervised freely.
        """
        self.memory.learn(label, sdr)

    # ------------------------------------------------------------------ #
    #  Recognition (after learning)                                        #
    # ------------------------------------------------------------------ #

    def recognize(self, sdr: SDR, top_k: int = 3) -> list[MatchResult]:
        """Recognize an input against all learned patterns."""
        return self.memory.match_all(sdr, top_k=top_k)

    def is_known(self, sdr: SDR) -> bool:
        result = self.memory.match(sdr)
        return result is not None and result.score >= self.novelty_threshold

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _find_candidate_match(self, sdr: SDR) -> Optional[CandidatePattern]:
        best = None
        best_score = self.candidate_match_threshold
        for cand in self._candidates:
            score = sdr.overlap_score(cand.sdr)
            if score > best_score:
                best_score = score
                best = cand
        return best

    def _prune_candidates(self):
        """Keep only the most recent/frequent candidates if we hit the limit."""
        if len(self._candidates) > self.max_candidates:
            # Sort by count desc, trim tail
            self._candidates.sort(key=lambda c: c.count, reverse=True)
            self._candidates = self._candidates[:self.max_candidates]

    def _auto_label(self, sdr: SDR, fallback: str) -> str:
        """
        Try to give a meaningful label based on the SDR's structure.
        In v1 this is just the fallback — future versions could use
        the active bit positions to infer something meaningful.
        """
        return fallback

    # ------------------------------------------------------------------ #
    #  Stats / inspection                                                  #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "seen": self._n_seen,
            "novel": self._n_novel,
            "promoted": self._n_promoted,
            "candidates": len(self._candidates),
            "known_patterns": len(self.memory),
            "novelty_rate": self._n_novel / max(1, self._n_seen),
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"UnsupervisedLearner("
            f"seen={s['seen']}, "
            f"patterns={s['known_patterns']}, "
            f"candidates={s['candidates']}, "
            f"promoted={s['promoted']})"
        )
