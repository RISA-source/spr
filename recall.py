"""
recall.py — Active Recall / Generation (v3.0)

Until now, sym-pattern only did recognition: input → what is this?
Active recall flips the direction: pattern → what does this produce?

Three capabilities:

1. COMPLETION
   Given partial input, find the most likely full completion.
   "The cat sat..." → "...on the mat"
   Uses stored examples as a template library. Finds the example whose
   prefix best matches the input, returns its suffix as completion.

2. EXPECTATION
   Given a recognized pattern label, generate a typical instance of it.
   "greeting" → "hello how are you"
   Returns the prototype's closest stored example — the most
   "central" instance the system has seen for this pattern.

3. ANALOGY
   A is to B as C is to ?
   Find the SDR transformation vector A→B, apply it to C, find nearest pattern.
   "cat is to animal as honda is to ?" → vehicle
   Pure symbolic vector arithmetic. Works on any learned patterns.

4. CLOZE (fill-in-the-blank)
   "The ___ sat on the mat" → "cat"
   Encodes the sentence with a masked token, finds best pattern match,
   then searches the example bank for the token that best fills the gap.

This is the flip side of recognition. Together they give the system
both perception (what is this?) and imagination (what should this be?).

Design: RecallEngine wraps PatternMemory and the example banks.
It works backward: from labels to SDRs to tokens, rather than tokens → SDRs → labels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .sdr import SDR
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, SequenceEncoder, STOP_WORDS


# ------------------------------------------------------------------ #
#  Result types                                                        #
# ------------------------------------------------------------------ #

@dataclass
class CompletionResult:
    partial_input: str
    completion: str              # suggested completion text
    full_text: str               # partial + completion
    matched_label: str           # which pattern triggered this
    confidence: float

    def __repr__(self):
        return f"Completion({self.partial_input!r} → {self.completion!r}, label={self.matched_label!r}, conf={self.confidence:.2f})"


@dataclass
class ExpectationResult:
    label: str                   # the pattern we're generating from
    typical_instance: str        # most central example text
    alternatives: list[str]      # other plausible instances
    prototype_score: float       # how close the typical instance is to the prototype

    def __repr__(self):
        return f"Expectation({self.label!r} → {self.typical_instance!r})"


@dataclass
class AnalogyResult:
    a: str
    b: str
    c: str
    answer_label: str            # the label of the answer pattern
    answer_instance: str         # a typical instance of the answer
    confidence: float
    reasoning: str               # human-readable explanation

    def __repr__(self):
        return f"Analogy({self.a!r}:{self.b!r} :: {self.c!r}:{self.answer_label!r}, conf={self.confidence:.2f})"


@dataclass
class ClozeResult:
    template: str                # original with ___ marker
    filled: str                  # completed sentence
    filler: str                  # just the word that filled the blank
    confidence: float
    alternatives: list[tuple[str, float]]  # other candidates

    def __repr__(self):
        return f"Cloze({self.template!r} → filler={self.filler!r}, conf={self.confidence:.2f})"


# ------------------------------------------------------------------ #
#  Recall Engine                                                       #
# ------------------------------------------------------------------ #

class RecallEngine:
    """
    Active recall: works backward from patterns to content.

    Requires access to:
      - PatternMemory (prototypes and their labels)
      - Example banks (the actual text examples stored per label)
      - Encoder (to re-encode text for comparison)

    The example banks are the key ingredient — they're what lets the
    system generate plausible content, not just abstract pattern labels.
    """

    def __init__(
        self,
        memory: PatternMemory,
        example_banks: dict[str, list[str]],   # label → list of raw text examples
        encoder: SequenceEncoder = None,
        token_encoder: TokenEncoder = None,
    ):
        self.memory = memory
        self.example_banks = example_banks      # the raw text, not just SDRs
        self.encoder = encoder or SequenceEncoder()
        self.token_encoder = token_encoder or TokenEncoder()
        self._sdr_cache: dict[str, SDR] = {}    # text → SDR cache

    def _encode(self, text: str) -> SDR:
        if text not in self._sdr_cache:
            self._sdr_cache[text] = self.encoder.encode(text)
        return self._sdr_cache[text]

    # ------------------------------------------------------------------ #
    #  1. COMPLETION                                                       #
    # ------------------------------------------------------------------ #

    def complete(self, partial: str, top_k: int = 3) -> list[CompletionResult]:
        """
        Given a partial input, find the most likely completion.

        Strategy:
          1. Encode the partial input
          2. Find which pattern it matches best
          3. Search that pattern's example bank for the best-matching example
          4. Extract the suffix (words beyond the partial) as the completion
        """
        partial_sdr = self._encode(partial)
        partial_tokens = partial.lower().split()

        # Find matching patterns
        matches = self.memory.match_all(partial_sdr, top_k=top_k)
        if not matches:
            return []

        results = []
        seen_completions = set()

        for match in matches:
            label = match.label
            if label not in self.example_banks:
                continue

            # Find best example in this pattern's bank
            best_example = None
            best_score = -1.0

            for example_text in self.example_banks[label]:
                ex_sdr = self._encode(example_text)
                # Score: how well does this example match the partial?
                score = partial_sdr.overlap_score(ex_sdr)
                if score > best_score:
                    best_score = score
                    best_example = example_text

            if best_example is None:
                continue

            # Extract completion: words in the example beyond the partial
            ex_tokens = best_example.lower().split()
            n_partial = len(partial_tokens)

            if len(ex_tokens) > n_partial:
                completion_tokens = ex_tokens[n_partial:]
                completion = " ".join(completion_tokens)
            else:
                # Example is shorter than partial — use full example as context
                completion = best_example

            if completion in seen_completions:
                continue
            seen_completions.add(completion)

            results.append(CompletionResult(
                partial_input=partial,
                completion=completion,
                full_text=partial + " " + completion,
                matched_label=label,
                confidence=best_score,
            ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  2. EXPECTATION                                                      #
    # ------------------------------------------------------------------ #

    def expect(self, label: str) -> Optional[ExpectationResult]:
        """
        Given a pattern label, generate a typical instance of it.

        Finds the stored example closest to the prototype — the most
        "central" or representative instance in the bank.
        """
        if label not in self.memory.prototypes:
            return None
        if label not in self.example_banks or not self.example_banks[label]:
            return None

        prototype = self.memory.prototypes[label]
        examples = self.example_banks[label]

        # Score each example against the prototype
        scored = []
        for text in examples:
            sdr = self._encode(text)
            score = sdr.overlap_score(prototype)
            scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        typical = scored[0][0]
        proto_score = scored[0][1]
        alternatives = [t for t, _ in scored[1:4]]

        return ExpectationResult(
            label=label,
            typical_instance=typical,
            alternatives=alternatives,
            prototype_score=proto_score,
        )

    def expect_all(self) -> dict[str, ExpectationResult]:
        """Get typical instance for every known pattern."""
        return {
            label: result
            for label in self.memory.prototypes
            if (result := self.expect(label)) is not None
        }

    # ------------------------------------------------------------------ #
    #  3. ANALOGY                                                          #
    # ------------------------------------------------------------------ #

    def analogy(self, a: str, b: str, c: str) -> Optional[AnalogyResult]:
        """
        A is to B as C is to ?

        SDR arithmetic:
          transform = B_proto XOR A_proto  (the "difference" between them)
          candidate = C_proto XOR transform
          find pattern whose prototype is closest to candidate

        This is the symbolic equivalent of word2vec king - man + woman = queen.
        Works on any pattern labels in memory.
        """
        if not all(x in self.memory.prototypes for x in [a, b, c]):
            missing = [x for x in [a, b, c] if x not in self.memory.prototypes]
            return None

        a_proto = self.memory.prototypes[a]
        b_proto = self.memory.prototypes[b]
        c_proto = self.memory.prototypes[c]

        # Soft delta transform — more stable than XOR on sparse binary vectors.
        # XOR flips bits discretely and creates too much noise.
        # Soft delta: treat bits as floats, compute B - A, add to C, resparsify.
        # This preserves the "direction" of change without flipping every differing bit.
        from .sdr import SDR_SIZE, SDR_SPARSITY
        n_active = int(SDR_SIZE * SDR_SPARSITY)

        a_f = a_proto.bits.astype(np.float32)
        b_f = b_proto.bits.astype(np.float32)
        c_f = c_proto.bits.astype(np.float32)

        # delta = B - A: +1 where B has bits A doesn't, -1 where A has bits B doesn't
        delta = b_f - a_f

        # Apply: push C in the same direction A→B
        candidate_f = c_f + delta

        top_indices = np.argsort(candidate_f)[-n_active:]
        candidate_sdr = SDR.from_indices(top_indices.tolist())

        # Find closest pattern (excluding a, b, c themselves)
        best_label = None
        best_score = -1.0
        for label, proto in self.memory.prototypes.items():
            if label in (a, b, c):
                continue
            score = candidate_sdr.overlap_score(proto)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        # Get a typical instance of the answer
        answer_instance = ""
        if best_label in self.example_banks and self.example_banks[best_label]:
            answer_instance = self.example_banks[best_label][0]

        reasoning = f"{a!r}→{b!r} transformation applied to {c!r} → nearest pattern is {best_label!r}"

        return AnalogyResult(
            a=a, b=b, c=c,
            answer_label=best_label,
            answer_instance=answer_instance,
            confidence=best_score,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------ #
    #  4. CLOZE (fill-in-the-blank)                                       #
    # ------------------------------------------------------------------ #

    def cloze(self, template: str, marker: str = "___", top_k: int = 5) -> Optional[ClozeResult]:
        """
        Fill in the blank: "The ___ sat on the mat" → "cat"

        Strategy:
          1. Find which pattern the template (with blank) best matches
          2. Search example bank for sentences that fit the template context
          3. Extract the word at the blank's position from the best matches
          4. Return the most frequent / highest-scoring filler
        """
        if marker not in template:
            return None

        tokens = template.lower().split()
        try:
            blank_pos = tokens.index(marker.lower())
        except ValueError:
            # Try case-insensitive
            blank_pos = next((i for i, t in enumerate(tokens) if marker.lower() in t), None)
            if blank_pos is None:
                return None

        # Encode template without the blank (as context)
        context_tokens = [t for t in tokens if t != marker.lower()]
        context_sdr = self.encoder.encode(" ".join(context_tokens))

        # Find best matching pattern
        matches = self.memory.match_all(context_sdr, top_k=3)
        if not matches:
            return None

        # Search examples from patterns whose score is within 75% of best.
        # Tighter than searching all banks — avoids cross-contamination.
        filler_scores: dict[str, float] = {}
        top_match = matches[0]
        valid_matches = [m for m in matches if m.score >= max(top_match.score * 0.75, 0.03)]

        for match in valid_matches:
            label = match.label
            if label not in self.example_banks:
                continue

            for example_text in self.example_banks[label]:
                ex_tokens = example_text.lower().split()
                if len(ex_tokens) <= blank_pos:
                    continue

                # Candidate filler: word at the blank position in this example
                filler = ex_tokens[blank_pos]
                # Note: NO stop word filter here — the answer could be "you", "not", "it", etc.

                # Score: compare context_sdr against example-without-filler
                # This asks "how well does this example fit the context around the blank?"
                ex_without = [t for i, t in enumerate(ex_tokens) if i != blank_pos]
                ex_sdr = self._encode(" ".join(ex_without))
                score = context_sdr.overlap_score(ex_sdr) * match.score
                filler_scores[filler] = max(filler_scores.get(filler, 0), score)

        if not filler_scores:
            return None

        ranked = sorted(filler_scores.items(), key=lambda x: x[1], reverse=True)
        best_filler = ranked[0][0]
        best_score = ranked[0][1]

        # Normalize score
        max_score = max(s for _, s in ranked)
        normalized = best_score / max_score if max_score > 0 else 0.0

        filled = template.replace(marker, best_filler)

        return ClozeResult(
            template=template,
            filled=filled,
            filler=best_filler,
            confidence=normalized,
            alternatives=[(f, s / max_score) for f, s in ranked[1:top_k]],
        )

    # ------------------------------------------------------------------ #
    #  Inspection                                                          #
    # ------------------------------------------------------------------ #

    def coverage(self) -> dict:
        """How many patterns have example banks? How rich are they?"""
        return {
            label: len(self.example_banks.get(label, []))
            for label in self.memory.prototypes
        }

    def __repr__(self):
        n_patterns = len(self.memory)
        n_examples = sum(len(v) for v in self.example_banks.values())
        return f"RecallEngine(patterns={n_patterns}, examples={n_examples})"