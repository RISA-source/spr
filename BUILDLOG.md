# sym-pattern — BUILDLOG

> Every decision, every mistake, every fix. Written as we go.
> This file grows with the project. Never cleaned up, only added to.

---

## What this project is

A symbolic AI pattern recognition system built to match neural-network-grade
pattern recognition capability — without using neural networks.

The core bet: if you represent everything as Sparse Distributed Representations
(SDRs) and layer enough structure on top, you can get NN-grade fuzzy matching,
noise robustness, and generalization using only explicit symbolic operations.

Language: Python first. Port to Go or Rust later once the architecture is stable.

---

## BUILDLOG: v1

### Session 1 — Architecture decision

**Date:** Session start  
**Question:** What approach gives NN-grade pattern matching symbolically?

**Options considered:**

1. Pure rule-based matching — brittle, won't generalize, ruled out immediately
2. Flat vector/cosine similarity — gets ~60-70% of the way, breaks on:
   - compositionality ("dog bites man" vs "man bites dog" same vector)
   - hierarchy (can't represent "poodle" under "dog" under "animal")
   - context sensitivity (same word, different meaning)
3. Sparse Distributed Representations + structural matching — chosen

**Decision: SDRs as the foundation.**

Reasons:
- Overlap naturally encodes similarity — no distance metric needed
- Robust to noise by design (losing a few bits doesn't kill the match)
- Composable via set operations (union = generalize, intersection = constrain)
- Fully inspectable — you can look at the active bits and reason about them
- This is how Numenta's HTM works, and how biological neurons actually code

**Key insight from discussion:** vectors alone aren't enough. You need:
- Vectors (fuzzy similarity, fast lookup)
- Structure (compositionality, hierarchy, context)
- Graded scoring (confidence at every level, not just true/false)

---

### Module 1: `sdr.py` — Sparse Distributed Representation

**What it does:**  
The atom of the whole system. Every symbol, token, pattern, concept is
an SDR — a 1024-dimensional binary vector with ~5% of bits active (~51 bits).

**Parameters chosen:**
- `SDR_SIZE = 1024` — started at 2048, tuned down (see bug below)
- `SDR_SPARSITY = 0.05` — 5% active bits. Started at 2%, tuned up (see bug below)

**Key operations implemented:**
- `from_hash(string)` — deterministic SDR from any string, no training needed
- `from_indices(list)` — build from explicit active bit positions
- `random()` — random SDR at correct sparsity (for testing)
- `overlap(other)` — raw bit overlap count (the core similarity primitive)
- `overlap_score(other)` — normalized similarity 0.0–1.0
- `subsumes(other, threshold)` — does one SDR contain another? (for hierarchy)
- `union(other)` — OR of two SDRs (more general)
- `intersection(other)` — AND of two SDRs (more specific / shared structure)
- `add_noise(flip_rate)` — randomly move active bits (robustness testing)
- `bundle(*others)` — majority-vote aggregation of multiple SDRs (prototype building)

**Design decision — `bundle()`:**  
Instead of averaging (which loses sparsity), we do majority vote per bit and
then take the top-N most-voted bits. This keeps the result a proper SDR.
Equivalent to averaging embeddings but stays in symbolic/binary space.

---

### Bug 1: `overlap_score` using Jaccard — scores too low

**Problem:**  
Initial `overlap_score` used Jaccard similarity (intersection / union).  
With 5% sparsity, two related sentences might share 10 bits out of a union
of ~90 bits = Jaccard ~0.11. Too low to distinguish signal from noise.

**Symptom:**  
All recognition scores were in range 0.01–0.05. Thresholds couldn't be
set to anything sensible — too loose and everything matched, too tight
and nothing did.

**Root cause:**  
Jaccard penalizes wide prototypes. When you bundle 5 examples, the prototype
has ~51 active bits (kept sparse by the bundling algorithm). But a query SDR
also has 51 bits. The union is ~80-90 bits. Even 15 overlapping bits = ~0.18
Jaccard. Fine for comparing two single SDRs, wrong for query-vs-prototype.

**Fix:**  
Changed denominator from `union` to `min(self.n_active(), other.n_active())`.
This asks "what fraction of the smaller SDR's bits does the match cover?"
which is the right question for recognition: does the input fit the prototype?

```python
# Before (Jaccard):
intersection / union

# After (min-normalized):
intersection / min(self.n_active(), other.n_active())
```

**Result:**  
Scores jumped from ~0.05 to ~0.20–0.35 for genuine matches.
`"bye see you tomorrow"` vs `farewell` prototype: 0.353. Clear signal.

---

### Bug 2: SDR parameters wrong — scores still weak

**Problem:**  
First run used `SDR_SIZE=2048, SDR_SPARSITY=0.02` (2% = ~41 bits active).
With a large space and few bits, even related things rarely share bits.

**Fix:**  
Tuned to `SDR_SIZE=1024, SDR_SPARSITY=0.05` (5% = ~51 bits active).
Smaller space + more active bits = more natural overlap between related SDRs.

---

### Module 2: `encoder.py` — Input Encoders

**What it does:**  
Bridges raw input (text, numbers, sequences) to SDR space.

**Three encoders:**

**`TokenEncoder`**  
Encodes individual words/tokens. Key feature: subword overlap.  
"running" and "runner" share the n-gram "runn" → their SDRs share bits naturally.  
Implementation: build token SDR as union of character 3-gram SDRs, weighted by vote count.  
This gives free morphological similarity without any training.

**`ScalarEncoder`**  
Encodes numbers. Nearby numbers share bits.  
Implementation: map value to a position in the SDR dimension space,
activate a contiguous window of bits around that position.  
`55` and `56` share most bits. `55` and `900` share almost none.  
This is directly from Numenta's HTM scalar encoding design.

**`SequenceEncoder`**  
Encodes ordered sequences of tokens as a single SDR.  
Key challenge: "dog bites man" ≠ "man bites dog" even though same words.  
Implementation: rotate each token's SDR by `position * shift` before combining.  
Position is encoded structurally in which bits are active, not separately.

**Design decision — position shift value:**  
Used shift of 3 (small) rather than 17 (initial value).  
Large shifts meant adjacent positions rarely overlapped — encoding was too strict.  
Small shift means nearby positions share structure, which is semantically right
(word at position 2 relates to word at position 3 more than position 20).

---

### Bug 3: Sequence encoder had duplicate `encode()` method

**Problem:**  
`SequenceEncoder` had two `encode()` methods defined — one that rotated bits with
shift=17, and a second that delegated to `_encode_tokens()` with shift=3.  
Python silently took the second (later defined) one. The first was dead code.

**Fix:**  
Removed the dead first `encode()`, kept the second. Renamed internal logic
to `_encode_tokens()` which handles both string and list input.

---

### Module 3: `memory.py` — Pattern Memory

**What it does:**  
The learned knowledge store. Stores one prototype SDR per named pattern,
matches input against all prototypes, returns ranked confidence scores.

**Key design:**  
Each pattern is represented by ONE prototype SDR, built by bundling all examples.
This is the "learned weight" analog — except it's a single interpretable vector
you can inspect, not a matrix of floats.

**`MatchResult` dataclass:**  
Returns more than just a label — includes: label, score (0–1), raw overlap count,
the matched prototype, the input SDR. Caller gets full picture.

**`match_all()` → top-k ranked results:**  
Returns multiple candidates, not just the best. This is important for:
- Seeing runner-up candidates (is it clearly one thing or ambiguous?)
- Ambiguity detection (two high scores = uncertain, worth flagging)
- Soft classification (weighted vote over top matches)

**`is_novel()`:**  
Checks if input matches anything well enough. This is the gateway to
unsupervised learning — if something is novel, the learner should try to
learn it rather than force a bad match.

**`ambiguous()`:**  
Checks if top two matches are too close to call. Returns true if
gap between first and second score is below threshold. Useful for
detecting inputs that genuinely sit between categories.

---

### Module 4: `learner.py` — Unsupervised Pattern Learner

**What it does:**  
Watches raw input stream, discovers recurring patterns, promotes them to
named patterns in memory automatically. No labels required.

**Two-tier architecture:**
- **Candidates:** SDRs seen but not yet trusted. Held in a list.
- **Memory (PatternMemory):** Promoted patterns. The real learned knowledge.

**The learning loop:**
1. New input arrives
2. Check against known patterns — if matches, strengthen prototype and return
3. If novel — check against candidates
4. If matches a candidate — bundle them (merge), increment count
5. If count hits `promotion_threshold` (default: 3) — promote to real pattern
6. If brand new — add as new candidate
7. Prune candidates if too many (keep most frequent)

**Design decision — promotion threshold:**  
Set to 3 by default. Means a pattern must recur at least 3 times before being
trusted. Too low (1) = everything gets promoted = noisy. Too high (10) = nothing
gets promoted = useless. 3 is conservative but works for small corpora.

**`teach()` method:**  
Supervised shortcut — bypass the candidate stage entirely and go straight to
memory. Allows mixing supervised and unsupervised freely in the same system.

---

### Bug 4: `memory or PatternMemory(...)` — classic Python truthiness trap

**Problem:**  
`UnsupervisedLearner.__init__` had:
```python
self.memory = memory or PatternMemory(match_threshold=novelty_threshold)
```
When `SymPattern` passed its `PatternMemory` instance to `UnsupervisedLearner`,
Python evaluated `memory or ...` — since `PatternMemory.__len__` returns 0
(empty store), `bool(memory)` was `False`. So Python created a NEW
`PatternMemory` and assigned that instead. The passed-in object was ignored.

**Symptom:**  
`SymPattern.memory` and `SymPattern.learner.memory` were different objects.
Calling `sp.teach()` wrote to `learner.memory`. Reading `sp.memory` returned
an empty store. Recognition always failed.

Confirmed with:
```python
sp.memory is sp.learner.memory  # False — should be True
```

**Fix:**  
```python
# Before:
self.memory = memory or PatternMemory(...)

# After:
self.memory = memory if memory is not None else PatternMemory(...)
```

**Lesson:**  
Never use `x or default` when `x` might be a valid but falsy object.
Always use `x if x is not None else default`.

---

### Module 5: `__init__.py` — SymPattern high-level API

**What it does:**  
One object, clean interface. Wraps encoder + memory + learner.

**Public API:**
- `teach(label, examples)` — supervised, give labeled examples
- `observe(text, label=None)` — show input, system decides if familiar or novel
- `observe_corpus(texts)` — bulk unsupervised learning
- `recognize(text, top_k=3)` — get ranked matches with scores
- `best_match(text)` — single best match or None
- `is_novel(text)` — has system seen anything like this?
- `is_ambiguous(text)` — does this sit between two known patterns?
- `encode(text)` — get raw SDR (for power users)
- `similarity(a, b)` — direct text similarity 0–1
- `stats()` — current state of learner and memory

---

### v1 Demo Results

**Demo 1 — Supervised recognition:**
```
"bye see you tomorrow"  → farewell  0.353  ✓
"what do you think"     → question  0.373  ✓
"hey good morning mate" → greeting  0.235  ✓
"hello friend"          → greeting  0.216  ✓
"godbye see u" (noisy)  → farewell  0.314  ✓  (subword match working)
```

**Demo 3 — Subword similarity:**
```
"playing" ↔ "player"   0.412  (share "-lay", "-layer", "-aying" n-grams)
"cat" ↔ "cats"          0.333
"run" ↔ "runs"          0.392
```

**Demo 3 — Noise robustness:**
```
noise=0.0  score=0.235  match=animal  ✓
noise=0.2  score=0.196  match=animal  ✓
noise=0.4  score=0.118  match=animal  ✓  (still recognizing at 40% bit corruption)
noise=0.5  score=0.137  match=animal  ✓
```

---

### Honest v1 assessment

**What v1 actually is:**  
A smart fuzzy matcher. Comparable in practical capability to TF-IDF or
bag-of-words with cosine similarity. Better than naive symbolic matching,
weaker than any trained neural model.

**What v1 does well:**
- Fuzzy/partial text matching with graded confidence
- Noise robustness (corrupted input still recognized)
- Subword morphological similarity for free (no training)
- Order-sensitive sequence encoding
- Novelty detection
- Ambiguity detection
- Composable representations (union/intersection)
- Fully inspectable — no black box

**What v1 cannot do:**
- Context-sensitive meaning ("bank" river vs bank account)
- Hierarchical abstraction (poodle → dog → animal)
- Compositional understanding ("dog bites man" vs "man bites dog" treated similarly)
- True continuous learning (learning is chunky, discrete, not gradient-smooth)
- Long-range dependencies in sequences
- Persistent storage (no save/load yet)
- File-to-file similarity comparison

**Gap to NN-grade, honestly:**  
Significant. A transformer is roughly: flat SDR matching (us) × hierarchical
composition × context flow × gradient-quality generalization. We have layer 1.

---

---

## BUILDLOG: v1.1 — Save / Load (Persistence)

**Problem being solved:**  
After every session, all learned patterns were gone. The system had no memory
between runs. Completely unusable in any real workflow.

**New module:** `persistence.py`

Two formats supported by file extension:

`.npz` (recommended) — numpy binary archive, compressed, fast.  
Packs all prototype bit arrays into one file alongside labels, example counts,
SDR size and match threshold. Can detect config mismatches on load.

`.json` — human readable. Active indices stored as integer lists.  
Useful for debugging, inspection, manual editing, or sending patterns to someone.
Slower on large stores.

**API added to `SymPattern`:**
```python
sp.save("patterns.npz")           # save
sp2 = SymPattern.load("patterns.npz")   # restore
```

**Design decision — store active indices not full bit arrays in JSON:**  
A full 1024-bit array as JSON would be a list of 1024 zeros and ones — unreadable
and large. Storing only the ~51 active indices is compact and human-inspectable.
NPZ stores the full bool array since numpy handles compression efficiently.

**Design decision — validate SDR_SIZE on load:**  
If someone saves with `SDR_SIZE=1024` and tries to load with `SDR_SIZE=2048`,
all the patterns would silently be wrong — bits pointing to wrong positions.
We detect this and raise a clear `ValueError` with instructions.

**Test result:**
```
Loaded .npz → greeting: 0.451  ✓
Loaded .json → farewell: 0.608 ✓
```
Both formats round-trip correctly. Patterns survive sessions.

---

## BUILDLOG: v1.2 — File Learning + File Similarity

**Problem being solved:**  
Couldn't learn from actual files. Couldn't compare two documents for similarity.
These are the most natural use cases for a pattern recognition system.

**New module:** `fileops.py`

**`learn_from_file(sp, path, label=None)`**  
Reads a file, chunks it, learns patterns from chunks.  
If `label` given → supervised (all chunks are examples of that category).  
If `label=None` → unsupervised (let patterns crystallize from recurrence).

Chunking strategy by file type:
- `.txt`/`.md` → split on blank lines (paragraph-aware), then by word count with overlap
- `.csv` → one chunk per row (joined fields)
- `.json` → one chunk per top-level list item or dict value

**Design decision — overlapping chunks for text:**  
Non-overlapping chunks cut across sentence boundaries, breaking semantic units.
Using `chunk_size // 2` stride means adjacent chunks share half their words,
so patterns that span a boundary still get captured.

**`compare_files(sp, path_a, path_b)`**  
Encodes both files as chunk SDRs, then for each chunk in A finds the best
matching chunk in B, aggregates scores → overall similarity score.

Returns `FileSimilarityReport` with:
- `overall_score` — mean of best-match scores across all chunks of A
- `top_pairs` — highest scoring chunk pairs (what actually matched)
- `chunk_scores` — per-chunk score list (useful for plotting)

**Test result with two test files (AI + weather topics):**
```
Overall similarity: 0.147
Top pair (0.176): weather chunks matched each other  ✓
Second pair (0.118): ML/AI chunks matched each other  ✓
```
The system correctly identified thematically similar sections across two
files it had never been explicitly told anything about. Topic-sensitive
without any supervised labels.

**Bug during testing:**  
File chunker was producing only 2 chunks for the test files because the
whole AI paragraph and whole weather paragraph were each under `chunk_size=50`
words, so they didn't get sub-chunked. Result was correct but showed that
small files work at paragraph granularity, which is fine.

---

## BUILDLOG: v1.3 — Better On-The-Go Learning

**Problem being solved:**  
v1 learner was too discrete. It either promoted a pattern or it didn't.
Once promoted, a pattern never got stronger from new observations.
Candidates didn't decay, so stale/one-off inputs cluttered the candidate pool forever.
No way to know how confident the system was in any given pattern.

**Changes to `learner.py`:**

**1. Continuous strengthening (Hebbian-style)**  
Every time an input matches a known pattern (above threshold), we call
`memory.learn()` again with that input — re-bundling it into the prototype.
The prototype keeps improving with every matching observation, not just at
the moment of promotion.

Counter `_n_strengthened` tracks this. In v1 this was always 0.

**2. Candidate decay**  
Every `decay_every=50` observations, all candidates get their `strength`
multiplied by `decay_rate=0.85`. Candidates that aren't re-seen fade away
and get pruned when `strength < 0.1`.

This prevents the candidate pool from filling up with one-off inputs that
the system saw once and never again. Keeps memory clean.

Candidate pruning now uses `strength × count` as the sort key rather than
just `count` — a candidate seen 3 times recently is ranked above one seen
5 times long ago but now faded.

**3. Confidence tracking**  
`PatternConfidence` dataclass tracks `match_count` and `total_score` per label.
`confidence_report()` returns avg score per pattern — tells you which patterns
the system recognizes reliably (high avg score) vs shakily (low avg score).

**4. `on_the_go` mode**  
When `on_the_go=True`:
- `novelty_threshold` lowered by 0.05 (accepts weaker matches as "known")
- `promotion_threshold` lowered by 1 (promotes patterns faster)
- `candidate_match_threshold` lowered by 0.05 (more lenient candidate merging)

Designed for live input streams where you want the system adapting quickly
rather than being conservative.

**Bug: missing `self.max_candidates` assignment**  
When rewriting the learner, `max_candidates` was kept as a parameter but
`self.max_candidates = max_candidates` was accidentally omitted from `__init__`.
`_prune_candidates()` then crashed with `AttributeError` on first call.

Fix: added the missing `self.max_candidates = max_candidates` line.

**Lesson:** always grep for parameter names in methods before assuming they're assigned.

**Test results:**
```
Stream of 8 inputs → 2 patterns promoted (ML cluster, cat cluster)  ✓
"the cat naps on the mat" (unseen) → recognized at 0.529           ✓
"deep learning is complex" (unseen) → recognized at 0.157          ✓  
"quantum physics" → [novel]                                         ✓
```

---

## State of the system after v1.1 → v1.3

**Files:**
```
sym_pattern/
  __init__.py    — SymPattern API (updated with save/load, learn_file, compare_files, confidence_report)
  sdr.py         — SDR core (unchanged from v1)
  encoder.py     — encoders (unchanged from v1)
  memory.py      — pattern store (unchanged from v1)
  learner.py     — v1.3: +decay, +continuous strengthening, +confidence, +on_the_go
  persistence.py — NEW v1.1: save/load .npz and .json
  fileops.py     — NEW v1.2: file learning and file comparison
```

**Capabilities now:**
- All of v1: fuzzy matching, subword similarity, noise robustness, novelty/ambiguity detection
- Patterns persist to disk and reload (v1.1)
- Learn from .txt/.md/.csv/.json files (v1.2)
- Compare two files for similarity with chunk-level detail (v1.2)
- Continuous prototype strengthening from observations (v1.3)
- Candidate decay — stale candidates auto-pruned (v1.3)
- Confidence report per pattern (v1.3)
- on_the_go mode for fast adaptation to live streams (v1.3)

**Still not there yet (next targets):**
- Hierarchical composition (v2.0) — biggest remaining gap to NN capability
- Temporal / sequence memory (v2.1)
- Context-sensitive encoding (v2.2)

*Next entry: v2.0 — Hierarchical Pattern Composition*

---

### What's missing — the roadmap

Listed in priority order (each one is a meaningful capability jump):

**v2.0 — Hierarchical Pattern Composition**  
Stack the pattern matcher on top of itself.  
Recognized patterns become atoms for the next layer.  
Layer 1 sees tokens. Layer 2 sees token-patterns. Layer 3 sees pattern-patterns.  
This is the biggest single jump toward NN capability.

**v2.1 — Temporal / Sequence Memory**  
Track what patterns follow what. Build transition tables.  
System can predict what comes next, not just recognize what's here.

**v2.2 — Context-sensitive encoding**  
Same word, different bits in different contexts.  
Requires the pattern at position N to influence encoding at position N±k.
This is what attention mechanisms solve in transformers.

**v3.0 — Active recall / generation**  
Pattern → generate expected continuation.  
This is the flip side of recognition: not just "what is this?" but "what should come next?"

**v∞ — Symbol graph**  
Patterns connected by typed relationships (is-a, has-a, causes, before, after).  
This is full symbolic knowledge representation — where classical AI and
our SDR system finally merge.

---

*Log maintained by: Claude + developer, session by session.*  
