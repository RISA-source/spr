# sym-pattern v1

Symbolic AI pattern recognition system — NN-grade matching without neural networks.

## Core idea

Every symbol is a **Sparse Distributed Representation (SDR)** — a sparse binary vector where:
- Similar things have **overlapping bits** (similarity = overlap, naturally)
- Matching is **graded** (confidence score 0.0–1.0, not just true/false)
- Representations are **composable** (union = generalize, intersection = constrain)
- Everything is **inspectable** (no black box weights)

## Architecture

```
Raw Input
    ↓
Encoder          — text/numbers/sequences → SDR
    ↓
PatternMemory    — stores prototypes, fuzzy matching
    ↓
UnsupervisedLearner — discovers patterns from raw input stream
    ↓
SymPattern       — high-level API tying it all together
```

## Quickstart

```python
from sym_pattern import SymPattern

sp = SymPattern()

# Supervised: teach it categories
sp.teach("greeting", ["hello world", "hi there", "hey how are you"])
sp.teach("farewell", ["goodbye", "see you later", "bye bye"])

# Recognize with confidence scores
results = sp.recognize("hello friend")
# → [Match('greeting', score=0.71), Match('farewell', score=0.12)]

best = sp.best_match("hey what's up")
# → Match('greeting', score=0.68)

# Unsupervised: let it discover patterns
sp2 = SymPattern(unsupervised=True)
for text in corpus:
    sp2.observe(text)
# Recurring patterns crystallize automatically
```

## Run the demo

```bash
cd sym-pattern
python demo.py
```

## Files

```
sym_pattern/
  __init__.py    — public API, SymPattern class
  sdr.py         — Sparse Distributed Representation core
  encoder.py     — input encoders (token, scalar, sequence, composite)
  memory.py      — pattern store + fuzzy matching
  learner.py     — unsupervised pattern discovery
demo.py          — three demos showing all capabilities
```

## Port notes (Go/Rust)

Each module has clean boundaries and no Python-specific magic:
- `SDR` is just a bool array + operations → trivial in Go/Rust
- `PatternMemory` is a hashmap + numpy ops → standard everywhere  
- `UnsupervisedLearner` is pure logic, no framework dependencies
- `numpy` used only for array ops → replace with ndarray (Rust) or gonum (Go)

## v1 capabilities

- Fuzzy/partial matching with confidence scores
- Noise robustness (corrupted input still recognized)
- Subword similarity (similar words share bits automatically)
- Order-sensitive sequence encoding
- Online unsupervised learning
- Novelty detection
- Ambiguity detection
- Composable representations

## What's next (v2)

- Hierarchical pattern composition (patterns of patterns)
- Temporal sequence memory (what comes after what)
- Active recall (pattern → generate expected continuation)
- Symbol graph (associative memory between patterns)
