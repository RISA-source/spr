"""
fileops.py — File Learning and File Similarity (v1.2)

Two capabilities:
  1. learn_from_file(path)  — chunk a text file, learn patterns from it
  2. compare_files(a, b)    — score similarity between two files section by section

Supports: .txt, .md, .csv (per row), .json (per entry if list/dict)

Usage:
    from sym_pattern.fileops import learn_from_file, compare_files

    sp = SymPattern()
    learn_from_file(sp, "corpus.txt", label="my_corpus")

    results = compare_files(sp, "file_a.txt", "file_b.txt")
    for r in results:
        print(r)
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
from .sdr import SDR


# ------------------------------------------------------------------ #
#  Chunking — split file into meaningful pieces                       #
# ------------------------------------------------------------------ #

def _read_chunks(path: Path, chunk_size: int = 50) -> Iterator[str]:
    """
    Read a file and yield text chunks.
    Strategy depends on file type:
      .txt/.md  → split by blank lines (paragraphs), then by chunk_size words
      .csv      → one chunk per row (joined fields)
      .json     → one chunk per top-level item if list, per value if dict
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        yield from _read_csv(path)
    elif suffix == ".json":
        yield from _read_json(path)
    else:
        yield from _read_text(path, chunk_size)


def _read_text(path: Path, chunk_size: int) -> Iterator[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Split on blank lines first (paragraph-aware)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        # If paragraph is short, yield as-is
        if len(words) <= chunk_size:
            yield para
        else:
            # Slide through with overlap for continuity
            for i in range(0, len(words), chunk_size // 2):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    yield chunk


def _read_csv(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            chunk = " ".join(cell.strip() for cell in row if cell.strip())
            if chunk:
                yield chunk


def _read_json(path: Path) -> Iterator[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                yield " ".join(str(v) for v in item.values())
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                yield v


# ------------------------------------------------------------------ #
#  File learning                                                       #
# ------------------------------------------------------------------ #

def learn_from_file(sp, path: str, label: str = None, chunk_size: int = 50) -> dict:
    """
    Learn patterns from a file.

    If label is given: all chunks taught as supervised examples of that label.
    If label is None: unsupervised — chunks are observed, patterns crystallize.

    Returns summary dict with chunk count and any patterns promoted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")

    chunks = list(_read_chunks(path, chunk_size=chunk_size))
    promoted = []

    if label:
        # Supervised: all chunks → one category
        sp.teach(label, chunks)
    else:
        # Unsupervised: observe each chunk
        for chunk in chunks:
            result = sp.observe(chunk)
            if result and result.score >= 0.9:
                promoted.append(result.label)

    return {
        "file": str(path),
        "chunks": len(chunks),
        "mode": "supervised" if label else "unsupervised",
        "label": label,
        "promoted": promoted,
    }


# ------------------------------------------------------------------ #
#  File similarity                                                     #
# ------------------------------------------------------------------ #

@dataclass
class ChunkSimilarity:
    chunk_a: str
    chunk_b: str
    score: float
    label_a: str       # best matching pattern for chunk_a
    label_b: str       # best matching pattern for chunk_b

    def __repr__(self):
        bar = "█" * int(self.score * 20)
        return (
            f"Sim({self.score:.3f}) |{bar:<20}|\n"
            f"  A: {self.chunk_a[:60]!r}\n"
            f"  B: {self.chunk_b[:60]!r}"
        )


@dataclass
class FileSimilarityReport:
    file_a: str
    file_b: str
    overall_score: float
    top_pairs: list[ChunkSimilarity]
    chunk_scores: list[float]

    def summary(self) -> str:
        bar = "█" * int(self.overall_score * 40)
        lines = [
            f"\nFile Similarity Report",
            f"  A: {self.file_a}",
            f"  B: {self.file_b}",
            f"  Overall: |{bar:<40}| {self.overall_score:.3f}",
            f"  Chunks compared: {len(self.chunk_scores)}",
            f"\nTop similar pairs:",
        ]
        for pair in self.top_pairs[:5]:
            lines.append(f"  {pair.score:.3f}  {pair.chunk_a[:50]!r}")
            lines.append(f"         ↔ {pair.chunk_b[:50]!r}")
        return "\n".join(lines)


def compare_files(
    sp,
    path_a: str,
    path_b: str,
    chunk_size: int = 50,
    top_k_pairs: int = 10,
) -> FileSimilarityReport:
    """
    Compare two files for similarity using learned SDR patterns.

    Strategy:
      1. Chunk both files
      2. Encode each chunk as an SDR
      3. For each chunk in A, find best matching chunk in B
      4. Aggregate scores → overall similarity

    Returns a FileSimilarityReport with overall score and top matching pairs.
    """
    path_a, path_b = Path(path_a), Path(path_b)

    chunks_a = list(_read_chunks(path_a, chunk_size))
    chunks_b = list(_read_chunks(path_b, chunk_size))

    if not chunks_a or not chunks_b:
        raise ValueError("One or both files produced no chunks.")

    # Encode all chunks
    sdrs_a = [sp.encode(c) for c in chunks_a]
    sdrs_b = [sp.encode(c) for c in chunks_b]

    # For each chunk in A, find best match in B
    all_pairs: list[ChunkSimilarity] = []
    chunk_scores: list[float] = []

    for ca, sa in zip(chunks_a, sdrs_a):
        best_score = 0.0
        best_cb = ""
        for cb, sb in zip(chunks_b, sdrs_b):
            score = sa.overlap_score(sb)
            if score > best_score:
                best_score = score
                best_cb = cb

        # Get pattern labels if available
        res_a = sp.memory.match_all(sa, top_k=1)
        res_b = sp.memory.match_all(sp.encode(best_cb), top_k=1)
        lab_a = res_a[0].label if res_a else "unknown"
        lab_b = res_b[0].label if res_b else "unknown"

        chunk_scores.append(best_score)
        all_pairs.append(ChunkSimilarity(
            chunk_a=ca,
            chunk_b=best_cb,
            score=best_score,
            label_a=lab_a,
            label_b=lab_b,
        ))

    all_pairs.sort(key=lambda p: p.score, reverse=True)
    overall = float(sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 0.0

    return FileSimilarityReport(
        file_a=str(path_a),
        file_b=str(path_b),
        overall_score=overall,
        top_pairs=all_pairs[:top_k_pairs],
        chunk_scores=chunk_scores,
    )