"""
persistence.py — Save and Load learned patterns (v1.1)

Serializes PatternMemory to disk so patterns survive between sessions.
Two formats:
  - .npz  (numpy native, fast, compact, recommended)
  - .json (human-readable, inspectable, slower on large stores)

Usage:
    save(memory, "my_patterns.npz")
    memory = load("my_patterns.npz")
"""

import json
import numpy as np
from pathlib import Path
from .memory import PatternMemory
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY


def save(memory: PatternMemory, path: str):
    """
    Save a PatternMemory to disk.
    Format chosen automatically by file extension:
      .npz  → numpy binary (fast, compact)
      .json → human-readable
    """
    path = Path(path)
    if path.suffix == ".json":
        _save_json(memory, path)
    else:
        _save_npz(memory, path)


def load(path: str) -> PatternMemory:
    """
    Load a PatternMemory from disk.
    Format detected automatically by file extension.
    """
    path = Path(path)
    if path.suffix == ".json":
        return _load_json(path)
    else:
        return _load_npz(path)


# ------------------------------------------------------------------ #
#  NPZ format (recommended)                                           #
# ------------------------------------------------------------------ #

def _save_npz(memory: PatternMemory, path: Path):
    """
    Pack all prototype bit arrays into one .npz archive.
    Also stores labels and example counts as metadata.
    """
    arrays = {}
    labels = list(memory.prototypes.keys())
    for i, label in enumerate(labels):
        arrays[f"proto_{i}"] = memory.prototypes[label].bits.astype(np.uint8)

    arrays["_labels"] = np.array(labels, dtype=object)
    arrays["_counts"] = np.array(
        [memory.example_counts.get(l, 1) for l in labels], dtype=np.int32
    )
    arrays["_sdr_size"] = np.array([SDR_SIZE], dtype=np.int32)
    arrays["_threshold"] = np.array([memory.match_threshold], dtype=np.float32)

    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> PatternMemory:
    data = np.load(path, allow_pickle=True)

    sdr_size_saved = int(data["_sdr_size"][0])
    if sdr_size_saved != SDR_SIZE:
        raise ValueError(
            f"SDR size mismatch: file has {sdr_size_saved}, current config is {SDR_SIZE}. "
            f"Set SDR_SIZE={sdr_size_saved} before loading."
        )

    threshold = float(data["_threshold"][0])
    memory = PatternMemory(match_threshold=threshold)

    labels = list(data["_labels"])
    counts = list(data["_counts"])

    for i, (label, count) in enumerate(zip(labels, counts)):
        bits = data[f"proto_{i}"].astype(bool)
        sdr = SDR(bits=bits, label=label)
        memory.prototypes[label] = sdr
        memory.example_counts[label] = int(count)
        # Reconstruct a placeholder example bank (single entry)
        memory._example_bank[label] = [sdr]

    return memory


# ------------------------------------------------------------------ #
#  JSON format (human readable)                                       #
# ------------------------------------------------------------------ #

def _save_json(memory: PatternMemory, path: Path):
    out = {
        "sdr_size": SDR_SIZE,
        "match_threshold": memory.match_threshold,
        "patterns": {}
    }
    for label, proto in memory.prototypes.items():
        out["patterns"][label] = {
            "active_indices": proto.active_indices().tolist(),
            "example_count": memory.example_counts.get(label, 1),
        }
    path.write_text(json.dumps(out, indent=2))


def _load_json(path: Path) -> PatternMemory:
    data = json.loads(path.read_text())

    if data["sdr_size"] != SDR_SIZE:
        raise ValueError(
            f"SDR size mismatch: file has {data['sdr_size']}, current is {SDR_SIZE}."
        )

    memory = PatternMemory(match_threshold=data["match_threshold"])
    for label, info in data["patterns"].items():
        sdr = SDR.from_indices(info["active_indices"], label=label)
        memory.prototypes[label] = sdr
        memory.example_counts[label] = info["example_count"]
        memory._example_bank[label] = [sdr]

    return memory