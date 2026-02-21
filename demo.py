"""
demo.py — sym_pattern v1 demos

Run this to see the system working. Three demos:
  1. Supervised: teach it categories, test fuzzy recognition
  2. Unsupervised: show it a raw corpus, watch patterns emerge
  3. SDR properties: show robustness, compositionality, similarity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sym_pattern import SymPattern, SDR, TokenEncoder, SequenceEncoder


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ------------------------------------------------------------------ #
#  Demo 1: Supervised pattern recognition                             #
# ------------------------------------------------------------------ #

def demo_supervised():
    separator("DEMO 1: Supervised Recognition")

    sp = SymPattern()

    # Teach categories with multiple examples
    sp.teach("greeting", [
        "hello there",
        "hi how are you",
        "hey what's up",
        "good morning",
        "good evening friend",
    ])

    sp.teach("farewell", [
        "goodbye see you",
        "bye bye take care",
        "see you later",
        "farewell and good luck",
        "until next time",
    ])

    sp.teach("question", [
        "what is your name",
        "how does this work",
        "can you help me",
        "where are you from",
        "why did this happen",
    ])

    print("\n→ Testing recognition on seen-ish inputs:")
    tests = [
        "hello friend",           # should → greeting
        "bye see you tomorrow",   # should → farewell
        "what do you think",      # should → question
        "hey good morning mate",  # should → greeting
    ]

    for text in tests:
        results = sp.recognize(text, top_k=3)
        print(f"\n  Input: {text!r}")
        for r in results:
            bar = "█" * int(r.score * 30)
            print(f"    {r.label:<12} {bar:<30} {r.score:.3f}")

    print("\n→ Testing on noisy / corrupted inputs:")
    noisy_tests = [
        "helo ther",               # typo-style
        "godbye see u",            # informal
        "wut is ur name",          # very informal
    ]
    for text in noisy_tests:
        best = sp.best_match(text)
        if best:
            print(f"  {text!r:35} → {best.label} ({best.score:.3f})")
        else:
            print(f"  {text!r:35} → [no match]")

    print("\n→ Novelty detection:")
    novel_tests = [
        "the quick brown fox",     # should be novel
        "hello world",             # should match greeting
        "pizza with extra cheese", # should be novel
    ]
    for text in novel_tests:
        novel = sp.is_novel(text)
        print(f"  {text!r:35} novel={novel}")


# ------------------------------------------------------------------ #
#  Demo 2: Unsupervised learning                                      #
# ------------------------------------------------------------------ #

def demo_unsupervised():
    separator("DEMO 2: Unsupervised Pattern Discovery")

    sp = SymPattern(unsupervised=True, promotion_threshold=3)

    # Feed a corpus without any labels
    corpus = [
        # These should cluster together
        "the cat sat on the mat",
        "a cat is sitting on a mat",
        "cats like to sit on mats",
        "the cat rested on the mat",
        # These should form another cluster
        "machine learning is powerful",
        "deep learning models are complex",
        "neural networks learn from data",
        "machine learning requires data",
        # And another
        "the weather is nice today",
        "it is a sunny day outside",
        "beautiful weather we are having",
        "lovely day for a walk outside",
        # Singletons — should stay as candidates
        "the president signed a bill",
        "quantum entanglement is fascinating",
    ]

    print("\n→ Feeding corpus unsupervised...")
    promoted = []
    for text in corpus:
        result = sp.observe(text)
        if result and result.score >= 0.9:  # freshly promoted
            promoted.append((text, result.label))

    print(f"\n→ After {len(corpus)} inputs:")
    stats = sp.learner.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n→ Recognizing against discovered patterns:")
    test_inputs = [
        "a cat sitting on the mat",     # should match cat-cluster
        "neural networks are amazing",  # should match ML-cluster
        "sunny and warm weather today", # should match weather-cluster
    ]
    for text in test_inputs:
        results = sp.recognize(text, top_k=2)
        print(f"\n  {text!r}")
        for r in results[:2]:
            print(f"    pattern_{r.label[-4:]} score={r.score:.3f}")


# ------------------------------------------------------------------ #
#  Demo 3: SDR properties                                             #
# ------------------------------------------------------------------ #

def demo_sdr_properties():
    separator("DEMO 3: SDR Properties (Robustness & Similarity)")

    enc = TokenEncoder(subword_overlap=True)

    print("\n→ Subword similarity (similar words share bits):")
    word_groups = [
        ["run", "running", "runner", "runs"],
        ["cat", "cats", "catlike", "category"],
        ["play", "playing", "player", "playful"],
    ]
    for group in word_groups:
        sdrs = [enc.encode(w) for w in group]
        print(f"\n  Group: {group}")
        for i, (wa, sa) in enumerate(zip(group, sdrs)):
            for wb, sb in zip(group[i+1:], sdrs[i+1:]):
                sim = sa.overlap_score(sb)
                bar = "█" * int(sim * 20)
                print(f"    {wa} ↔ {wb}: {bar:<20} {sim:.3f}")


    print("\n→ Noise robustness:")
    sp_noise = SymPattern()
    sp_noise.teach("animal", [
        "the cat sat on the mat",
        "a dog ran through the park",
        "birds fly high in the sky",
        "fish swim deep in the ocean",
        "horses gallop across open fields",
    ])

    original = sp_noise.encode("the cat sat on the mat")
    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        noisy = original.add_noise(flip_rate=noise_level)
        results = sp_noise.memory.match_all(noisy, top_k=1)
        score = results[0].score if results else 0.0
        label = results[0].label if results else "none"
        bar = "█" * int(score * 20)
        print(f"  noise={noise_level:.1f}  {bar:<20} score={score:.3f}  match={label}")

    print("\n→ Compositionality (union/intersection):")
    se = SequenceEncoder()
    s1 = se.encode("big red dog")
    s2 = se.encode("small red cat")
    union = s1.union(s2, label="union")
    inter = s1.intersection(s2, label="intersection")
    print(f"  s1 active bits:           {s1.n_active()}")
    print(f"  s2 active bits:           {s2.n_active()}")
    print(f"  union active bits:        {union.n_active()} (more general)")
    print(f"  intersection active bits: {inter.n_active()} (shared structure)")
    print(f"  s1 ↔ s2 similarity:       {s1.overlap_score(s2):.3f}")
    print(f"  inter subsumes s1:        {inter.subsumes(s1, threshold=0.3)}")


# ------------------------------------------------------------------ #
#  Run all demos                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    demo_supervised()
    demo_unsupervised()
    demo_sdr_properties()
    print("\n" + "="*60)
    print("  sym_pattern v1 — all demos complete")
    print("="*60 + "\n")
