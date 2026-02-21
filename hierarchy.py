"""
hierarchy.py — Hierarchical Pattern Composition (v2.0)

The biggest single jump toward NN capability in sym-pattern.

The problem with a single flat pattern layer:
  - It can recognize "cat" and "sat" and "mat"
  - But it can't recognize "thing that sits on things" as an abstraction
  - It can't recognize that [greeting → question → farewell] is a conversation shape
  - It has no concept of LEVELS of abstraction

What neural nets do across layers:
  Layer 1: edges
  Layer 2: shapes (composed edges)
  Layer 3: objects (composed shapes)
  Layer 4: scenes (composed objects)

What we do:
  Layer 1: token SDRs → recognized as low-level patterns (e.g. "animal_action")
  Layer 2: pattern-label SDRs → recognized as mid-level compositions (e.g. "narrative_unit")
  Layer 3: composition SDRs → recognized as high-level abstractions (e.g. "story_arc")

Key insight: at each layer, the RECOGNIZED LABELS from the layer below become
the INPUT ATOMS for the layer above. Labels get re-encoded as SDRs and fed up.
This is compositionality — patterns made of patterns made of patterns.

Each layer is a full PatternMemory + UnsupervisedLearner.
The HierarchicalSystem connects them and manages the feed-forward signal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .sdr import SDR, SDR_SIZE, SDR_SPARSITY
from .memory import PatternMemory, MatchResult
from .encoder import TokenEncoder, SequenceEncoder
from .learner import UnsupervisedLearner


# ------------------------------------------------------------------ #
#  Layer activation — what one layer produces for the next            #
# ------------------------------------------------------------------ #

@dataclass
class LayerActivation:
    """
    What a layer produces when it processes input.
    This is passed UP to the next layer as its input.
    """
    layer_id: int
    input_sdr: SDR                      # what came in
    matches: list[MatchResult]          # what was recognized
    activation_sdr: SDR                 # re-encoded recognized labels → SDR for next layer
    is_novel: bool                      # nothing matched at this layer

    @property
    def best(self) -> Optional[MatchResult]:
        return self.matches[0] if self.matches else None

    @property
    def best_label(self) -> Optional[str]:
        return self.best.label if self.best else None

    @property
    def best_score(self) -> float:
        return self.best.score if self.best else 0.0

    def __repr__(self):
        if self.is_novel:
            return f"Layer{self.layer_id}(novel)"
        return f"Layer{self.layer_id}({self.best_label!r} @ {self.best_score:.2f})"


# ------------------------------------------------------------------ #
#  Single layer                                                        #
# ------------------------------------------------------------------ #

class PatternLayer:
    """
    One layer of the hierarchy. Wraps a PatternMemory + UnsupervisedLearner.

    Receives SDR input → recognizes patterns → produces activation SDR for next layer.

    The activation SDR is built by encoding the recognized label(s) as SDRs
    and bundling them — this is how recognition at layer N becomes input at layer N+1.
    """

    def __init__(
        self,
        layer_id: int,
        novelty_threshold: float = 0.20,
        promotion_threshold: int = 3,
        on_the_go: bool = False,
        label_encoder: TokenEncoder = None,
    ):
        self.layer_id = layer_id
        self.memory = PatternMemory(match_threshold=novelty_threshold)
        self.learner = UnsupervisedLearner(
            memory=self.memory,
            novelty_threshold=novelty_threshold,
            promotion_threshold=promotion_threshold,
            on_the_go=on_the_go,
        )
        # Each layer has its own label encoder — converts its own recognized
        # label strings back into SDRs for the layer above
        self.label_encoder = label_encoder or TokenEncoder(subword_overlap=True)
        self._n_processed = 0

    def process(self, input_sdr: SDR, learn: bool = True) -> LayerActivation:
        """
        Process one input SDR through this layer.

        learn=True: update prototypes from this input (training mode)
        learn=False: recognize only, don't update (inference mode)
        """
        self._n_processed += 1

        # Recognize
        matches = self.memory.match_all(input_sdr, top_k=3)
        valid = [m for m in matches if m.score >= self.learner.novelty_threshold]
        is_novel = len(valid) == 0

        # Learn if in training mode
        if learn:
            if valid:
                # Strengthen best match
                self.memory.learn(valid[0].label, input_sdr)
            else:
                # Observe as potential new pattern
                self.learner.observe(input_sdr)

        # Build activation SDR for next layer
        # Encode top matching labels and bundle them
        activation_sdr = self._build_activation(valid, input_sdr)

        return LayerActivation(
            layer_id=self.layer_id,
            input_sdr=input_sdr,
            matches=valid,
            activation_sdr=activation_sdr,
            is_novel=is_novel,
        )

    def teach(self, label: str, sdr: SDR):
        """Supervised teaching — directly assign this SDR to a label."""
        self.learner.teach(label, sdr)

    def _build_activation(self, matches: list[MatchResult], fallback: SDR) -> SDR:
        """
        Build the SDR that gets passed to the next layer.

        If we recognized something: encode the top label(s) as SDRs and bundle.
        If novel: pass through the raw input SDR (so upper layers can still learn).
        """
        if not matches:
            return fallback

        # Encode top-1 or top-2 labels (weighted by score)
        label_sdrs = []
        for m in matches[:2]:
            lsdr = self.label_encoder.encode(m.label)
            label_sdrs.append(lsdr)

        if len(label_sdrs) == 1:
            return label_sdrs[0]
        return label_sdrs[0].bundle(*label_sdrs[1:])

    def stats(self) -> dict:
        return {
            "layer": self.layer_id,
            "processed": self._n_processed,
            "patterns": len(self.memory),
            **self.learner.stats(),
        }

    def __repr__(self):
        return f"PatternLayer(id={self.layer_id}, patterns={len(self.memory)}, processed={self._n_processed})"


# ------------------------------------------------------------------ #
#  Hierarchical system — stack of layers                              #
# ------------------------------------------------------------------ #

@dataclass
class HierarchicalResult:
    """Full result from processing through all layers."""
    layer_activations: list[LayerActivation]   # one per layer
    depth_reached: int                          # deepest layer that fired
    abstract_label: Optional[str]              # highest-level recognition

    @property
    def bottom(self) -> LayerActivation:
        return self.layer_activations[0]

    @property
    def top(self) -> LayerActivation:
        return self.layer_activations[-1]

    def summary(self) -> str:
        parts = []
        for act in self.layer_activations:
            if act.is_novel:
                parts.append(f"L{act.layer_id}:[novel]")
            else:
                parts.append(f"L{act.layer_id}:{act.best_label}({act.best_score:.2f})")
        return " → ".join(parts)

    def __repr__(self):
        return f"HResult({self.summary()})"


class HierarchicalSystem:
    """
    A stack of PatternLayers that process input bottom-up.

    Layer 0: sees raw token/sequence SDRs
    Layer 1: sees label-SDRs from Layer 0 activations
    Layer 2: sees label-SDRs from Layer 1 activations
    ...

    Each layer learns at its own level of abstraction.
    The system as a whole builds a multi-level understanding of its input.

    Teaching: you can teach at any layer directly (supervised),
    or let all layers learn unsupervised from a stream.
    """

    def __init__(
        self,
        n_layers: int = 3,
        novelty_thresholds: list[float] = None,
        promotion_thresholds: list[int] = None,
        on_the_go: bool = False,
    ):
        """
        n_layers: how many abstraction levels (2-4 is typical, 3 is default)
        novelty_thresholds: per-layer thresholds (lower layers more sensitive)
        promotion_thresholds: per-layer promotion counts
        """
        # Default: lower layers are more sensitive (lower threshold)
        # Upper layers are more conservative (higher threshold)
        if novelty_thresholds is None:
            novelty_thresholds = [max(0.15, 0.20 + i * 0.03) for i in range(n_layers)]
        if promotion_thresholds is None:
            promotion_thresholds = [max(2, 3 + i) for i in range(n_layers)]

        self.layers: list[PatternLayer] = [
            PatternLayer(
                layer_id=i,
                novelty_threshold=novelty_thresholds[i],
                promotion_threshold=promotion_thresholds[i],
                on_the_go=on_the_go,
            )
            for i in range(n_layers)
        ]
        self.n_layers = n_layers
        self._n_processed = 0

    def process(self, input_sdr: SDR, learn: bool = True) -> HierarchicalResult:
        """
        Feed one SDR through all layers bottom-up.
        Each layer's activation becomes the next layer's input.
        """
        self._n_processed += 1
        activations = []
        current_sdr = input_sdr

        for layer in self.layers:
            activation = layer.process(current_sdr, learn=learn)
            activations.append(activation)
            # Feed this layer's activation SDR to the next layer
            current_sdr = activation.activation_sdr

        # Find deepest layer that actually recognized something
        depth = 0
        abstract_label = None
        for i, act in enumerate(activations):
            if not act.is_novel:
                depth = i
                abstract_label = act.best_label

        return HierarchicalResult(
            layer_activations=activations,
            depth_reached=depth,
            abstract_label=abstract_label,
        )

    def process_sequence(
        self,
        sdrs: list[SDR],
        learn: bool = True,
    ) -> list[HierarchicalResult]:
        """Process a sequence of SDRs. Returns one result per input."""
        return [self.process(sdr, learn=learn) for sdr in sdrs]

    def teach_layer(self, layer_id: int, label: str, sdr: SDR):
        """Directly teach a specific layer."""
        if 0 <= layer_id < self.n_layers:
            self.layers[layer_id].teach(label, sdr)

    def stats(self) -> list[dict]:
        return [layer.stats() for layer in self.layers]

    def summary(self) -> str:
        lines = [f"HierarchicalSystem ({self.n_layers} layers, {self._n_processed} processed)"]
        for layer in self.layers:
            s = layer.stats()
            lines.append(
                f"  Layer {s['layer']}: {s['patterns']} patterns, "
                f"{s['promoted']} promoted, {s['strengthened']} strengthened"
            )
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()