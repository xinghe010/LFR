"""Same-classifier symbolic controls, isolated from the frozen full model.

All variants retain the identical TW encoder and classifier. Only the
label-blind scalar added to the positive-class logit changes. ``combination``
delegates to the current continuous LFR implementation exactly.
"""

import torch

from fusion_model import PremiseSelectionModel as ContinuousModel
from model import SYMBOLIC_FEATURE_INDICES


VARIANTS = ("overlap", "idf", "sine", "trigger", "weighted_similarity",
            "combination")


class PremiseSelectionModel(ContinuousModel):
    def __init__(self, *args, symbolic_variant="combination", rule_weight=1.0,
                 **kwargs):
        if symbolic_variant not in VARIANTS:
            raise ValueError(f"Unknown symbolic variant: {symbolic_variant}")
        if kwargs.get("fusion_mode", "continuous") != "continuous":
            raise ValueError("Symbolic controls use continuous-logit fusion")
        kwargs["fusion_mode"] = "continuous"
        kwargs["symbolic_baseline"] = "lfr"
        super().__init__(*args, **kwargs)
        if not self.use_classifier_constraint:
            raise ValueError("Symbolic controls require the shared constrained classifier")
        self.symbolic_variant = symbolic_variant
        self.rule_weight = float(rule_weight)
        if self.rule_weight < 0:
            raise ValueError("rule_weight must be nonnegative")

    def predict_logits(self, batch):
        if self.symbolic_variant == "combination":
            return super().predict_logits(batch)
        features = getattr(batch, "symbolic_features", None)
        if features is None or features.ndim != 2 or features.shape[1] != 6:
            raise ValueError("six-column label-blind symbolic cache is required")
        index = SYMBOLIC_FEATURE_INDICES[self.symbolic_variant]
        value = features[:, index]
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("symbolic feature must be finite in [0,1]")
        hs, ht = self.pooled_features(batch)
        base = self.classifier(hs, ht, torch.zeros_like(batch.similarity))
        bias = self.rule_weight * value.to(base.dtype)
        return base + torch.stack((torch.zeros_like(bias), bias), dim=1)
