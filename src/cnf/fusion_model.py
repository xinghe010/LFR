"""CNF fusion variants, kept identical to the verified MPTP fusion logic."""
import torch
from torch import nn
import torch.nn.functional as F
from model import PremiseSelectionModel as OriginalModel


class PremiseSelectionModel(OriginalModel):
    def __init__(self, *args, fusion_mode="binary", gate_hidden=16, **kwargs):
        if fusion_mode not in ("binary", "continuous", "gated", "neural_control"):
            raise ValueError(fusion_mode)
        if kwargs.get("symbolic_baseline", "lfr") != "lfr":
            raise ValueError("Fusion variants require the LFR feature definition")
        super().__init__(*args, **kwargs)
        self.fusion_mode = fusion_mode
        self.trigger_weight = float(kwargs.get("rho_t") if kwargs.get("rho_t") is not None else kwargs.get("rho_w", 1.0))
        self.similarity_weight = float(kwargs.get("rho_s") if kwargs.get("rho_s") is not None else kwargs.get("rho_w", 1.0))
        self.trigger_enabled = bool(kwargs.get("use_trigger_rule", True))
        self.similarity_enabled = bool(kwargs.get("use_similarity_rule", True))
        if fusion_mode in ("gated", "neural_control"):
            channels = args[1] if len(args) > 1 else kwargs["node_out_channels"]
            if gate_hidden < 1:
                raise ValueError("gate_hidden must be positive")
            self.rule_gate = nn.Sequential(
                nn.Linear(2 * channels + 3, gate_hidden), nn.Tanh(),
                nn.Linear(gate_hidden, 2))

    def pooled_features(self, batch):
        hs, ht = self.initial(batch.x_s), self.initial(batch.x_t)
        hs = self.dag_emb(hs, batch.term_walk_index_s, batch.edge_index_s,
                          batch.edge_attr_s, batch.node_attr_s, batch.x_s_batch)
        ht = self.dag_emb(ht, batch.term_walk_index_t, batch.edge_index_t,
                          batch.edge_attr_t, batch.node_attr_t, batch.x_t_batch)
        return self.pooling(hs, batch.x_s_batch), self.pooling(ht, batch.x_t_batch)

    def fuse(self, base_logits, hs, ht, rules):
        if self.fusion_mode == "continuous":
            weights = torch.ones_like(rules)
        elif self.fusion_mode == "gated":
            context = torch.cat((hs, ht, rules, rules[:, 0:1] * rules[:, 1:2]), dim=1)
            weights = torch.sigmoid(self.rule_gate(context))
        elif self.fusion_mode == "neural_control":
            summaries = torch.stack((
                F.cosine_similarity(hs, ht, dim=1),
                (hs - ht).abs().mean(1), (hs * ht).mean(1)), dim=1)
            context = torch.cat((hs, ht, summaries), dim=1)
            residual = 2 * torch.sigmoid(self.rule_gate(context)) - 1
            bias = (self.trigger_weight * residual[:, 0] +
                    self.similarity_weight * residual[:, 1])
            return base_logits + torch.stack((torch.zeros_like(bias), bias), dim=1)
        else:
            raise ValueError("binary mode delegates to original forward")
        enabled = rules.new_tensor([float(self.trigger_enabled), float(self.similarity_enabled)])
        rho = rules.new_tensor([self.trigger_weight, self.similarity_weight])
        bias = (weights * rules * enabled * rho).sum(1)
        return base_logits + torch.stack((torch.zeros_like(bias), bias), dim=1)

    def predict_logits(self, batch):
        if self.fusion_mode == "binary":
            return super().predict_logits(batch)
        hs, ht = self.pooled_features(batch)
        base = (self.classifier(hs, ht, torch.zeros_like(batch.similarity))
                if self.use_classifier_constraint else self.classifier(hs, ht))
        if not self.use_classifier_constraint:
            return base
        if self.fusion_mode == "neural_control":
            return self.fuse(base, hs, ht, None)
        features = getattr(batch, "symbolic_features", None)
        if features is None or features.ndim != 2 or features.shape[1] < 6:
            raise ValueError("continuous fusion requires verified six-column symbolic cache")
        trigger = batch.similarity.reshape(-1, 2)[:, 0]
        continuous = features[:, 5].to(trigger.dtype)
        rules = torch.stack((trigger, continuous), dim=1)
        if not torch.isfinite(rules).all() or (rules < 0).any() or (rules > 1).any():
            raise ValueError("rule features must be finite in [0,1]")
        return self.fuse(base, hs, ht, rules)
