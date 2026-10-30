import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from similarity import *
import time

class WithinLayer(torch.nn.Module):
    """Inject independent trigger and weighted-similarity logit evidence."""
    def __init__(self, gpuid, rho_w=1.0, rho_s=None,
                 use_trigger_rule=True, use_similarity_rule=True):
        super(WithinLayer, self).__init__()

        self.within_constr = self.get_within_constr()

        self.zero = Variable(torch.zeros(1), requires_grad=False)
        rho_w = torch.ones(1) * rho_w
        self.rho_w = nn.Parameter(rho_w, requires_grad=False)
        self.rho_s = float(rho_w.item() if rho_s is None else rho_s)
        self.use_trigger_rule = bool(use_trigger_rule)
        self.use_similarity_rule = bool(use_similarity_rule)

        if len(self.within_constr) != 0:
            print('within-layer constraint enabled')

    def get_within_constr(self):
        layers = []

        layers.append(Similarity(0))
        layers.append(Similarity(0))

        return layers

    def forward(self, pred_use_y,  similarity):

        features = similarity.to(device=pred_use_y.device,
                                 dtype=pred_use_y.dtype).reshape(-1, 2)
        if features.size(0) != pred_use_y.size(0):
            raise ValueError(
                f"Expected two rule values per example, got {tuple(similarity.shape)} "
                f"for {pred_use_y.size(0)} predictions")
        bias = torch.zeros_like(pred_use_y)
        if self.use_trigger_rule:
            bias = bias + self.rho_w.to(pred_use_y.dtype) * features[:, 0]
        if self.use_similarity_rule:
            bias = bias + self.rho_s * features[:, 1]
        constrained_pred_use_y = pred_use_y + bias

        return constrained_pred_use_y

if __name__ == '__main__':
    pass
