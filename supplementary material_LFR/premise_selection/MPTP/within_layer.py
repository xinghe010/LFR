import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from similarity import *
import time

class WithinLayer(torch.nn.Module):
    def __init__(self, gpuid, rho_w):
        super(WithinLayer, self).__init__()

        self.within_constr = self.get_within_constr()

        self.zero = Variable(torch.zeros(1), requires_grad=False)
        rho_w = torch.ones(1) * rho_w
        if gpuid != -1:
            rho_w = rho_w.cuda()
            self.zero = self.zero.cuda()
        self.rho_w = nn.Parameter(rho_w, requires_grad=False)

        if len(self.within_constr) != 0:
            print('within-layer constraint enabled')

    def get_within_constr(self):
        layers = []

        layers.append(Similarity(0))
        layers.append(Similarity(0))

        return layers

    def forward(self, pred_use_y,  similarity):

        simi = [0]*2
        for i in range(2):

            simi[i] = self.within_constr[i](similarity[i::2]).contiguous()
        rho_w = self.rho_w

        constrained_pred_use_y = pred_use_y + rho_w * torch.max(torch.ones_like(simi[0]), simi[0]+simi[1] - 1)

        return constrained_pred_use_y

if __name__ == '__main__':
    pass
