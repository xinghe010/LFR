import torch
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Similarity(torch.nn.Module):
    def __init__(self, gpuid):
        super(Similarity, self).__init__()

        self.one = Variable(torch.Tensor([1.0]), requires_grad=False)
        self.gpuid = gpuid

    def logic(self, similarity):

        simi = similarity

        R = Variable(simi, requires_grad=False)

        d = R

        return d

    def forward(self, similarity):
        return self.logic(similarity)
