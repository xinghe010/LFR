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
        if self.gpuid >= 0:
            self.one = self.one.cuda()

    def logic(self, similarity):

        simi = similarity

        R = Variable(simi, requires_grad=False)
        if self.gpuid != -1:
            R = R.cuda()

        d = R

        return d

    def forward(self, similarity):
        return self.logic(similarity)
