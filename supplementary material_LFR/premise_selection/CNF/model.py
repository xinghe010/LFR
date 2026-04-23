import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool

from within_layer import *

import numpy as np

class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.8)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x

class MLPBlock2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False,
                 constrain=False,
                 rho_w=1.0):
        super(MLPBlock2, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.constrain = constrain
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.8)
        self.reset_parameters()

        if self.constrain:
            self.similarity = WithinLayer(0, rho_w)

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, simi):

        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)

        if self.constrain:
            x[:, 1] = self.similarity(x[:, 1], simi)

        x = self.activation(x)
        return x

class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output

class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers,
                 use_lce_and=True, use_lce_or=True, use_lce_not=True,
                 use_lce_contains=True, use_lce_equivalence=True,
                 use_lce_forall=True, use_lce_exists=True,
                 lce_scale=0.5):
        super(DAGEmbedding, self).__init__()
        self.K = layers
        self.use_lce_and = use_lce_and
        self.use_lce_or = use_lce_or
        self.use_lce_not = use_lce_not
        self.use_lce_contains = use_lce_contains
        self.use_lce_equivalence = use_lce_equivalence
        self.use_lce_forall = use_lce_forall
        self.use_lce_exists = use_lce_exists
        self.lce_scale = lce_scale
        self.F_T = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_M = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_B = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_trans_TW = nn.ModuleList([
            MLPBlock(node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_AND = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_OR = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_FORALL = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_EXISTS = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_CONTAINS = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_NOT = nn.ModuleList([
            MLPBlock(node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_EQUIVALENCE = nn.ModuleList([
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])

    def d0(self, x):

        size_Z = x.size(1)
        index_x = int(size_Z/2)

        x1 = x[:, :index_x]
        x2 = x[:, index_x:]
        x1 = self.min_max_scaler(x1)
        x2 = self.min_max_scaler(x2)

        res = torch.max(torch.zeros_like(x1), x1 + x2 - 1)
        return res

    def d1(self, x):

        index_x = int(x.size(1) / 2)

        x1 = x[:, :index_x]
        x2 = x[:, index_x:]
        x1 = self.min_max_scaler(x1)
        x2 = self.min_max_scaler(x2)

        res = torch.min(torch.ones_like(x1), x1 + x2)
        return res

    def d2(self, x):

        return torch.min(torch.zeros_like(x), torch.ones_like(x) - x.sum(dim=1, keepdim=True)) \
               + torch.min(torch.ones_like(x), x.size(0) - x.sum(dim=1, keepdim=True))

    def d3(self, x, x_len):

        x_tuple = x.split(x_len, dim=1)
        x_list = []
        for x_1 in x_tuple:
            x_list.append(self.min_max_scaler(x_1))
        x_max = x_list[0]
        for x_item in x_list:
            x_max = torch.maximum(x_max, x_item)
        return x_max

    def d4(self, x, x_len):

        x_tuple = x.split(x_len, dim=1)
        x_list = []
        for x_1 in x_tuple:
            x_list.append(self.min_max_scaler(x_1))
        x_min = x_list[0]
        for x_item in x_list:
            x_min = torch.minimum(x_min, x_item)
        return x_min

    def min_max_scaler(self, x):
        min_values = torch.min(x, dim=1, keepdim=True).values
        max_values = torch.max(x, dim=1, keepdim=True).values
        res = (x - min_values) / (max_values - min_values)
        return res

    def get_symbol_child(self, x, node_attr, edge_index, symbol_index, max_children=2):
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None

        edge_mask = is_symbol[edge_index[0]]
        parents = edge_index[0, edge_mask]
        children = edge_index[1, edge_mask]
        child_feats = x[children]

        _, inv = parents.unique(sorted=True, return_inverse=True)
        n_sym = index_symbol.size(0)
        d = x.size(1)

        sort_idx = inv.argsort()
        sorted_feats = child_feats[sort_idx]

        if sorted_feats.size(0) == max_children * n_sym:
            return sorted_feats.reshape(n_sym, max_children * d), index_symbol

        counts = torch.bincount(inv, minlength=n_sym)
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum[:-1]])
        result = torch.zeros(n_sym, max_children * d, device=x.device)
        result[:, :d] = sorted_feats[starts]
        for c in range(1, max_children):
            has_c = counts > c
            idx = (starts + c).clamp(max=sorted_feats.size(0) - 1)
            result[has_c, c*d:(c+1)*d] = sorted_feats[idx[has_c]]
        return result, index_symbol

    def get_symbol_child2(self, x, node_attr, edge_index, symbol_index):
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None

        edge_mask = is_symbol[edge_index[0]]
        parents = edge_index[0, edge_mask]
        children = edge_index[1, edge_mask]
        child_feats = x[children]

        _, inv = parents.unique(sorted=True, return_inverse=True)
        n_sym = index_symbol.size(0)
        d = x.size(1)
        counts = torch.bincount(inv, minlength=n_sym)
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum[:-1]])

        child_norms = child_feats.norm(p=2, dim=1)

        max_norm = child_norms.max().item() + 1.0
        sort_key = inv.float() * max_norm - child_norms
        sort_idx = sort_key.argsort()
        sorted_feats = child_feats[sort_idx]
        sorted_inv = inv[sort_idx]

        counts2 = torch.bincount(sorted_inv, minlength=n_sym)
        cumsum2 = counts2.cumsum(0)
        starts2 = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum2[:-1]])

        result = torch.zeros(n_sym, 2 * d, device=x.device)
        first_idx = starts2
        result[:, :d] = sorted_feats[first_idx]
        has_second = counts2 >= 2
        second_idx = (starts2 + 1).clamp(max=sorted_feats.size(0) - 1)
        result[has_second, d:] = sorted_feats[second_idx[has_second]]
        return result, index_symbol

    def forward(self, x, term_walk_index, edge_index, node_attr, batch):
        N = x.size()[0]

        and_index, or_index, not_index, exists_index, forall_index = 4, 10, 3, 1, 9
        contains_index, equivalence_index = 11, 6

        for i in range(self.K):
            term_walk_feat = torch.cat([x[term_walk_index[0]],
                                        x[term_walk_index[1]],
                                        x[term_walk_index[2]]], dim=1)

            trans_T = self.F_T[i](term_walk_feat)
            m_T = scatter_mean(trans_T,
                               index=term_walk_index[0],
                               dim=0, dim_size=N)

            trans_M = self.F_M[i](term_walk_feat)
            m_M = scatter_mean(trans_M,
                               index=term_walk_index[1],
                               dim=0, dim_size=N)

            trans_B = self.F_B[i](term_walk_feat)
            m_B = scatter_mean(trans_B,
                               index=term_walk_index[2],
                               dim=0, dim_size=N)

            m_TW = m_T + m_M + m_B
            m_TW = self.F_trans_TW[i](m_TW)
            x = x + m_TW

            if self.use_lce_and:
                x_and, and_x_index = self.get_symbol_child(x, node_attr, edge_index, and_index)
                if x_and is not None:
                    x[and_x_index] -= self.F_AND[i](x_and) * self.lce_scale
            if self.use_lce_or:
                x_or, or_x_index = self.get_symbol_child(x, node_attr, edge_index, or_index)
                if x_or is not None:
                    x[or_x_index] -= self.F_OR[i](x_or) * self.lce_scale
            if self.use_lce_not:
                x_not, not_x_index = self.get_symbol_child(x, node_attr, edge_index, not_index, max_children=1)
                if x_not is not None:
                    x[not_x_index] += self.F_NOT[i](x_not) * self.lce_scale
            if self.use_lce_contains:
                x_contains, contains_x_index = self.get_symbol_child(x, node_attr, edge_index, contains_index)
                if x_contains is not None:
                    x[contains_x_index] += self.F_CONTAINS[i](x_contains) * self.lce_scale
            if self.use_lce_equivalence:
                x_equivalence, equivalence_x_index = self.get_symbol_child(x, node_attr, edge_index, equivalence_index)
                if x_equivalence is not None:
                    x[equivalence_x_index] += self.F_EQUIVALENCE[i](x_equivalence) * self.lce_scale
            if self.use_lce_forall:
                x_forall, forall_x_index = self.get_symbol_child2(x, node_attr, edge_index, forall_index)
                if x_forall is not None:
                    x[forall_x_index] += self.F_FORALL[i](x_forall) * self.lce_scale
            if self.use_lce_exists:
                x_exists, exists_x_index = self.get_symbol_child2(x, node_attr, edge_index, exists_index)
                if x_exists is not None:
                    x[exists_x_index] += self.F_EXISTS[i](x_exists) * self.lce_scale

        return x

class DAGPooling(nn.Module):
    def __init__(self):
        super(DAGPooling, self).__init__()

    def forward(self, x, batch_index):
        output = global_mean_pool(x, batch_index)
        return output

class Classifier(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(2 * node_out_channels, node_out_channels, batch=True),
            MLPBlock(node_out_channels, 2, activation=nn.Softmax(1), batch=True),
        )

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y

class Classifier2(nn.Module):
    def __init__(self, node_out_channels, rho_w=1.0):
        super(Classifier2, self).__init__()
        self.first = MLPBlock(2 * node_out_channels, node_out_channels,
                              batch=True)
        self.second = MLPBlock2(node_out_channels, 2, activation=nn.Softmax(1),
                                batch=True, constrain=True, rho_w=rho_w)

    def forward(self, conj_batch, prem_batch, similarity):

        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.first(x_concat)
        pred_y = self.second(pred_y, similarity)
        return pred_y

class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers,
                 use_lce_and=True, use_lce_or=True, use_lce_not=True,
                 use_lce_contains=True, use_lce_equivalence=True,
                 use_lce_forall=True, use_lce_exists=True,
                 use_classifier_constraint=True,
                 lce_scale=0.5, rho_w=1.0):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers,
                                    use_lce_and=use_lce_and, use_lce_or=use_lce_or,
                                    use_lce_not=use_lce_not, use_lce_contains=use_lce_contains,
                                    use_lce_equivalence=use_lce_equivalence,
                                    use_lce_forall=use_lce_forall, use_lce_exists=use_lce_exists,
                                    lce_scale=lce_scale)
        self.pooling = DAGPooling()
        self.use_classifier_constraint = use_classifier_constraint
        if use_classifier_constraint:
            self.classifier = Classifier2(node_out_channels, rho_w=rho_w)
        else:
            self.classifier = Classifier(node_out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None

    def forward(self, batch):
        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)
        h_s = self.dag_emb(h_s, batch.term_walk_index_s, batch.edge_index_s, batch.node_attr_s, batch.x_s_batch)
        h_t = self.dag_emb(h_t, batch.term_walk_index_t, batch.edge_index_t, batch.node_attr_t, batch.x_t_batch)
        h_g_s = self.pooling(h_s, batch.x_s_batch)
        h_g_t = self.pooling(h_t, batch.x_t_batch)
        if self.use_classifier_constraint:
            pred_y = self.classifier(h_g_s, h_g_t, batch.similarity)
        else:
            pred_y = self.classifier(h_g_s, h_g_t)

        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()

        one = torch.ones(batch.y.size()[0], device=batch.y.device)
        zero = torch.zeros(batch.y.size()[0], device=batch.y.device)
        self.tp = ((pred_label == one) & (batch.y == one)).sum().cpu().item()
        self.fp = ((pred_label == one) & (batch.y == zero)).sum().cpu().item()
        self.tn = ((pred_label == zero) & (batch.y == zero)).sum().cpu().item()
        self.fn = ((pred_label == zero) & (batch.y == one)).sum().cpu().item()

        loss = self.criterion(pred_y, batch.y)
        return loss
