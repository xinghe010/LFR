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
                 constrain=False):
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
            self.similarity = WithinLayer(0, 1)

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
    def __init__(self, node_out_channels, layers):
        super(DAGEmbedding, self).__init__()
        self.K = layers
        self.F_T = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_M = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_B = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_trans_TW = nn.ModuleList([
            MLPBlock(node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_AND = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_OR = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_FORALL = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_EXISTS = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])

        self.F_CONTAINS = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])

        self.F_NOT = nn.ModuleList([
            MLPBlock(node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])

        self.F_EQUIVALENCE = nn.ModuleList([
            MLPBlock(2 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
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

    def get_symbol_child(self, x, node_attr, edge_index, symbol_index):
        temp = node_attr[:, symbol_index]
        index_symbol = torch.nonzero(temp == 1).squeeze()
        index_symbol = index_symbol.view(-1)
        if len(index_symbol) == 0:
            return None, None
        expected_dim = None
        list_symbol_child = []
        for i in index_symbol:
            temp = torch.nonzero(edge_index[0] == i).squeeze()
            index_symbol_child = edge_index[1][temp].view(-1)

            index_symbol_child = index_symbol_child[:2]

            x_symbol_child = x[index_symbol_child].view(-1)
            if expected_dim is None:
                expected_dim = x_symbol_child.size(0)
            if x_symbol_child.size(0) == expected_dim:
                x_symbol_child = x_symbol_child.unsqueeze(dim=0)
                list_symbol_child.append(x_symbol_child)
        if len(list_symbol_child) == 0:
            return None, None
        symbol_child = torch.cat(list_symbol_child, dim=0)
        index_symbol = index_symbol[:len(list_symbol_child)]

        return symbol_child, index_symbol

    def get_symbol_child2(self, x, node_attr, edge_index, symbol_index):
        temp = node_attr[:, symbol_index]
        index_symbol = torch.nonzero(temp == 1).squeeze()
        index_symbol = index_symbol.view(-1)
        if len(index_symbol) == 0:
            return None, None
        expected_dim = None
        list_symbol_child = []
        valid_indices = []
        for idx, i in enumerate(index_symbol):
            temp = torch.nonzero(edge_index[0] == i).squeeze()
            index_symbol_child = edge_index[1][temp].view(-1)
            if len(index_symbol_child) > 2:
                child_norm = torch.norm(x[index_symbol_child], p=2, dim=1)
                top2_index = torch.topk(child_norm, 2).indices
                index_symbol_child = index_symbol_child[top2_index]
            elif len(index_symbol_child) < 2:
                continue

            x_symbol_child = x[index_symbol_child].view(-1)
            if expected_dim is None:
                expected_dim = x_symbol_child.size(0)
            if x_symbol_child.size(0) == expected_dim:
                x_symbol_child = x_symbol_child.unsqueeze(dim=0)
                list_symbol_child.append(x_symbol_child)
                valid_indices.append(idx)
        if len(list_symbol_child) == 0:
            return None, None
        symbol_child = torch.cat(list_symbol_child, dim=0)
        index_symbol = index_symbol[valid_indices]

        return symbol_child, index_symbol

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

            def _safe_apply(func, x_sym, idx, sign=1.0):
                if x_sym is not None:
                    try:
                        x[idx] += sign * func(x_sym) * 0.5
                    except RuntimeError:
                        pass

            x_and, and_idx = self.get_symbol_child(x, node_attr, edge_index, and_index)
            _safe_apply(self.F_AND[i], x_and, and_idx, -1.0)
            x_or, or_idx = self.get_symbol_child(x, node_attr, edge_index, or_index)
            _safe_apply(self.F_OR[i], x_or, or_idx, -1.0)
            x_not, not_idx = self.get_symbol_child(x, node_attr, edge_index, not_index)
            _safe_apply(self.F_NOT[i], x_not, not_idx, 1.0)
            x_cont, cont_idx = self.get_symbol_child(x, node_attr, edge_index, contains_index)
            _safe_apply(self.F_CONTAINS[i], x_cont, cont_idx, 1.0)
            x_eq, eq_idx = self.get_symbol_child(x, node_attr, edge_index, equivalence_index)
            _safe_apply(self.F_EQUIVALENCE[i], x_eq, eq_idx, 1.0)

            x_forall, forall_idx = self.get_symbol_child2(x, node_attr, edge_index, forall_index)
            _safe_apply(self.F_FORALL[i], x_forall, forall_idx, 1.0)
            x_exists, exists_idx = self.get_symbol_child2(x, node_attr, edge_index, exists_index)
            _safe_apply(self.F_EXISTS[i], x_exists, exists_idx, 1.0)

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
            MLPBlock2(2 * node_out_channels, node_out_channels,
                      batch=True),
            MLPBlock2(node_out_channels, 2, activation=nn.Softmax(1),
                      batch=True, constrain=True))

    def forward(self, conj_batch, prem_batch):

        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y

class Classifier2(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier2, self).__init__()
        self.first = MLPBlock(2 * node_out_channels, node_out_channels,
                              batch=True)
        self.second = MLPBlock2(node_out_channels, 2, activation=nn.Softmax(1),
                                batch=True, constrain=True)

    def forward(self, conj_batch, prem_batch, similarity):

        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.first(x_concat)
        pred_y = self.second(pred_y, similarity)
        return pred_y

class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers)
        self.pooling = DAGPooling()
        self.classifier = Classifier2(node_out_channels)
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
        pred_y = self.classifier(h_g_s, h_g_t, batch.similarity)

        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()

        device = batch.y.device
        one = torch.ones(batch.y.size()[0]).to(device)
        zero = torch.zeros(batch.y.size()[0]).to(device)
        self.tp = ((pred_label == one) & (batch.y == one)).sum().cpu().item()
        self.fp = ((pred_label == one) & (batch.y == zero)).sum().cpu().item()
        self.tn = ((pred_label == zero) & (batch.y == zero)).sum().cpu().item()
        self.fn = ((pred_label == zero) & (batch.y == one)).sum().cpu().item()

        loss = self.criterion(pred_y, batch.y)
        return loss
