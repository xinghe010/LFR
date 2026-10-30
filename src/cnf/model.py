import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool

from within_layer import *

import numpy as np

NODE_ATTR_CHANNELS = 13
EDGE_ATTR_CHANNELS = 169
DEFAULT_OPERATOR_TYPE_INDICES = {
    "exists": 1, "not": 3, "and": 4, "equivalence": 6,
    "forall": 9, "or": 10, "implication": 11,
}
SYMBOLIC_FEATURE_INDICES = {
    "overlap": 0, "idf": 1, "sine": 2, "trigger": 3,
    "weighted_similarity": 4, "weighted_similarity_score": 5,
}


class StableLayerNorm(nn.LayerNorm):
    """Per-example normalization with legacy BatchNorm state keys."""
    def __init__(self, features):
        super().__init__(features)
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))


class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False,
                 normalize=True):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.normalize = normalize
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = StableLayerNorm(out_channels)
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
        if self.batch and self.normalize:
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
                 rho_w=1.0, rho_s=None,
                 use_trigger_rule=True, use_similarity_rule=True,
                 normalize=True):
        super(MLPBlock2, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.normalize = normalize
        self.drop = drop
        self.constrain = constrain
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = StableLayerNorm(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.8)
        self.reset_parameters()
        
        if self.constrain:
            self.similarity = WithinLayer(
                0, rho_w, rho_s=rho_s,
                use_trigger_rule=use_trigger_rule,
                use_similarity_rule=use_similarity_rule)

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
        if self.batch and self.normalize:
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
        
        
        indices = x.long().reshape(-1) if x.dim() == 1 else torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output


class OperatorEmbeddingAugmentation(nn.Module):
    """Learn an explicit embedding for logical-operator node types."""
    def __init__(self, node_attr_channels, channels, operator_type_indices):
        super().__init__()
        self.embedding = nn.Embedding(node_attr_channels, channels)
        self.register_buffer("operator_mask", torch.tensor(
            [i in set(operator_type_indices.values()) for i in range(node_attr_channels)],
            dtype=torch.bool))
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, node_attr):
        type_ids = node_attr.argmax(dim=1)
        return self.embedding(type_ids) * self.operator_mask[type_ids].unsqueeze(1)


class RelationalMessagePassing(nn.Module):
    """Basis-decomposed relation-specific messages over typed directed edges."""
    def __init__(self, channels, relation_count, bases=4):
        super().__init__()
        self.bases = nn.Parameter(torch.empty(bases, channels, channels))
        self.coefficients = nn.Embedding(relation_count, bases)
        self.self_linear = nn.Linear(channels, channels, bias=False)
        self.norm = StableLayerNorm(channels)
        nn.init.xavier_normal_(self.bases)
        nn.init.xavier_normal_(self.coefficients.weight)
        nn.init.xavier_normal_(self.self_linear.weight)

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.numel() == 0:
            return F.relu(self.norm(self.self_linear(x)))
        edge_graph = batch[edge_index[0]]
        directed_edges = torch.cat([
            torch.cat([edge_index[:, edge_graph == graph_id],
                       edge_index[:, edge_graph == graph_id].flip(0)], dim=1)
            for graph_id in torch.unique(edge_graph, sorted=True)
        ], dim=1)
        if edge_attr.size(0) != directed_edges.size(1):
            raise ValueError("edge_attr must contain forward and reverse relation rows")
        relation_ids = edge_attr.argmax(dim=1)
        source, target = directed_edges
        transformed = torch.einsum("ed,bdh->ebh", x[source], self.bases)
        messages = torch.einsum("ebh,eb->eh", transformed,
                                self.coefficients(relation_ids))
        aggregated = scatter_mean(messages, target, dim=0, dim_size=x.size(0))
        return F.relu(self.norm(aggregated + self.self_linear(x)))


class ParameterMatchedGenericMLP(nn.Module):
    """High-capacity shared MLP that never observes node/operator type."""
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.first = MLPBlock(channels, hidden_channels, batch=True)
        self.second = MLPBlock(hidden_channels, channels, batch=True)

    def forward(self, x):
        return self.second(self.first(x))


class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers,
                 use_lce_and=True, use_lce_or=True, use_lce_not=True,
                 use_lce_contains=True, use_lce_equivalence=True,
                 use_lce_forall=True, use_lce_exists=True,
                 lce_scale=0.5, encoder_variant="lce",
                 node_attr_channels=NODE_ATTR_CHANNELS,
                 edge_attr_channels=EDGE_ATTR_CHANNELS,
                 relation_bases=4, generic_mlp_width=None,
                 use_term_walk=True, quantifier_aggregation="top2",
                 commutative_aggregation="ordered", operator_type_indices=None,
                 variable_type_index=2):
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
        self.encoder_variant = encoder_variant
        self.use_term_walk = use_term_walk
        self.quantifier_aggregation = quantifier_aggregation
        self.commutative_aggregation = commutative_aggregation
        self.operator_type_indices = dict(
            DEFAULT_OPERATOR_TYPE_INDICES if operator_type_indices is None
            else operator_type_indices)
        required_operator_names = set(DEFAULT_OPERATOR_TYPE_INDICES)
        if set(self.operator_type_indices) != required_operator_names:
            raise ValueError(
                "operator_type_indices must define exactly "
                f"{sorted(required_operator_names)}")
        if any(index < 0 or index >= node_attr_channels
               for index in self.operator_type_indices.values()):
            raise ValueError("operator_type_indices contains an out-of-range index")
        if variable_type_index < 0 or variable_type_index >= node_attr_channels:
            raise ValueError("variable_type_index is out of range")
        self.variable_type_index = variable_type_index
        if quantifier_aggregation not in {"top2", "all_child_mean", "binder_body"}:
            raise ValueError(f"Unknown quantifier aggregation: {quantifier_aggregation}")
        if commutative_aggregation not in {"ordered", "mean", "sum"}:
            raise ValueError(f"Unknown commutative aggregation: {commutative_aggregation}")
        if encoder_variant not in {"lce", "tw", "operator_embedding",
                                   "relation_messages", "parameter_matched_mlp"}:
            raise ValueError(f"Unknown encoder_variant: {encoder_variant}")
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
        if encoder_variant == "lce":
            commutative_input = (2 * node_out_channels
                                 if commutative_aggregation == "ordered"
                                 else node_out_channels)
            self.F_AND = nn.ModuleList([MLPBlock(commutative_input, node_out_channels, batch=True) for _ in range(layers)])
            self.F_OR = nn.ModuleList([MLPBlock(commutative_input, node_out_channels, batch=True) for _ in range(layers)])
            self.F_FORALL = nn.ModuleList([MLPBlock(2 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
            self.F_EXISTS = nn.ModuleList([MLPBlock(2 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
            self.F_CONTAINS = nn.ModuleList([MLPBlock(2 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
            self.F_NOT = nn.ModuleList([MLPBlock(node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
            self.F_EQUIVALENCE = nn.ModuleList([MLPBlock(2 * node_out_channels, node_out_channels, batch=True) for _ in range(layers)])
        if encoder_variant == "operator_embedding":
            self.operator_embeddings = nn.ModuleList([
                OperatorEmbeddingAugmentation(
                    node_attr_channels, node_out_channels, self.operator_type_indices)
                for _ in range(layers)
            ])
        if encoder_variant == "relation_messages":
            self.relation_layers = nn.ModuleList([
                RelationalMessagePassing(node_out_channels, edge_attr_channels,
                                         bases=relation_bases)
                for _ in range(layers)
            ])
        if encoder_variant == "parameter_matched_mlp":
            width = generic_mlp_width or round(6.5 * node_out_channels)
            self.generic_layers = nn.ModuleList([
                ParameterMatchedGenericMLP(node_out_channels, width)
                for _ in range(layers)
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
        """Vectorized: find children features for all nodes of a given symbol type.
        max_children=1 for unary (NOT), max_children=2 for binary (AND, OR, etc.)"""
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None

        edge_mask = is_symbol[edge_index[0]]
        parents = edge_index[0, edge_mask]
        children = edge_index[1, edge_mask]
        if children.numel() == 0:
            return None, None
        child_feats = x[children]

        _, inv = parents.unique(sorted=True, return_inverse=True)
        n_sym = index_symbol.size(0)
        d = x.size(1)

        sort_idx = torch.argsort(inv, stable=True)
        sorted_feats = child_feats[sort_idx]

        
        if sorted_feats.size(0) == max_children * n_sym:
            return sorted_feats.reshape(n_sym, max_children * d), index_symbol

        
        counts = torch.bincount(inv, minlength=n_sym)
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum[:-1]])
        result = torch.zeros(n_sym, max_children * d, device=x.device,
                             dtype=x.dtype)
        result[:, :d] = sorted_feats[starts]
        for c in range(1, max_children):
            has_c = counts > c
            idx = (starts + c).clamp(max=sorted_feats.size(0) - 1)
            result[has_c, c*d:(c+1)*d] = sorted_feats[idx[has_c]]
        return result, index_symbol

    def get_symbol_child2(self, x, node_attr, edge_index, symbol_index):
        """Vectorized: find top-2 children (by norm) for quantifier nodes."""
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None

        edge_mask = is_symbol[edge_index[0]]
        parents = edge_index[0, edge_mask]
        children = edge_index[1, edge_mask]
        if children.numel() == 0:
            return None, None
        child_feats = x[children]

        _, inv = parents.unique(sorted=True, return_inverse=True)
        n_sym = index_symbol.size(0)
        d = x.size(1)
        counts = torch.bincount(inv, minlength=n_sym)
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum[:-1]])

        
        child_norms = child_feats.norm(p=2, dim=1)
        
        norm_order = torch.argsort(child_norms, descending=True, stable=True)
        parent_order = torch.argsort(inv[norm_order], stable=True)
        sort_idx = norm_order[parent_order]
        sorted_feats = child_feats[sort_idx]
        sorted_inv = inv[sort_idx]

        
        counts2 = torch.bincount(sorted_inv, minlength=n_sym)
        cumsum2 = counts2.cumsum(0)
        starts2 = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), cumsum2[:-1]])

        
        result = torch.zeros(n_sym, 2 * d, device=x.device, dtype=x.dtype)
        first_idx = starts2
        result[:, :d] = sorted_feats[first_idx]
        has_second = counts2 >= 2
        second_idx = (starts2 + 1).clamp(max=sorted_feats.size(0) - 1)
        result[has_second, d:] = sorted_feats[second_idx[has_second]]
        return result, index_symbol

    def get_quantifier_children(self, x, node_attr, edge_index, symbol_index):
        if self.quantifier_aggregation == "top2":
            return self.get_symbol_child2(x, node_attr, edge_index, symbol_index)
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None
        mask = is_symbol[edge_index[0]]
        parents, children = edge_index[0, mask], edge_index[1, mask]
        if children.numel() == 0:
            return None, None
        _, groups = parents.unique(sorted=True, return_inverse=True)
        dimension = x.size(1)
        if self.quantifier_aggregation == "all_child_mean":
            pooled = scatter_mean(x[children], groups, dim=0,
                                  dim_size=index_symbol.numel())
            return torch.cat([pooled, pooled], dim=1), index_symbol
        is_variable = node_attr[children, self.variable_type_index].bool()
        binder = scatter_mean(x[children[is_variable]], groups[is_variable],
                              dim=0, dim_size=index_symbol.numel())
        body = scatter_mean(x[children[~is_variable]], groups[~is_variable],
                            dim=0, dim_size=index_symbol.numel())
        return torch.cat([binder.reshape(-1, dimension),
                          body.reshape(-1, dimension)], dim=1), index_symbol

    def get_commutative_children(self, x, node_attr, edge_index, symbol_index):
        if self.commutative_aggregation == "ordered":
            return self.get_symbol_child(x, node_attr, edge_index, symbol_index)
        is_symbol = node_attr[:, symbol_index].bool()
        index_symbol = is_symbol.nonzero(as_tuple=True)[0]
        if index_symbol.numel() == 0:
            return None, None
        mask = is_symbol[edge_index[0]]
        parents, children = edge_index[0, mask], edge_index[1, mask]
        if children.numel() == 0:
            return None, None
        _, groups = parents.unique(sorted=True, return_inverse=True)
        if self.commutative_aggregation == "mean":
            pooled = scatter_mean(x[children], groups, dim=0,
                                  dim_size=index_symbol.numel())
        else:
            pooled = torch.zeros(index_symbol.numel(), x.size(1),
                                 device=x.device, dtype=x.dtype)
            pooled.index_add_(0, groups, x[children])
        return pooled, index_symbol

    def forward(self, x, term_walk_index, edge_index, edge_attr, node_attr, batch):
        N = x.size()[0]
        
        and_index = self.operator_type_indices["and"]
        or_index = self.operator_type_indices["or"]
        not_index = self.operator_type_indices["not"]
        exists_index = self.operator_type_indices["exists"]
        forall_index = self.operator_type_indices["forall"]
        contains_index = self.operator_type_indices["implication"]
        equivalence_index = self.operator_type_indices["equivalence"]

        
        for i in range(self.K):
            if self.use_term_walk:
                term_walk_feat = torch.cat([x[term_walk_index[0]],
                                            x[term_walk_index[1]],
                                            x[term_walk_index[2]]], dim=1)
                trans_T = self.F_T[i](term_walk_feat)
                m_T = scatter_mean(trans_T, index=term_walk_index[0], dim=0, dim_size=N)
                trans_M = self.F_M[i](term_walk_feat)
                m_M = scatter_mean(trans_M, index=term_walk_index[1], dim=0, dim_size=N)
                trans_B = self.F_B[i](term_walk_feat)
                m_B = scatter_mean(trans_B, index=term_walk_index[2], dim=0, dim_size=N)
                x = x + self.F_trans_TW[i](m_T + m_M + m_B)
            if self.encoder_variant == "operator_embedding":
                x = x + self.operator_embeddings[i](node_attr) * self.lce_scale
            elif self.encoder_variant == "relation_messages":
                x = x + self.relation_layers[i](x, edge_index, edge_attr, batch) * self.lce_scale
            elif self.encoder_variant == "parameter_matched_mlp":
                x = x + self.generic_layers[i](x) * self.lce_scale
            
            if self.encoder_variant == "lce" and self.use_lce_and:
                x_and, and_x_index = self.get_commutative_children(x, node_attr, edge_index, and_index)
                if x_and is not None:
                    x[and_x_index] -= self.F_AND[i](x_and) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_or:
                x_or, or_x_index = self.get_commutative_children(x, node_attr, edge_index, or_index)
                if x_or is not None:
                    x[or_x_index] -= self.F_OR[i](x_or) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_not:
                x_not, not_x_index = self.get_symbol_child(x, node_attr, edge_index, not_index, max_children=1)
                if x_not is not None:
                    x[not_x_index] += self.F_NOT[i](x_not) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_contains:
                x_contains, contains_x_index = self.get_symbol_child(x, node_attr, edge_index, contains_index)
                if x_contains is not None:
                    x[contains_x_index] += self.F_CONTAINS[i](x_contains) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_equivalence:
                x_equivalence, equivalence_x_index = self.get_symbol_child(x, node_attr, edge_index, equivalence_index)
                if x_equivalence is not None:
                    x[equivalence_x_index] += self.F_EQUIVALENCE[i](x_equivalence) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_forall:
                x_forall, forall_x_index = self.get_quantifier_children(x, node_attr, edge_index, forall_index)
                if x_forall is not None:
                    x[forall_x_index] += self.F_FORALL[i](x_forall) * self.lce_scale
            if self.encoder_variant == "lce" and self.use_lce_exists:
                x_exists, exists_x_index = self.get_quantifier_children(x, node_attr, edge_index, exists_index)
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
            MLPBlock(node_out_channels, 2, activation=nn.Identity(),
                     batch=True, normalize=False),
        )

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y


class Classifier2(nn.Module):  
    def __init__(self, node_out_channels, rho_w=1.0, rho_t=None, rho_s=None,
                 use_trigger_rule=True, use_similarity_rule=True):
        super(Classifier2, self).__init__()
        self.first = MLPBlock(2 * node_out_channels, node_out_channels,
                              batch=True)
        trigger_weight = rho_w if rho_t is None else rho_t
        self.second = MLPBlock2(
            node_out_channels, 2, activation=nn.Identity(), batch=True,
            constrain=True, rho_w=trigger_weight, rho_s=rho_s,
            use_trigger_rule=use_trigger_rule,
            use_similarity_rule=use_similarity_rule, normalize=False)
        
        
        
        
        

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
                 lce_scale=0.5, rho_w=1.0, rho_t=None, rho_s=None,
                 use_trigger_rule=True, use_similarity_rule=True,
                 encoder_variant="lce", relation_bases=4,
                 generic_mlp_width=None, symbolic_baseline="lfr",
                 use_term_walk=True, quantifier_aggregation="top2",
                 commutative_aggregation="ordered", operator_type_indices=None,
                 variable_type_index=2):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers,
                                    use_lce_and=use_lce_and, use_lce_or=use_lce_or,
                                    use_lce_not=use_lce_not, use_lce_contains=use_lce_contains,
                                    use_lce_equivalence=use_lce_equivalence,
                                    use_lce_forall=use_lce_forall, use_lce_exists=use_lce_exists,
                                    lce_scale=lce_scale,
                                    encoder_variant=encoder_variant,
                                    relation_bases=relation_bases,
                                    generic_mlp_width=generic_mlp_width,
                                    use_term_walk=use_term_walk,
                                    quantifier_aggregation=quantifier_aggregation,
                                    commutative_aggregation=commutative_aggregation,
                                    operator_type_indices=operator_type_indices,
                                    variable_type_index=variable_type_index)
        self.pooling = DAGPooling()
        if symbolic_baseline != "lfr" and symbolic_baseline not in SYMBOLIC_FEATURE_INDICES:
            raise ValueError(f"Unknown symbolic_baseline: {symbolic_baseline}")
        self.symbolic_baseline = symbolic_baseline
        if symbolic_baseline != "lfr":
            use_trigger_rule, use_similarity_rule = False, True
        self.use_classifier_constraint = use_classifier_constraint
        if use_classifier_constraint:
            self.classifier = Classifier2(
                node_out_channels, rho_w=rho_w, rho_t=rho_t, rho_s=rho_s,
                use_trigger_rule=use_trigger_rule,
                use_similarity_rule=use_similarity_rule)
        else:
            self.classifier = Classifier(node_out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None

    def predict_logits(self, batch):
        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)
        h_s = self.dag_emb(h_s, batch.term_walk_index_s, batch.edge_index_s, batch.edge_attr_s, batch.node_attr_s, batch.x_s_batch)
        h_t = self.dag_emb(h_t, batch.term_walk_index_t, batch.edge_index_t, batch.edge_attr_t, batch.node_attr_t, batch.x_t_batch)
        h_g_s = self.pooling(h_s, batch.x_s_batch)
        h_g_t = self.pooling(h_t, batch.x_t_batch)
        if self.use_classifier_constraint:
            similarity = batch.similarity
            if self.symbolic_baseline != "lfr":
                if not hasattr(batch, "symbolic_features"):
                    raise ValueError("symbolic_baselines.jsonl was not included when the dataset cache was built")
                feature = batch.symbolic_features[:, SYMBOLIC_FEATURE_INDICES[self.symbolic_baseline]]
                similarity = torch.stack([torch.zeros_like(feature), feature], dim=1)
            pred_y = self.classifier(h_g_s, h_g_t, similarity)
        else:
            pred_y = self.classifier(h_g_s, h_g_t)
        return pred_y

    def predict_proba(self, batch):
        return torch.softmax(self.predict_logits(batch), dim=1)

    def forward(self, batch):
        logits = self.predict_logits(batch)
        pred_label = torch.argmax(logits, dim=1)
        self.corrects = (pred_label == batch.y).sum().cpu().item()
        one = torch.ones(batch.y.size()[0], device=batch.y.device)
        zero = torch.zeros(batch.y.size()[0], device=batch.y.device)
        self.tp = ((pred_label == one) & (batch.y == one)).sum().cpu().item()
        self.fp = ((pred_label == one) & (batch.y == zero)).sum().cpu().item()
        self.tn = ((pred_label == zero) & (batch.y == zero)).sum().cpu().item()
        self.fn = ((pred_label == zero) & (batch.y == one)).sum().cpu().item()
        return self.criterion(logits, batch.y)
