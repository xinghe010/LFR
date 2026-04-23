import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch.nn.functional import one_hot
import json
import re

from formula_parser import fof_formula_transformer
from graph import Graph
from utils import read_file, load_pickle_file, Statements

from document_process import get_simi_from_json, get_simi_from_json2, get_simi_sw_from_json
VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
CONNECTIVE_PATTERN = {"!", "?", "|", "&", "=>", "<=>", "~"}
BOOL = "$true"

class PairData(Data):
    def __init__(self, x_s=None, term_walk_index_s=None, edge_index_s=None, node_attr_s=None, edge_attr_s=None,
                 cross_index_s=None, x_t=None, term_walk_index_t=None, edge_index_t=None, node_attr_t=None,
                 edge_attr_t=None, cross_index_t=None, y=None, similarity=None):
        super().__init__()
        self.x_s = x_s
        self.x_t = x_t
        self.term_walk_index_s = term_walk_index_s
        self.term_walk_index_t = term_walk_index_t

        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.node_attr_s = node_attr_s
        self.node_attr_t = node_attr_t
        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t
        self.cross_index_s = cross_index_s
        self.cross_index_t = cross_index_t
        self.y = y

        self.similarity = similarity

    def __inc__(self, key, value, *args, **kwargs):
        if key == "term_walk_index_s":
            return self.x_s.size(0)
        if key == "term_walk_index_t":
            return self.x_t.size(0)
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        if key == "cross_index_s":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        if key == "cross_index_t":
            return torch.tensor([[self.x_t.size(0)], [self.x_s.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class FormulaGraphDataset(InMemoryDataset):

    def __init__(self,
                 root,
                 data_class,
                 statements_file,
                 node_dict_file,
                 node_attr_dict_file,
                 edge_attr_dict_file,
                 rename=True,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.node_dict = load_pickle_file(node_dict_file)

        self.node_attr_dict = load_pickle_file(node_attr_dict_file)
        self.edge_attr_dict = load_pickle_file(edge_attr_dict_file)
        super(FormulaGraphDataset, self).__init__(root, transform,
                                                  pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_class)]

    def graph_process(self, G):
        nodes = []
        term_walk_indices = []
        edge_indices = []
        edge_attrs = []
        edge_attrs_reverse = []
        node_attrs = []

        for node in G:
            nodes.append(node.name)
            if node.name in {"!", "?"}:
                node_attrs.append(node.name)
                for child in node.children:
                    edge_indices.append([node.id, child.id])
                    if re.match(VARIABLE_PATTERN, child.name):
                        edge_attrs.append("{}_{}".format(node.name, "var"))
                        edge_attrs_reverse.append("{}_{}".format(
                            "var", node.name))
                    elif re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append("{}_{}".format(node.name, "pred"))
                        edge_attrs_reverse.append("{}_{}".format(
                            "pred", node.name))
                    elif child.name == "=":
                        edge_attrs.append("{}_{}".format(node.name, "="))
                        edge_attrs_reverse.append("{}_{}".format(
                            "=", node.name))
                    elif child.name in CONNECTIVE_PATTERN:
                        edge_attrs.append("{}_{}".format(
                            node.name, child.name))
                        edge_attrs_reverse.append("{}_{}".format(
                            child.name, node.name))
                    else:
                        raise AssertionError("Extra Attribution")
            if node.name in {"&", "|", "<=>", "~"}:
                node_attrs.append(node.name)
                for child in node.children:
                    edge_indices.append([node.id, child.id])
                    if re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append("{}_{}".format(node.name, "pred"))
                        edge_attrs_reverse.append("{}_{}".format(
                            "pred", node.name))
                    elif child.name == "=":
                        edge_attrs.append("{}_{}".format(node.name, "="))
                        edge_attrs_reverse.append("{}_{}".format(
                            "=", node.name))
                    elif child.name in CONNECTIVE_PATTERN:
                        edge_attrs.append("{}_{}".format(
                            node.name, child.name))
                        edge_attrs_reverse.append("{}_{}".format(
                            child.name, node.name))
                    else:
                        raise AssertionError("Extra Attribution")
            if node.name == "=>":
                node_attrs.append(node.name)
                for index, child in enumerate(node.children, 1):
                    edge_indices.append([node.id, child.id])
                    if re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append("{}_{}_{}".format(
                            node.name, "pred", index))
                        edge_attrs_reverse.append("{}_{}_{}".format(
                            "pred", node.name, index))
                    elif child.name == "=":
                        edge_attrs.append("{}_{}_{}".format(
                            node.name, "=", index))
                        edge_attrs_reverse.append("{}_{}_{}".format(
                            "=", node.name, index))
                    elif child.name in CONNECTIVE_PATTERN:
                        edge_attrs.append("{}_{}_{}".format(
                            node.name, child.name, index))
                        edge_attrs_reverse.append("{}_{}_{}".format(
                            child.name, node.name, index))
                    else:
                        raise AssertionError("Extra Attribution")
            if node.name == "=":
                node_attrs.append(node.name)
                for child in node.children:
                    edge_indices.append([node.id, child.id])
                    if re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append("{}_{}".format(node.name, "func"))
                        edge_attrs_reverse.append("{}_{}".format(
                            "func", node.name))
                    elif re.match(VARIABLE_PATTERN, child.name):
                        edge_attrs.append("{}_{}".format(node.name, "var"))
                        edge_attrs_reverse.append("{}_{}".format(
                            "var", node.name))
                    else:
                        raise AssertionError("Extra Attribution")
            if re.match(FUNCTOR_PATTERN, node.name):
                parents_tokens = [parent.name for parent in node.parents]
                if len(parents_tokens
                       ) == 0 or parents_tokens[0] in CONNECTIVE_PATTERN:
                    node_attrs.append("pred")
                    for index, child in enumerate(node.children, 1):
                        edge_indices.append([node.id, child.id])
                        if re.match(FUNCTOR_PATTERN, child.name):
                            edge_attrs.append("{}_{}_{}".format(
                                "pred", "func", index))
                            edge_attrs_reverse.append("{}_{}_{}".format(
                                "func", "pred", index))
                        elif re.match(VARIABLE_PATTERN, child.name):
                            edge_attrs.append("{}_{}_{}".format(
                                "pred", "var", index))
                            edge_attrs_reverse.append("{}_{}_{}".format(
                                "var", "pred", index))
                        else:
                            raise AssertionError("Extra Attribution")
                else:
                    node_attrs.append("func")
                    for index, child in enumerate(node.children, 1):
                        edge_indices.append([node.id, child.id])
                        if re.match(FUNCTOR_PATTERN, child.name):
                            edge_attrs.append("{}_{}_{}".format(
                                "func", "func", index))
                            edge_attrs_reverse.append("{}_{}_{}".format(
                                "func", "func", index))
                        elif re.match(VARIABLE_PATTERN, child.name):
                            edge_attrs.append("{}_{}_{}".format(
                                "func", "var", index))
                            edge_attrs_reverse.append("{}_{}_{}".format(
                                "var", "func", index))
                        else:
                            raise AssertionError("Extra Attribution")
            if node.name == "$true":
                node_attrs.append(node.name)
            if re.match(VARIABLE_PATTERN, node.name):
                node_attrs.append("var")
        edge_indices = np.array(edge_indices, dtype=np.int64).reshape(-1, 2).T
        edge_attrs = edge_attrs + edge_attrs_reverse

        for node in G:

            if node.parents and node.children:
                for parent in node.parents:
                    for child in node.children:
                        term_walk_indices.append([parent.id,
                                                  node.id,
                                                  child.id])

        term_walk_indices = np.array(
            term_walk_indices, dtype=np.int64).reshape(-1, 3).T

        assert len(nodes) == len(node_attrs)
        assert (2 * edge_indices.shape[1]) == len(edge_attrs)

        return nodes, term_walk_indices, edge_indices, node_attrs, edge_attrs

    def build_cross_index(self, c_nodes, p_nodes):
        cross_indices = []
        for i, c_name in enumerate(c_nodes):
            if re.match(FUNCTOR_PATTERN, c_name) or c_name == "=":
                for j, p_name in enumerate(p_nodes):
                    if c_name == p_name:
                        cross_indices.extend([i, j])
        cross_indices = \
            np.array(cross_indices, dtype=np.int64).reshape(-1, 2).T
        return cross_indices

    def vectorization(self, objects, object_dict):
        indices = [object_dict[obj] for obj in objects]
        onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
        return onehot

    def process(self):
        raw_examples = \
            [json.loads(line) for line in read_file(self.raw_paths[0])]
        dataList = []

        import os
        root_parent = os.path.dirname(self.root)
        simi2_candidates = [
            os.path.join(root_parent, "similarity2.json"),
            os.path.join(self.root, "..", "similarity2.json"),
            "dataset/similarity2.json",
            "similarity2.json",
        ]
        simi4_candidates = [
            os.path.join(root_parent, "similarity4.json"),
            os.path.join(self.root, "..", "similarity4.json"),
            "dataset/similarity4.json",
            "similarity4.json",
        ]
        simi2_path = next((p for p in simi2_candidates if os.path.exists(p)), None)
        simi4_path = next((p for p in simi4_candidates if os.path.exists(p)), None)
        simi_dict = get_simi_from_json2(simi2_path) if simi2_path else {}
        simi_sw_dict = get_simi_sw_from_json(simi4_path) if simi4_path else {}

        for example in raw_examples:
            conj, prem, label = example
            conj_graph = Graph(fof_formula_transformer(self.statements[conj]),
                               rename=self.rename)
            prem_graph = Graph(fof_formula_transformer(self.statements[prem]),
                               rename=self.rename)

            c_nodes, c_term_walk_indices, c_edge_indices, c_node_attrs, c_edge_attrs = \
                self.graph_process(conj_graph)
            p_nodes, p_term_walk_indices, p_edge_indices, p_node_attrs, p_edge_attrs = \
                self.graph_process(prem_graph)
            cross_indices = self.build_cross_index(c_nodes, p_nodes)

            similarity1 = simi_dict.get(conj, {}).get(prem, 0)
            similarity2 = simi_sw_dict.get(conj, {}).get(prem, 0)

            data = PairData(
                x_s=self.vectorization(c_nodes, self.node_dict),
                term_walk_index_s=torch.from_numpy(c_term_walk_indices),
                edge_index_s=torch.from_numpy(c_edge_indices),
                node_attr_s=self.vectorization(c_node_attrs,
                                               self.node_attr_dict),
                edge_attr_s=self.vectorization(c_edge_attrs,
                                               self.edge_attr_dict),
                cross_index_s=torch.from_numpy(cross_indices),
                x_t=self.vectorization(p_nodes, self.node_dict),
                term_walk_index_t=torch.from_numpy(p_term_walk_indices),
                edge_index_t=torch.from_numpy(p_edge_indices),
                node_attr_t=self.vectorization(p_node_attrs,
                                               self.node_attr_dict),
                edge_attr_t=self.vectorization(p_edge_attrs,
                                               self.edge_attr_dict),
                cross_index_t=torch.from_numpy(cross_indices)[[1, 0]],
                y=torch.LongTensor([label]),
                similarity=torch.LongTensor([similarity1, similarity2]))
            dataList.append(data)
        data, slices = self.collate(data_list=dataList)
        torch.save((data, slices), self.processed_paths[0])
