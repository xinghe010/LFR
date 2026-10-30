import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch.nn.functional import one_hot
import json
import re
import hashlib
import fcntl
from pathlib import Path

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
        self.symbolic_features = None

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
                 symbolic_features_file="symbolic_baselines.jsonl",
                 rename=True,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.symbolic_features_file = symbolic_features_file
        self.node_dict = load_pickle_file(node_dict_file)
        
        self.node_attr_dict = load_pickle_file(node_attr_dict_file)
        self.edge_attr_dict = load_pickle_file(edge_attr_dict_file)
        signature_paths = [Path(root) / "raw" / f"{data_class}.json",
                           Path(statements_file), Path(node_dict_file),
                           Path(node_attr_dict_file), Path(edge_attr_dict_file)]
        data_root = Path(root).parent
        
        
        
        rule_metadata = data_root / "rule_cache_metadata.json"
        if rule_metadata.is_file():
            signature_paths.append(rule_metadata)
        else:
            signature_paths += [data_root / "similarity2.json",
                                data_root / "similarity4.json"]
        symbolic_metadata = data_root / f"{symbolic_features_file}.metadata.json"
        signature_paths.append(symbolic_metadata if symbolic_metadata.is_file()
                               else data_root / symbolic_features_file)
        digest = hashlib.sha256()
        for path in signature_paths:
            if path.is_file():
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        self.cache_signature = digest.hexdigest()[:16]
        
        
        
        processed_dir = Path(root) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        lock_path = processed_dir / f".{data_class}_{self.cache_signature}.lock"
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                super(FormulaGraphDataset, self).__init__(root, transform,
                                                          pre_transform)
                self.data, self.slices = torch.load(
                    self.processed_paths[0], weights_only=False)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return [f"{self.data_class}_{self.cache_signature}.pt"]

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

    def node_ids(self, objects):
        unknown = self.node_dict.get("__UNK__")
        if unknown is None:
            return [self.node_dict[obj] for obj in objects]
        return [self.node_dict.get(obj, unknown) for obj in objects]

    def process(self):
        raw_examples = \
            [json.loads(line) for line in read_file(self.raw_paths[0])]
        dataList = []
        
        
        simi_dict = get_simi_from_json2(os.path.join(os.path.dirname(self.root), "similarity2.json"))      
        simi_sw_dict = get_simi_sw_from_json(os.path.join(os.path.dirname(self.root), "similarity4.json"))      
        symbolic_path = os.path.join(os.path.dirname(self.root), self.symbolic_features_file)
        symbolic_dict = {}
        if os.path.isfile(symbolic_path):
            required_pairs = {(item[0], item[1]) for item in raw_examples}
            with open(symbolic_path, encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    key = (item["conjecture"], item["premise"])
                    if key in required_pairs:
                        symbolic_dict[key] = [
                            item["overlap"], item["idf"], item["sine"], item["trigger"],
                            item["weighted_similarity"], item["weighted_similarity_score"]]
            missing = required_pairs.difference(symbolic_dict)
            if missing:
                raise KeyError(f"symbolic cache misses {len(missing)} {self.data_class} pairs")

        formula_cache = {}
        def formula_data(name):
            if name not in formula_cache:
                graph = Graph(fof_formula_transformer(self.statements[name]), rename=self.rename)
                nodes, term_walk, edge_index, node_attrs, edge_attrs = self.graph_process(graph)
                formula_cache[name] = (nodes, torch.LongTensor(self.node_ids(nodes)),
                    torch.from_numpy(term_walk), torch.from_numpy(edge_index),
                    self.vectorization(node_attrs, self.node_attr_dict),
                    self.vectorization(edge_attrs, self.edge_attr_dict))
            return formula_cache[name]

        for example in raw_examples:
            conj, prem, label = example
            c_nodes, c_x, c_term_walk, c_edge_index, c_node_attr, c_edge_attr = formula_data(conj)
            p_nodes, p_x, p_term_walk, p_edge_index, p_node_attr, p_edge_attr = formula_data(prem)
            cross_indices = self.build_cross_index(c_nodes, p_nodes)
            
            similarity1 = simi_dict[conj][prem]
            similarity2 = simi_sw_dict[conj][prem]

            data = PairData(
                x_s=c_x, term_walk_index_s=c_term_walk,
                edge_index_s=c_edge_index, node_attr_s=c_node_attr,
                edge_attr_s=c_edge_attr,
                cross_index_s=torch.from_numpy(cross_indices),
                x_t=p_x, term_walk_index_t=p_term_walk,
                edge_index_t=p_edge_index, node_attr_t=p_node_attr,
                edge_attr_t=p_edge_attr,
                cross_index_t=torch.from_numpy(cross_indices)[[1, 0]],
                y=torch.LongTensor([label]),
                similarity=torch.tensor([similarity1, similarity2], dtype=torch.float32))
            if symbolic_dict:
                data.symbolic_features = torch.tensor(
                    symbolic_dict[(conj, prem)], dtype=torch.float32).view(1, -1)
            dataList.append(data)
        data, slices = self.collate(data_list=dataList)
        target = Path(self.processed_paths[0])
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            torch.save((data, slices), temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
