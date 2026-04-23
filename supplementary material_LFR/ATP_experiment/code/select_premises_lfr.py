#!/usr/bin/env python3
import os, sys, json, pickle, re, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PremiseSelectionModel
from graph import Graph, Node
from formula_parser import fof_formula_transformer
from dataset import PairData, FormulaGraphDataset
from utils import Statements, load_pickle_file
from torch_geometric.loader import DataLoader

VARIABLE_PATTERN = r"^[A-Z][A-Z_0-9]*$"
FUNCTOR_PATTERN = r"^[a-z][a-z_0-9]*$"
CONNECTIVE_PATTERN = {"&", "|", "~", "!", "?", "=>", "<=>", "=", "$true", "$false"}

def find_problem_file(problem_name, problem_dir):
    match = re.match(r'[a-z]+\d*_(.+)', problem_name)
    if match:
        theory = match.group(1)
        candidate = os.path.join(problem_dir, f"{theory}__{problem_name}.p")
        if os.path.exists(candidate):
            return candidate
    for fn in os.listdir(problem_dir):
        if fn.endswith('.p') and problem_name in fn:
            return os.path.join(problem_dir, fn)
    return None

def parse_axioms_from_file(filepath):
    axioms = {}
    buffer = []
    with open(filepath) as f:
        for line in f:
            if '%' in line:
                line = line.split('%')[0]
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if line.endswith('.'):
                full = ' '.join(buffer)
                buffer = []
                clean = full.replace(' ', '')
                if full.startswith('fof') and ',axiom,' in clean:
                    idx = full.find('(') + 1
                    name = full[idx:full.find(',', idx)].strip()
                    axioms[name] = full
    return axioms

def graph_process_full(G):
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
                    edge_attrs.append(f"{node.name}_var")
                    edge_attrs_reverse.append(f"var_{node.name}")
                elif re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.append(f"{node.name}_pred")
                    edge_attrs_reverse.append(f"pred_{node.name}")
                elif child.name == "=":
                    edge_attrs.append(f"{node.name}_=")
                    edge_attrs_reverse.append(f"=_{node.name}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.append(f"{node.name}_{child.name}")
                    edge_attrs_reverse.append(f"{child.name}_{node.name}")
                else:
                    continue
        elif node.name in {"&", "|", "<=>", "~"}:
            node_attrs.append(node.name)
            for child in node.children:
                edge_indices.append([node.id, child.id])
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.append(f"{node.name}_pred")
                    edge_attrs_reverse.append(f"pred_{node.name}")
                elif child.name == "=":
                    edge_attrs.append(f"{node.name}_=")
                    edge_attrs_reverse.append(f"=_{node.name}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.append(f"{node.name}_{child.name}")
                    edge_attrs_reverse.append(f"{child.name}_{node.name}")
                else:
                    continue
        elif node.name == "=>":
            node_attrs.append(node.name)
            for index, child in enumerate(node.children, 1):
                edge_indices.append([node.id, child.id])
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.append(f"{node.name}_pred_{index}")
                    edge_attrs_reverse.append(f"pred_{node.name}_{index}")
                elif child.name == "=":
                    edge_attrs.append(f"{node.name}_=_{index}")
                    edge_attrs_reverse.append(f"=_{node.name}_{index}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.append(f"{node.name}_{child.name}_{index}")
                    edge_attrs_reverse.append(f"{child.name}_{node.name}_{index}")
                else:
                    continue
        elif node.name == "=":
            node_attrs.append(node.name)
            for child in node.children:
                edge_indices.append([node.id, child.id])
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.append(f"{node.name}_func")
                    edge_attrs_reverse.append(f"func_{node.name}")
                elif re.match(VARIABLE_PATTERN, child.name):
                    edge_attrs.append(f"{node.name}_var")
                    edge_attrs_reverse.append(f"var_{node.name}")
                else:
                    continue
        elif re.match(FUNCTOR_PATTERN, node.name):
            parents_tokens = [parent.name for parent in node.parents]
            if len(parents_tokens) == 0 or parents_tokens[0] in CONNECTIVE_PATTERN:
                node_attrs.append("pred")
                for index, child in enumerate(node.children, 1):
                    edge_indices.append([node.id, child.id])
                    if re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append(f"pred_func_{index}")
                        edge_attrs_reverse.append(f"func_pred_{index}")
                    elif re.match(VARIABLE_PATTERN, child.name):
                        edge_attrs.append(f"pred_var_{index}")
                        edge_attrs_reverse.append(f"var_pred_{index}")
                    else:
                        continue
            else:
                node_attrs.append("func")
                for index, child in enumerate(node.children, 1):
                    edge_indices.append([node.id, child.id])
                    if re.match(FUNCTOR_PATTERN, child.name):
                        edge_attrs.append(f"func_func_{index}")
                        edge_attrs_reverse.append(f"func_func_{index}")
                    elif re.match(VARIABLE_PATTERN, child.name):
                        edge_attrs.append(f"func_var_{index}")
                        edge_attrs_reverse.append(f"var_func_{index}")
                    else:
                        continue
        elif node.name == "$true":
            node_attrs.append(node.name)
        elif re.match(VARIABLE_PATTERN, node.name):
            node_attrs.append("var")
        else:
            node_attrs.append(node.name)

    if edge_indices:
        edge_indices = np.array(edge_indices, dtype=np.int64).reshape(-1, 2).T
    else:
        edge_indices = np.zeros((2, 0), dtype=np.int64)
    edge_attrs = edge_attrs + edge_attrs_reverse

    for node in G:
        if node.parents and node.children:
            for parent in node.parents:
                for child in node.children:
                    term_walk_indices.append([parent.id, node.id, child.id])

    if term_walk_indices:
        term_walk_indices = np.array(term_walk_indices, dtype=np.int64).reshape(-1, 3).T
    else:
        term_walk_indices = np.zeros((3, 0), dtype=np.int64)

    return nodes, term_walk_indices, edge_indices, node_attrs, edge_attrs

def vectorization(objects, object_dict):
    indices = [object_dict.get(obj, 0) for obj in objects]
    from torch.nn.functional import one_hot
    onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
    return onehot

def main():
    parser = argparse.ArgumentParser(description='TW-GNN-LFR Premise Selection')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--statements', type=str, default='statements')
    parser.add_argument('--node_dict', type=str, default='node_dict.pkl')
    parser.add_argument('--node_attr_dict', type=str, default='node_attr_dict.pkl')
    parser.add_argument('--edge_attr_dict', type=str, default='edge_attr_dict.pkl')
    parser.add_argument('--chainy_dir', type=str, required=True)
    parser.add_argument('--test_problems', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--top_k', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=32, help='node_out_channels')
    parser.add_argument('--layers', type=int, default=2, help='TW layers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for inference')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    with open(args.node_dict, 'rb') as f:
        node_dict = pickle.load(f)
    node_attr_dict = load_pickle_file(args.node_attr_dict)
    edge_attr_dict = load_pickle_file(args.edge_attr_dict)

    checkpoint = torch.load(args.model, map_location=args.device)
    model = PremiseSelectionModel(
        node_in_channels=len(node_dict),
        node_out_channels=args.dim,
        layers=args.layers
    ).to(args.device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded: {len(node_dict)} node types, dim={args.dim}, layers={args.layers}")

    stmts = Statements(args.statements)
    print(f"Statements: {len(stmts)}")

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    if isinstance(test_probs, dict):
        prob_names = sorted(test_probs.keys())
    else:
        prob_names = sorted(test_probs)
    print(f"Test problems: {len(prob_names)}")

    print("Building graph cache...")
    graph_cache = {}
    for name in tqdm(stmts.statements.keys(), desc="Caching graphs"):
        try:
            formula = stmts.statements[name]
            Node.rest_id()
            g = Graph(fof_formula_transformer(formula), rename=True)
            nodes, tw_idx, edge_idx, node_attrs, edge_attrs = graph_process_full(g)
            if len(nodes) != len(node_attrs):
                continue

            x = vectorization(nodes, node_dict)
            tw = torch.from_numpy(tw_idx)
            ei = torch.from_numpy(edge_idx)

            ei_rev = ei.flip(0) if ei.size(1) > 0 else ei
            ei_full = torch.cat([ei, ei_rev], dim=1) if ei.size(1) > 0 else ei
            na = vectorization(node_attrs, node_attr_dict)
            ea = vectorization(edge_attrs, edge_attr_dict) if edge_attrs else torch.zeros((0, len(edge_attr_dict)))

            graph_cache[name] = (x, tw, ei_full, na, ea)
        except Exception:
            pass
    print(f"Cached {len(graph_cache)} graphs")

    rankings = {}
    print("\nGenerating rankings...")

    for prob_name in tqdm(prob_names, desc="Ranking"):
        pf = find_problem_file(prob_name, args.chainy_dir)
        if not pf or prob_name not in graph_cache:
            rankings[prob_name] = []
            continue

        conj_data = graph_cache[prob_name]
        local_axioms = parse_axioms_from_file(pf)

        valid_axioms = [ax for ax in local_axioms if ax in graph_cache]
        if not valid_axioms:
            rankings[prob_name] = []
            continue

        c_x, c_tw, c_ei, c_na, c_ea = conj_data
        c_cross_empty = np.zeros((2, 0), dtype=np.int64)

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(valid_axioms), args.batch_size):
                batch_axioms = valid_axioms[i:i + args.batch_size]
                pair_list = []
                for ax_name in batch_axioms:
                    p_x, p_tw, p_ei, p_na, p_ea = graph_cache[ax_name]
                    pair = PairData(
                        x_s=c_x, term_walk_index_s=c_tw,
                        edge_index_s=c_ei, node_attr_s=c_na, edge_attr_s=c_ea,
                        cross_index_s=torch.from_numpy(c_cross_empty),
                        x_t=p_x, term_walk_index_t=p_tw,
                        edge_index_t=p_ei, node_attr_t=p_na, edge_attr_t=p_ea,
                        cross_index_t=torch.from_numpy(c_cross_empty),
                        y=torch.LongTensor([0]),
                        similarity=torch.FloatTensor([0.0])
                    )
                    pair_list.append(pair)

                loader = DataLoader(pair_list, batch_size=len(pair_list),
                                    follow_batch=['x_s', 'x_t'])
                for batch in loader:
                    batch = batch.to(args.device)
                    h_s = model.initial(batch.x_s)
                    h_t = model.initial(batch.x_t)
                    h_s = model.dag_emb(h_s, batch.term_walk_index_s,
                                        batch.edge_index_s, batch.node_attr_s,
                                        batch.x_s_batch)
                    h_t = model.dag_emb(h_t, batch.term_walk_index_t,
                                        batch.edge_index_t, batch.node_attr_t,
                                        batch.x_t_batch)
                    h_g_s = model.pooling(h_s, batch.x_s_batch)
                    h_g_t = model.pooling(h_t, batch.x_t_batch)

                    x_concat = torch.cat([h_g_s, h_g_t], dim=1)
                    pred = model.classifier.first(x_concat)
                    pred = model.classifier.second.lin(pred)
                    if model.classifier.second.batch and pred.size(0) > 1:
                        pred = model.classifier.second.BN(pred)
                    probs = F.softmax(pred, dim=1)[:, 1]
                    all_scores.extend(list(zip(batch_axioms, probs.cpu().tolist())))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        rankings[prob_name] = [name for name, _ in all_scores[:args.top_k]]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rankings, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Problems: {len(rankings)}")
    if rankings:
        avg_prems = sum(len(v) for v in rankings.values()) / len(rankings)
        print(f"Avg premises/problem: {avg_prems:.0f}")

if __name__ == '__main__':
    main()
