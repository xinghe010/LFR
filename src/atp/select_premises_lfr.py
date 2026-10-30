
import os, sys, json, pickle, re, argparse, hashlib, time
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model_args(path):
    with open(path) as stream:
        document = json.load(stream)
    options = document.get('args', document.get('config', document))
    if not isinstance(options, dict):
        raise ValueError('Model config must contain an args/config object')
    return options

def main():
    parser = argparse.ArgumentParser(description='TW-GNN-LFR Premise Selection')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True,
                        help='Frozen training config/effective_args for this checkpoint')
    parser.add_argument('--training_result', type=str, required=True,
                        help='ATP selector results.json proving checkpoint/config provenance')
    parser.add_argument('--component', required=True,
                        choices=('tw', 'tw_lce', 'tw_rules', 'full'))
    parser.add_argument('--statements', type=str, default='statements')
    parser.add_argument('--node_dict', type=str, default='node_dict.pkl')
    parser.add_argument('--node_attr_dict', type=str, default='node_attr_dict.pkl')
    parser.add_argument('--edge_attr_dict', type=str, default='edge_attr_dict.pkl')
    parser.add_argument('--chainy_dir', type=str, required=True)
    parser.add_argument('--test_problems', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--top_k', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference')
    parser.add_argument('--similarity2', type=str, default=None,
                        help='corrected trigger-rule JSON (default: beside statements)')
    parser.add_argument('--similarity4', type=str, default=None,
                        help='corrected weighted-similarity JSON (default: beside statements)')
    parser.add_argument('--rule_metadata', type=str, default=None)
    parser.add_argument('--candidate_manifest', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    with open(args.node_dict, 'rb') as f:
        node_dict = pickle.load(f)
    node_attr_dict = load_pickle_file(args.node_attr_dict)
    edge_attr_dict = load_pickle_file(args.edge_attr_dict)

    training_args = load_model_args(args.model_config)
    with open(args.training_result) as stream:
        training_result = json.load(stream)
    if (training_result.get('status') != 'PASS' or
            training_result.get('component') != args.component or
            training_result.get('epochs') != 30 or
            training_result.get('checkpoint_sha256') != file_sha256(args.model) or
            training_result.get('effective_config_sha256') != file_sha256(args.model_config)):
        raise ValueError('Training result does not authenticate this 30-epoch checkpoint/config')
    option_keys = {
        'use_lce_and', 'use_lce_or', 'use_lce_not', 'use_lce_contains',
        'use_lce_equivalence', 'use_lce_forall', 'use_lce_exists',
        'use_classifier_constraint', 'lce_scale', 'rho_w', 'rho_t', 'rho_s',
        'use_trigger_rule', 'use_similarity_rule', 'encoder_variant',
        'relation_bases', 'generic_mlp_width', 'symbolic_baseline',
        'use_term_walk', 'quantifier_aggregation', 'commutative_aggregation',
    }
    model_options = {key: training_args[key] for key in option_keys
                     if key in training_args}
    expected_components = {
        'tw': ('tw', 0), 'tw_lce': ('lce', 0),
        'tw_rules': ('tw', 1), 'full': ('lce', 1),
    }
    observed = (model_options.get('encoder_variant', 'lce'),
                int(model_options.get('use_classifier_constraint', 1)))
    if observed != expected_components[args.component]:
        raise ValueError(
            f'Component {args.component} requires {expected_components[args.component]}, '
            f'but model config specifies {observed}')
    operator_names = {
        'exists': '?', 'not': '~', 'and': '&', 'equivalence': '<=>',
        'forall': '!', 'or': '|', 'implication': '=>',
    }
    missing_operators = [symbol for symbol in operator_names.values()
                         if symbol not in node_attr_dict]
    if missing_operators:
        raise ValueError(f'ATP node-attribute dictionary lacks {missing_operators}')
    model_options['operator_type_indices'] = {
        name: node_attr_dict[symbol] for name, symbol in operator_names.items()}
    model_options['variable_type_index'] = node_attr_dict['var']
    dimension = int(training_args['node_out_channels'])
    layers = int(training_args['layers'])

    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    model = PremiseSelectionModel(
        node_in_channels=len(node_dict),
        node_out_channels=dimension,
        layers=layers,
        **model_options,
    ).to(args.device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded: {len(node_dict)} node types, dim={dimension}, layers={layers}")

    stmts = Statements(args.statements)
    print(f"Statements: {len(stmts)}")

    rules_dir = os.path.dirname(os.path.abspath(args.statements))
    similarity2_path = args.similarity2 or os.path.join(rules_dir, 'similarity2.json')
    similarity4_path = args.similarity4 or os.path.join(rules_dir, 'similarity4.json')
    metadata_path = args.rule_metadata or os.path.join(rules_dir, 'rule_cache_metadata.json')
    for rule_path in (similarity2_path, similarity4_path, metadata_path):
        if not os.path.isfile(rule_path):
            raise FileNotFoundError(
                f"Rule cache missing: {rule_path}. Generate corrected caches before ATP ranking.")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if not metadata.get('corrected'):
        raise ValueError(f'Rule cache is not marked corrected: {metadata_path}')
    expected_hashes = metadata['outputs']
    if file_sha256(similarity2_path) != expected_hashes['similarity2_sha256']:
        raise ValueError('similarity2 hash does not match rule metadata')
    if file_sha256(similarity4_path) != expected_hashes['similarity4_sha256']:
        raise ValueError('similarity4 hash does not match rule metadata')
    if file_sha256(args.statements) != metadata['inputs']['statements_sha256']:
        raise ValueError('statements hash does not match rule metadata')
    with open(args.candidate_manifest) as stream:
        candidate_manifest = json.load(stream)
    if (candidate_manifest.get('status') != 'PASS' or
            candidate_manifest.get('labels_used') is not False or
            candidate_manifest.get('test_problems_sha256') != file_sha256(args.test_problems) or
            candidate_manifest.get('statements_sha256') != file_sha256(args.statements) or
            candidate_manifest.get('rule_cache_metadata_sha256') != file_sha256(metadata_path)):
        raise ValueError('ATP candidate/rule manifest is invalid or belongs to different inputs')
    with open(similarity2_path) as f:
        trigger_rules = json.load(f)
    with open(similarity4_path) as f:
        weighted_rules = json.load(f)

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    if isinstance(test_probs, dict):
        prob_names = sorted(test_probs.keys())
    else:
        prob_names = sorted(test_probs)
    print(f"Test problems: {len(prob_names)}")

    print("Building graph cache...")
    graph_cache = {}
    graph_failures = []
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
        except Exception as exc:
            graph_failures.append((name, f"{type(exc).__name__}: {exc}"))
    if graph_failures:
        preview = '; '.join(
            f"{name}: {error}" for name, error in graph_failures[:5])
        raise RuntimeError(
            f"Failed to build {len(graph_failures)} formula graph(s); "
            f"first failures: {preview}")
    print(f"Cached {len(graph_cache)} graphs")

    rankings = {}
    candidate_counts = {}
    ranking_started = time.perf_counter()
    print("\nGenerating rankings...")

    for prob_name in tqdm(prob_names, desc="Ranking"):
        pf = find_problem_file(prob_name, args.chainy_dir)
        if not pf or prob_name not in graph_cache:
            rankings[prob_name] = []
            continue

        conj_data = graph_cache[prob_name]
        local_axioms = parse_axioms_from_file(pf)

        valid_axioms = [ax for ax in local_axioms if ax in graph_cache]
        candidate_counts[prob_name] = len(valid_axioms)
        expected_candidates = candidate_manifest['candidate_counts'].get(prob_name)
        if expected_candidates != len(valid_axioms):
            raise RuntimeError(
                f'Candidate set changed for {prob_name}: expected {expected_candidates}, '
                f'found {len(valid_axioms)} graphable axioms')
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
                    try:
                        rule_features = [trigger_rules[prob_name][ax_name],
                                         weighted_rules[prob_name][ax_name]]
                    except KeyError as exc:
                        raise KeyError(
                            f"Rule cache has no entry for ({prob_name}, {ax_name})") from exc
                    p_x, p_tw, p_ei, p_na, p_ea = graph_cache[ax_name]
                    pair = PairData(
                        x_s=c_x, term_walk_index_s=c_tw,
                        edge_index_s=c_ei, node_attr_s=c_na, edge_attr_s=c_ea,
                        cross_index_s=torch.from_numpy(c_cross_empty),
                        x_t=p_x, term_walk_index_t=p_tw,
                        edge_index_t=p_ei, node_attr_t=p_na, edge_attr_t=p_ea,
                        cross_index_t=torch.from_numpy(c_cross_empty),
                        y=torch.LongTensor([0]),
                        similarity=torch.tensor(rule_features, dtype=torch.float32)
                    )
                    pair_list.append(pair)

                loader = DataLoader(pair_list, batch_size=len(pair_list),
                                    follow_batch=['x_s', 'x_t'])
                for batch in loader:
                    batch = batch.to(args.device)
                    probs = model.predict_proba(batch)[:, 1]
                    all_scores.extend(list(zip(batch_axioms, probs.cpu().tolist())))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        rankings[prob_name] = [name for name, _ in all_scores[:args.top_k]]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rankings, f, indent=2)

    elapsed = time.perf_counter() - ranking_started
    metadata_output = args.output + '.metadata.json'
    with open(metadata_output, 'w') as stream:
        json.dump({
            'schema': 'tw-gnn-atp-ranking-v1', 'status': 'PASS',
            'component': args.component, 'problems': len(prob_names),
            'checkpoint': os.path.abspath(args.model),
            'checkpoint_sha256': file_sha256(args.model),
            'model_config': os.path.abspath(args.model_config),
            'model_config_sha256': file_sha256(args.model_config),
            'training_result_sha256': file_sha256(args.training_result),
            'test_problems_sha256': file_sha256(args.test_problems),
            'statements_sha256': file_sha256(args.statements),
            'rule_metadata_sha256': file_sha256(metadata_path),
            'candidate_manifest_sha256': file_sha256(args.candidate_manifest),
            'candidate_counts': candidate_counts,
            'candidate_total': sum(candidate_counts.values()),
            'top_k': args.top_k, 'ranking_seconds': elapsed,
            'candidates_per_second': sum(candidate_counts.values()) / max(elapsed, 1e-12),
            'output_sha256': file_sha256(args.output),
            'operator_type_indices': model_options['operator_type_indices'],
            'variable_type_index': model_options['variable_type_index'],
        }, stream, indent=2, sort_keys=True)

    print(f"\nSaved: {args.output}")
    print(f"Problems: {len(rankings)}")
    if rankings:
        avg_prems = sum(len(v) for v in rankings.values()) / len(rankings)
        print(f"Avg premises/problem: {avg_prems:.0f}")

if __name__ == '__main__':
    main()
