
import os, sys, json, pickle, re, argparse, hashlib, time
import torch
import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src', 'mptp'))
from graph import Graph, Node
from src.atp.formula_parser import fof_formula_transformer
from src.atp.dataset import PairData, FormulaGraphDataset
from utils import Statements, load_pickle_file
from torch_geometric.loader import DataLoader
from fusion_model import PremiseSelectionModel

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
    parser.add_argument('--symbolic_cache', type=str, required=True,
                        help='six-column cache generated on the D0/candidate union')
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
    if (training_result.get('schema') != 'twgnn-revised-selector-result-v2' or
            training_result.get('purpose') != 'atp_d0' or
            training_result.get('status') != 'PASS' or
            training_result.get('component') != args.component or
            training_result.get('epochs') != 30 or
            training_result.get('checkpoint_sha256') != file_sha256(args.model) or
            training_result.get('effective_config_sha256') != file_sha256(args.model_config)):
        raise ValueError('Training result does not authenticate this revised checkpoint/config')
    if (training_args.get('purpose') != 'atp_d0' or training_args.get('epochs') != 30 or
            training_args.get('model', {}).get('fusion_mode') != 'continuous'):
        raise ValueError('ATP ranking requires the 30-epoch continuous model config')
    model_options = dict(training_args['model'])
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
    if (not isinstance(checkpoint, dict) or
            checkpoint.get('epoch') != training_result.get('selection', {}).get('epoch') or
            checkpoint.get('config_sha256') != training_args.get('config_sha256') or
            checkpoint.get('selection') != 'max validation ACC; earliest epoch on tie'):
        raise ValueError('Checkpoint does not match the validation-selected training result')
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
    if (candidate_manifest.get('schema') != 'twgnn-atp-unified-candidate-cache-v2' or
            candidate_manifest.get('status') != 'PASS' or
            candidate_manifest.get('labels_used') is not False or
            candidate_manifest.get('test_problems_sha256') != file_sha256(args.test_problems) or
            candidate_manifest.get('statements_sha256') != file_sha256(args.statements) or
            candidate_manifest.get('rule_cache_metadata_sha256') != file_sha256(metadata_path) or
            candidate_manifest.get('symbolic_cache_metadata_sha256') !=
            file_sha256(args.symbolic_cache + '.metadata.json') or
            training_result.get('provenance', {}).get('candidate_manifest_sha256') !=
            file_sha256(args.candidate_manifest) or
            training_result.get('provenance', {}).get('rule_metadata_sha256') !=
            file_sha256(metadata_path) or
            training_result.get('provenance', {}).get('symbolic_metadata_sha256') !=
            file_sha256(args.symbolic_cache + '.metadata.json')):
        raise ValueError('ATP candidate/rule manifest is invalid or belongs to different inputs')
    with open(args.symbolic_cache + '.metadata.json') as stream:
        symbolic_metadata = json.load(stream)
    if (symbolic_metadata.get('labels_ignored') is not True or
            symbolic_metadata.get('output_sha256') != file_sha256(args.symbolic_cache)):
        raise ValueError('ATP continuous feature cache is invalid')
    with open(similarity2_path) as f:
        trigger_rules = json.load(f)
    with open(similarity4_path) as f:
        weighted_rules = json.load(f)
    symbolic_features = {}
    with open(args.symbolic_cache) as stream:
        for line in stream:
            item = json.loads(line)
            if item['conjecture'] in candidate_manifest['candidate_counts']:
                key = (item['conjecture'], item['premise'])
                symbolic_features[key] = [item[name] for name in (
                    'overlap', 'idf', 'sine', 'trigger',
                    'weighted_similarity', 'weighted_similarity_score')]

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    if isinstance(test_probs, dict):
        prob_names = sorted(test_probs.keys())
    else:
        prob_names = sorted(test_probs)
    print(f"Test problems: {len(prob_names)}")

    print("Building graph cache with the training dataset's graph encoding...")
    graph_cache = {}
    graph_failures = []
    candidate_pairs_path = os.path.join(os.path.dirname(args.candidate_manifest),
                                        'candidate_pairs.jsonl')
    if file_sha256(candidate_pairs_path) != candidate_manifest['candidate_pairs_sha256']:
        raise ValueError('candidate pair list changed after cache generation')
    needed = set(prob_names)
    with open(candidate_pairs_path) as stream:
        for line in stream:
            conjecture, premise = json.loads(line)
            needed.add(conjecture)
            needed.add(premise)
    for name in tqdm(sorted(needed), desc="Caching graphs"):
        try:
            formula = stmts.statements[name]
            Node.rest_id()
            g = Graph(fof_formula_transformer(formula), rename=True)
            nodes, tw_idx, edge_idx, node_attrs, edge_attrs = (
                FormulaGraphDataset.graph_process(None, g))
            if len(nodes) != len(node_attrs):
                continue
            unknown = node_dict.get('__UNK__')
            x = torch.LongTensor([node_dict.get(token, unknown) if unknown is not None
                                  else node_dict[token] for token in nodes])
            tw = torch.from_numpy(tw_idx)
            ei = torch.from_numpy(edge_idx)
            na = FormulaGraphDataset.vectorization(None, node_attrs, node_attr_dict)
            ea = (FormulaGraphDataset.vectorization(None, edge_attrs, edge_attr_dict)
                  if edge_attrs else torch.zeros((0, len(edge_attr_dict))))

            graph_cache[name] = (x, tw, ei, na, ea)
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
                    try:
                        continuous_features = symbolic_features[(prob_name, ax_name)]
                    except KeyError as exc:
                        raise KeyError(f'Six-column cache misses ({prob_name}, {ax_name})') from exc
                    if int(continuous_features[3]) != int(rule_features[0]):
                        raise ValueError('Trigger feature disagrees with corrected rule cache')
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
                    pair.symbolic_features = torch.tensor(
                        continuous_features, dtype=torch.float32).view(1, 6)
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
            'schema': 'tw-gnn-atp-ranking-continuous-v2', 'status': 'PASS',
            'component': args.component, 'problems': len(prob_names),
            'checkpoint': os.path.abspath(args.model),
            'checkpoint_sha256': file_sha256(args.model),
            'model_config': os.path.abspath(args.model_config),
            'model_config_sha256': file_sha256(args.model_config),
            'training_result_sha256': file_sha256(args.training_result),
            'test_problems_sha256': file_sha256(args.test_problems),
            'statements_sha256': file_sha256(args.statements),
            'rule_metadata_sha256': file_sha256(metadata_path),
            'symbolic_cache_sha256': file_sha256(args.symbolic_cache),
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
