
import os, sys, json, collections, math, hashlib, argparse, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Statements
import collections
import re

_SYMBOL = re.compile(r'(?<![A-Za-z0-9_])[a-z][a-z0-9_]*')
_LOGICAL_WORDS = {'true', 'false'}

def _extract_symbols(formula):
    
    
    first_comma = formula.find(',')
    second_comma = formula.find(',', first_comma + 1)
    if first_comma < 0 or second_comma < 0:
        raise ValueError(f"Malformed TPTP statement: {formula[:80]}")
    body = formula[second_comma + 1:]
    return {token for token in _SYMBOL.findall(body)
            if token not in _LOGICAL_WORDS}

def _load_pairs(path):
    paths = sorted(p for p in ([path] if os.path.isfile(path) else []) or
                   [str(p) for p in __import__('pathlib').Path(path).glob('*/raw/*.json')])
    if not paths:
        raise FileNotFoundError(f"No pair JSON files found under {path}")
    pairs = []
    for pair_path in paths:
        text = open(pair_path).read()
        try:
            loaded = json.loads(text)
            pairs.extend(loaded if isinstance(loaded, list) else [loaded])
        except json.JSONDecodeError:
            pairs.extend(json.loads(line) for line in text.splitlines() if line.strip())
    
    
    identifiers = []
    for sample in pairs:
        if isinstance(sample, dict):
            identifiers.append({"conjecture": sample["conjecture"], "premise": sample["premise"]})
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            identifiers.append([sample[0], sample[1]])
        else:
            raise ValueError(f"Malformed pair record: {sample!r}")
    return identifiers, paths

def _build_tri_dict(fof_dict):

    symbol_count = collections.Counter()
    for symbols in fof_dict.values():
        for s in symbols:
            symbol_count[s] += 1

    tri_dict = collections.defaultdict(set)
    for name, symbols in sorted(fof_dict.items()):
        if not symbols:
            continue
        min_count = min(symbol_count[s] for s in symbols)
        for symbol in sorted(symbols):
            if symbol_count[symbol] == min_count:
                tri_dict[symbol].add(name)
    return tri_dict

def generate_similarity2(statements, train_data):
    pairs_by_conjecture = collections.defaultdict(set)
    for sample in train_data:
        if isinstance(sample, dict):
            pairs_by_conjecture[sample['conjecture']].add(sample['premise'])
        else:
            pairs_by_conjecture[sample[0]].add(sample[1])

    premise_names = set().union(*pairs_by_conjecture.values())
    needed_names = set(pairs_by_conjecture) | premise_names
    missing_names = needed_names - set(statements.statements)
    if missing_names:
        raise KeyError(f"Statements missing {len(missing_names)} requested formula(s)")
    all_symbols = {
        name: _extract_symbols(statements.statements[name])
        for name in sorted(needed_names)
    }
    premise_symbols = {name: all_symbols.get(name, set()) for name in premise_names}
    tri_dict = _build_tri_dict(premise_symbols)
    simi_dict = {}

    for conj in sorted(pairs_by_conjecture):
        simi_dict[conj] = {}
        conj_symbols = all_symbols.get(conj, set())

        triggered = set()
        for s in sorted(conj_symbols):
            if s in tri_dict:
                triggered.update(tri_dict[s])

        triggered2 = set(triggered)
        for t in sorted(triggered):
            t_symbols = premise_symbols.get(t, set())
            for s in sorted(t_symbols):
                if s in tri_dict:
                    triggered2.update(tri_dict[s])

        for prem in sorted(pairs_by_conjecture[conj]):
            simi_dict[conj][prem] = 1 if prem in triggered2 else 0

    return simi_dict

def generate_similarity4(statements, train_data, h=0.2, w0=1.0):
    pairs_by_conjecture = collections.defaultdict(set)
    for sample in train_data:
        if isinstance(sample, dict):
            pairs_by_conjecture[sample['conjecture']].add(sample['premise'])
        else:
            pairs_by_conjecture[sample[0]].add(sample[1])

    premise_names = set().union(*pairs_by_conjecture.values())
    needed_names = set(pairs_by_conjecture) | premise_names
    missing_names = needed_names - set(statements.statements)
    if missing_names:
        raise KeyError(f"Statements missing {len(missing_names)} requested formula(s)")
    all_formula_symbols = {
        name: _extract_symbols(statements.statements[name])
        for name in sorted(needed_names)
    }
    premise_symbols = {
        name: all_formula_symbols.get(name, set()) for name in premise_names
    }
    symbol_count = collections.Counter(
        symbol for symbols in premise_symbols.values() for symbol in symbols)
    symbol_weights = {}
    for symbol, count in symbol_count.items():
        occ = float(count)
        softplus_occ = occ + math.log1p(math.exp(-occ))
        symbol_weights[symbol] = 1.0 + float(w0) / softplus_occ

    simi_dict = {}
    for conj in sorted(pairs_by_conjecture):
        simi_dict[conj] = {}
        conj_symbols = all_formula_symbols.get(conj, set())

        selected = set()
        for name in sorted(pairs_by_conjecture[conj]):
            symbols = premise_symbols[name]
            denominator = sum(symbol_weights[s] for s in sorted(symbols))
            relevance = (sum(symbol_weights[s] for s in sorted(conj_symbols & symbols))
                         / denominator) if denominator else 0.0
            if relevance >= h:
                selected.add(name)

        for prem in sorted(pairs_by_conjecture[conj]):
            simi_dict[conj][prem] = 1 if prem in selected else 0

    return simi_dict

def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate corrected LFR rule caches")
    parser.add_argument("statements", nargs="?", default="dataset/statements")
    parser.add_argument("train_data", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=".")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--keep-processed", action="store_true",
                        help="Do not invalidate PyG processed caches in output_dir")
    args = parser.parse_args()
    statements_path = args.statements
    train_data_path = args.train_data
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stmts = Statements(statements_path)
    print(f"Statements: {len(stmts)}")

    if train_data_path:
        train_data, pair_paths = _load_pairs(train_data_path)
        print(f"Requested pairs: {len(train_data)} from {len(pair_paths)} file(s)")
    else:
        parser.error("train_data is required and must list every pair needing a rule value")

    print("Generating similarity2 (two-pass triggering)...")
    stage_started = time.perf_counter()
    simi2 = generate_similarity2(stmts, train_data)
    path2 = os.path.join(output_dir, "similarity2.json")
    if Path(path2).is_symlink():
        raise RuntimeError(f"Refusing to overwrite rule-cache symlink: {path2}")
    with open(path2, 'w') as f:
        json.dump(simi2, f)
    trigger_seconds = time.perf_counter() - stage_started
    print(f"Saved: {path2} ({len(simi2)} conjectures)")

    print("Generating similarity4 (symbol weight)...")
    stage_started = time.perf_counter()
    simi4 = generate_similarity4(
        stmts, train_data, h=args.threshold, w0=args.w0)
    path4 = os.path.join(output_dir, "similarity4.json")
    if Path(path4).is_symlink():
        raise RuntimeError(f"Refusing to overwrite rule-cache symlink: {path4}")
    with open(path4, 'w') as f:
        json.dump(simi4, f)
    weighted_seconds = time.perf_counter() - stage_started
    print(f"Saved: {path4} ({len(simi4)} conjectures)")

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    metadata = {
        "schema": "tw-gnn-lfr-rule-cache-v1",
        "corrected": True,
        "trigger_rounds": 2,
        "weighted_similarity": {
            "threshold": args.threshold,
            "w0": args.w0,
            "weight": "1 + w0 / log(1 + exp(occ))",
            "coverage_denominator": "sum of weights over premise symbols",
        },
        "inputs": {
            "statements": os.path.abspath(statements_path),
            "statements_sha256": sha256(statements_path),
            "pair_files": [
                {"path": os.path.abspath(path), "sha256": sha256(path)}
                for path in pair_paths
            ],
            "labels_ignored": True,
            "candidate_library": "union of premise identifiers in requested pair files",
            "formula_visibility": "global unlabeled statement text",
        },
        "outputs": {
            "similarity2_sha256": sha256(path2),
            "similarity4_sha256": sha256(path4),
        },
        "resources": {
            "trigger_generation_seconds": trigger_seconds,
            "weighted_similarity_generation_seconds": weighted_seconds,
            "total_wall_seconds_before_metadata_write": time.perf_counter() - total_started,
        },
    }
    metadata_path = os.path.join(output_dir, "rule_cache_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Saved: {metadata_path}")
    if not args.keep_processed:
        removed = []
        for processed_file in Path(output_dir).glob('*/processed/*.pt'):
            processed_file.unlink()
            removed.append(str(processed_file))
        print(f"Invalidated {len(removed)} processed PyG cache file(s)")

if __name__ == '__main__':
    main()
