
"""Generate label-blind overlap, IDF, SInE, and proposed-rule features."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import re
import time

SYMBOL = re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9_]*")
LOGICAL_WORDS = {"true", "false"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_statements(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        first_comma = line.find(",")
        if first_comma < 0 or "(" not in line[:first_comma]:
            raise ValueError(f"Malformed TPTP statement: {line[:80]}")
        key = line[line.find("(") + 1:first_comma].strip()
        result[key] = line
    return result


def symbols(formula):
    first = formula.find(",")
    second = formula.find(",", first + 1)
    if first < 0 or second < 0:
        raise ValueError(f"Malformed TPTP statement: {formula[:80]}")
    return {token for token in SYMBOL.findall(formula[second + 1:])
            if token not in LOGICAL_WORDS}


def load_pairs(paths):
    rows = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append((item["conjecture"], item["premise"]))
            elif isinstance(item, list) and len(item) >= 2:
                rows.append((item[0], item[1]))
            else:
                raise ValueError(f"Malformed pair at {path}:{number}")
    return rows


def sine_selected(conjecture_symbols, premise_symbols, occurrence,
                  tolerance, generality_threshold, depth):
    triggers = collections.defaultdict(set)
    for premise, premise_set in premise_symbols.items():
        if not premise_set:
            continue
        least = min(occurrence[symbol] for symbol in premise_set)
        for symbol in premise_set:
            if (occurrence[symbol] <= generality_threshold or
                    occurrence[symbol] <= tolerance * least):
                triggers[symbol].add(premise)
    active_symbols = set(conjecture_symbols)
    selected = set()
    for _ in range(depth):
        newly_selected = set().union(
            *(triggers.get(symbol, set()) for symbol in active_symbols))
        newly_selected -= selected
        if not newly_selected:
            break
        selected.update(newly_selected)
        active_symbols.update(*[premise_symbols[name] for name in newly_selected])
    return selected


def proposed_trigger_selected(conjecture_symbols, premise_symbols, occurrence):
    """Two-pass least-frequency triggering used by the submitted LFR model."""
    return sine_selected(conjecture_symbols, premise_symbols, occurrence,
                         tolerance=1.0, generality_threshold=0, depth=2)


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statements", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sine-tolerance", type=float, default=1.5)
    parser.add_argument("--sine-generality-threshold", type=int, default=0)
    parser.add_argument("--sine-depth", type=int, default=2)
    parser.add_argument("--weighted-threshold", type=float, default=0.2)
    parser.add_argument("--w0", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new path")
    if args.sine_tolerance < 1 or args.sine_depth < 0:
        parser.error("SInE requires tolerance >= 1 and depth >= 0")

    preprocessing_started = time.perf_counter()
    statement_map = load_statements(args.statements)
    input_pairs = load_pairs(args.pairs)
    
    
    pairs = sorted(set(input_pairs))
    names = {name for pair in pairs for name in pair}
    missing = names - set(statement_map)
    if missing:
        raise KeyError(f"Missing {len(missing)} formulas")
    formula_symbols = {name: symbols(statement_map[name]) for name in sorted(names)}
    premise_names = sorted({premise for _, premise in pairs})
    premise_symbols = {name: formula_symbols[name] for name in premise_names}
    occurrence = collections.Counter(
        symbol for values in premise_symbols.values() for symbol in values)
    count = len(premise_names)
    idf = {symbol: math.log((count + 1) / (frequency + 1)) + 1
           for symbol, frequency in occurrence.items()}
    rarity = {
        symbol: 1 + args.w0 /
        (float(frequency) + math.log1p(math.exp(-float(frequency))))
        for symbol, frequency in occurrence.items()
    }

    pairs_by_conjecture = collections.defaultdict(list)
    for conjecture, premise in pairs:
        pairs_by_conjecture[conjecture].append(premise)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    generation_started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for conjecture in sorted(pairs_by_conjecture):
            query = formula_symbols[conjecture]
            sine = sine_selected(
                query, premise_symbols, occurrence, args.sine_tolerance,
                args.sine_generality_threshold, args.sine_depth)
            trigger = proposed_trigger_selected(query, premise_symbols, occurrence)
            for premise in sorted(pairs_by_conjecture[conjecture]):
                candidate = premise_symbols[premise]
                shared = query & candidate
                overlap = len(shared) / len(candidate) if candidate else 0.0
                
                
                idf_denominator = sum(idf[symbol] for symbol in sorted(candidate))
                idf_score = (sum(idf[symbol] for symbol in sorted(shared)) /
                             idf_denominator if idf_denominator else 0.0)
                weighted_denominator = sum(
                    rarity[symbol] for symbol in sorted(candidate))
                weighted_score = (sum(rarity[symbol] for symbol in sorted(shared)) /
                                  weighted_denominator if weighted_denominator else 0.0)
                item = {
                    "conjecture": conjecture,
                    "premise": premise,
                    "overlap": overlap,
                    "idf": idf_score,
                    "sine": int(premise in sine),
                    "trigger": int(premise in trigger),
                    "weighted_similarity": int(
                        weighted_score >= args.weighted_threshold),
                    "weighted_similarity_score": weighted_score,
                }
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    generation_seconds = time.perf_counter() - generation_started
    metadata = {
        "schema": "tw-gnn-symbolic-baselines-v1",
        "labels_ignored": True,
        "input_pair_rows": len(input_pairs),
        "pairs": len(pairs),
        "duplicate_pair_rows_removed": len(input_pairs) - len(pairs),
        "premise_library_size": len(premise_names),
        "definitions": {
            "overlap": "|symbols(conjecture) intersect symbols(premise)| / |symbols(premise)|",
            "idf": "IDF-weighted shared-symbol coverage with smoothed IDF",
            "sine": "Hoder-Voronkov trigger relation with tolerance, generality threshold, and bounded depth",
            "trigger": "submitted two-pass least-frequency trigger (SInE tolerance=1, generality=0, depth=2)",
        },
        "parameters": {
            "sine_tolerance": args.sine_tolerance,
            "sine_generality_threshold": args.sine_generality_threshold,
            "sine_depth": args.sine_depth,
            "weighted_threshold": args.weighted_threshold,
            "w0": args.w0,
        },
        "inputs": {
            "statements": str(args.statements.resolve()),
            "statements_sha256": sha256(args.statements),
            "pair_files": [{"path": str(path.resolve()), "sha256": sha256(path)}
                           for path in args.pairs],
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "resources": {
            "input_and_statistics_seconds": preprocessing_seconds,
            "feature_generation_seconds": generation_seconds,
            "total_wall_seconds_before_metadata_write": time.perf_counter() - total_started,
        },
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
