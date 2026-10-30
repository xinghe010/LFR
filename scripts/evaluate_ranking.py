"""Evaluate premise rankings with MAP, MRR, and Recall@K.

Inputs
------
--rankings : JSON object mapping each conjecture name to an ordered list of
             premise names, as produced by src/atp/select_premises_*.py.
--labels   : JSONL/JSON rows [conjecture, premise, label] (or dictionaries with
             conjecture/premise/label keys).  Only label == 1 is relevant.

The script is model-agnostic: it evaluates neural and symbolic rankings alike.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_relevant(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_labels(path: Path) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return relevant
    try:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        if isinstance(row, dict):
            conjecture = row.get("conjecture") or row.get("problem") or row.get("c")
            premise = row.get("premise") or row.get("axiom") or row.get("p")
            label = row.get("label", row.get("y"))
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            conjecture, premise, label = row[0], row[1], row[2]
        else:
            raise ValueError(f"Malformed label row: {row!r}")
        if conjecture is None or premise is None:
            raise ValueError(f"Malformed label row: {row!r}")
        if _is_relevant(label):
            relevant[str(conjecture)].add(str(premise))
    return relevant


def load_rankings(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): [str(x) for x in v] for k, v in data.items()}
    if isinstance(data, list):
        result = {}
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("Ranking list rows must be objects")
            conjecture = row.get("conjecture") or row.get("problem")
            ranking = row.get("ranking") or row.get("premises")
            if conjecture is None or ranking is None:
                raise ValueError(f"Malformed ranking row: {row!r}")
            result[str(conjecture)] = [str(x) for x in ranking]
        return result
    raise ValueError("Rankings must be a JSON object or a list of objects")


def average_precision(ranking: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, name in enumerate(ranking, 1):
        if name in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, name in enumerate(ranking, 1):
        if name in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true",
                        help="fail if a ranked conjecture has no label row")
    args = parser.parse_args()

    rankings = load_rankings(args.rankings)
    relevant = load_labels(args.labels)
    if not rankings:
        raise ValueError("No rankings found")
    if not relevant:
        raise ValueError("No relevant labels found")

    per_problem = {}
    missing = []
    for conjecture, ranking in rankings.items():
        if conjecture not in relevant:
            missing.append(conjecture)
            if args.strict:
                raise KeyError(f"No label rows for {conjecture}")
            continue
        rel = relevant[conjecture]
        per_problem[conjecture] = {
            "AP": average_precision(ranking, rel),
            "RR": reciprocal_rank(ranking, rel),
            "relevant": len(rel),
            "ranked": len(ranking),
            **{f"Recall@{k}": recall_at_k(ranking, rel, k) for k in args.k},
        }

    if not per_problem:
        raise ValueError("No conjecture had both rankings and relevant labels")

    summary = {
        "problems": len(per_problem),
        "MAP": sum(v["AP"] for v in per_problem.values()) / len(per_problem),
        "MRR": sum(v["RR"] for v in per_problem.values()) / len(per_problem),
    }
    for k in args.k:
        summary[f"Recall@{k}"] = sum(
            v[f"Recall@{k}"] for v in per_problem.values()) / len(per_problem)
    summary["missing_label_problems"] = missing

    result = {
        "schema": "tw-gnn-ranking-evaluation-v1",
        "rankings": str(args.rankings),
        "rankings_sha256": sha256(args.rankings),
        "labels": str(args.labels),
        "labels_sha256": sha256(args.labels),
        "k": args.k,
        "summary": summary,
        "per_problem": per_problem,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
