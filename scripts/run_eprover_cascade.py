"""Run E-prover cascade over ranked premises and record proved problems.

This is the proof-runner used for the single-round ATP evaluation.  It expects
one TPTP ``.p`` problem per test conjecture and a ranking JSON produced by
``src/atp/select_premises_*.py``.  For each K in --k-values, only the first K
ranked axioms are kept; the conjecture and all selected axioms are written to a
temporary TPTP file and passed to E-prover.

The script is deliberately independent of PyTorch.  It can therefore run on a
CPU-only machine after rankings and problem files are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_tptp(text: str) -> list[str]:
    """Split TPTP text into top-level statements ending with a period."""
    text = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    statements, buf = [], []
    depth = 0
    quote = False
    escaped = False
    for index, char in enumerate(text):
        buf.append(char)
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                quote = False
            continue
        if char == "'":
            quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "." and depth == 0:
            nxt = text[index + 1:index + 2]
            if not nxt or nxt.isspace():
                statement = "".join(buf).strip()
                if statement:
                    statements.append(statement)
                buf = []
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def parse_statement(statement: str):
    match = re.match(r"\s*(fof|cnf)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,", statement, re.I | re.S)
    if match:
        return match.group(1).lower(), match.group(2).strip(), match.group(3).strip().lower()
    if re.match(r"\s*include\s*\(", statement, re.I):
        return "include", None, None
    return None, None, None


def build_problem(problem_path: Path, ranking: list[str], limit: int, output_path: Path):
    """Write a TPTP file containing the conjecture and selected axioms."""
    statements = split_tptp(problem_path.read_text(encoding="utf-8", errors="replace"))
    axioms, conjectures, includes = {}, [], []
    for statement in statements:
        kind, name, role = parse_statement(statement)
        if kind == "include":
            includes.append(statement)
        elif kind in {"fof", "cnf"} and role == "axiom":
            axioms[name] = statement
        elif kind in {"fof", "cnf"} and role == "conjecture":
            conjectures.append(statement)
    if not conjectures:
        raise ValueError(f"No conjecture found in {problem_path}")
    selected = [name for name in ranking[:limit] if name in axioms]
    missing = [name for name in ranking[:limit] if name not in axioms]
    with output_path.open("w", encoding="utf-8") as stream:
        for statement in includes:
            stream.write(statement + "\n")
        for name in selected:
            stream.write(axioms[name] + "\n")
        for statement in conjectures:
            stream.write(statement + "\n")
    return selected, missing


def find_problem(problem_dir: Path, name: str, pattern: str | None) -> Path:
    if pattern:
        candidate = problem_dir / pattern.format(problem=name)
    else:
        candidate = problem_dir / f"{name}.p"
    if candidate.is_file():
        return candidate
    matches = sorted(problem_dir.glob(f"*{name}*.p"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .p file for {name} under {problem_dir}")
    raise FileExistsError(f"Multiple .p files for {name}: {matches[:5]}")


def run_eprover(eprover: Path, problem: Path, cpu_limit: int, memory_limit: int,
                strategy: str, extra_args: list[str], timeout: int):
    cmd = [
        str(eprover),
        "--auto",
        f"--{strategy}" if strategy else "--auto",
        f"--cpu-limit={cpu_limit}",
        f"--memory-limit={memory_limit}",
        "-p",
        "--tstp-format",
        str(problem),
        *extra_args,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        proved = bool(re.search(
            r"Proof found|SZS status Theorem|SZS status Unsatisfiable|# Proof found",
            output, re.I))
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "proved": proved,
            "elapsed_sec": elapsed,
            "output": output,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + "\n" + \
                 ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        return {
            "command": cmd,
            "returncode": "timeout",
            "proved": False,
            "elapsed_sec": elapsed,
            "output": output,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--eprover", type=Path, required=True,
                        help="Path to the E-prover executable (3.2.5 recommended)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--cpu-limit", type=int, default=10,
                        help="CPU seconds per E-prover call")
    parser.add_argument("--memory-limit", type=int, default=10000,
                        help="Memory limit in MB per E-prover call")
    parser.add_argument("--strategy", default="satauto",
                        help="E-prover strategy flag, e.g. satauto or auto")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Additional E-prover argument; repeatable")
    parser.add_argument("--problem-pattern", default=None,
                        help="Optional pattern such as '{problem}__*.p'")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Directory for generated TPTP files and prover logs")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write selected TPTP files but do not call E-prover")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N problems (smoke test)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on a missing problem or missing ranked axiom")
    args = parser.parse_args()

    rankings = json.loads(args.rankings.read_text(encoding="utf-8"))
    if not isinstance(rankings, dict):
        raise ValueError("Rankings must be a JSON object: problem -> ordered premises")
    if args.limit:
        rankings = dict(list(rankings.items())[:args.limit])

    workdir = args.workdir or (args.output.parent / (args.output.stem + "_work"))
    workdir.mkdir(parents=True, exist_ok=True)
    timeout = args.cpu_limit + 30

    results = {}
    proved = 0
    for index, (problem_name, ranking) in enumerate(sorted(rankings.items()), 1):
        try:
            problem_path = find_problem(args.problem_dir, problem_name, args.problem_pattern)
        except (FileNotFoundError, FileExistsError) as exc:
            if args.strict:
                raise
            results[problem_name] = {"proved": False, "error": str(exc)}
            continue

        record = {"problem": problem_name, "problem_file": str(problem_path),
                  "ranking_length": len(ranking), "proved": False, "k": None,
                  "used_axioms": [], "attempts": []}
        for k in sorted(args.k_values):
            selected_path = workdir / f"{problem_name}.k{k}.p"
            try:
                selected, missing = build_problem(problem_path, ranking, k, selected_path)
            except Exception as exc:
                record["error"] = str(exc)
                break
            if missing and args.strict:
                raise KeyError(f"{problem_name}: ranked axioms absent from problem: {missing[:5]}")
            if args.dry_run:
                record["attempts"].append({"k": k, "selected": len(selected),
                                           "missing": missing, "proved": None})
                continue
            run = run_eprover(args.eprover, selected_path, args.cpu_limit,
                              args.memory_limit, args.strategy, args.extra_arg, timeout)
            output_path = workdir / f"{problem_name}.k{k}.out"
            output_path.write_text(run["output"], encoding="utf-8")
            used = [name for name in selected
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(",
                                 run["output"])]
            record["attempts"].append({
                "k": k, "selected": len(selected), "missing": missing,
                "proved": run["proved"], "returncode": run["returncode"],
                "elapsed_sec": run["elapsed_sec"],
                "output": str(output_path), "used_axioms": used,
            })
            if run["proved"]:
                record.update({"proved": True, "k": k, "used_axioms": used})
                break
        results[problem_name] = record
        if record["proved"]:
            proved += 1
        print(f"[{index}/{len(rankings)}] {problem_name}: "
              f"{'proved' if record['proved'] else 'not proved'}")

    summary = {
        "schema": "tw-gnn-atp-proof-run-v1",
        "rankings": str(args.rankings),
        "rankings_sha256": sha256(args.rankings),
        "problems": len(results),
        "proved": proved,
        "proof_rate": proved / len(results) if results else 0.0,
        "k_values": sorted(args.k_values),
        "cpu_limit": args.cpu_limit,
        "memory_limit": args.memory_limit,
        "strategy": args.strategy,
        "eprover": str(args.eprover),
        "eprover_sha256": sha256(args.eprover) if args.eprover.is_file() else None,
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {args.output}")

    if not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
