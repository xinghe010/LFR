"""Orchestrate the ten-round ATP feedback loop.

Round 0 runs the proof runner on the initial rankings.  Each following round
uses the proof dependencies found so far as feedback, calls an external
selector-update command, re-ranks the same 280 problems, and runs the proof
runner again.  Cumulative coverage is the union of problems proved in any round.

The selector training and ranking commands are external because checkpoints are
not distributed with this repository.  Templates receive:

  {round}       current feedback round (1..N)
  {feedback}    JSONL file with (conjecture, axiom) feedback pairs
  {model_dir}   directory holding the updated selector
  {rankings}    output ranking JSON expected from the rank command
  {problem_dir} directory containing the 280 TPTP problems
  {output_dir}  root of the feedback run

Example:
  python scripts/run_feedback_loop.py \
    --rankings round0/rankings.json --problem-dir /data/atp/problems \
    --eprover /opt/eprover/eprover --rounds 10 \
    --update-command "python /path/to/update_selector.py --feedback {feedback} --output {model_dir}" \
    --rank-command "python src/atp/select_premises_lfr.py --model {model_dir}/best.pt ... --output {rankings}"
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def run_command(command: str, shell: bool) -> None:
    print(f"$ {command}")
    completed = subprocess.run(
        command if shell else shlex.split(command),
        shell=shell, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {command}")


def load_proved(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True,
                        help="Initial round-0 rankings JSON")
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--eprover", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of feedback rounds after round 0")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--cpu-limit", type=int, default=10)
    parser.add_argument("--memory-limit", type=int, default=10000)
    parser.add_argument("--strategy", default="satauto")
    parser.add_argument("--proof-runner", type=Path,
                        default=Path(__file__).with_name("run_eprover_cascade.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--update-command", default=None,
                        help="Template for selector update; required when --rounds > 0")
    parser.add_argument("--rank-command", default=None,
                        help="Template for re-ranking; required when --rounds > 0")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--shell", action="store_true",
                        help="Run commands through the shell instead of shlex.split")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N problems (smoke test)")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.rounds > 0 and (not args.update_command or not args.rank_command):
        raise SystemExit("--update-command and --rank-command are required when --rounds > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir or (args.output_dir / "model")
    model_dir.mkdir(parents=True, exist_ok=True)

    def run_proofs(round_index: int, rankings: Path) -> Path:
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        output = round_dir / "proofs.json"
        cmd = [
            args.python, str(args.proof_runner),
            "--rankings", str(rankings),
            "--problem-dir", str(args.problem_dir),
            "--eprover", str(args.eprover),
            "--output", str(output),
            "--k-values", *[str(k) for k in args.k_values],
            "--cpu-limit", str(args.cpu_limit),
            "--memory-limit", str(args.memory_limit),
            "--strategy", args.strategy,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.strict:
            cmd.append("--strict")
        subprocess.run(cmd, check=True)
        return output

    initial_rankings = args.rankings
    proof_files = [run_proofs(0, initial_rankings)]
    feedback: set[tuple[str, str]] = set()
    for record in load_proved(proof_files[-1]).values():
        if record.get("proved"):
            for axiom in record.get("used_axioms", []):
                feedback.add((record["problem"], axiom))

    for round_index in range(1, args.rounds + 1):
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        feedback_path = round_dir / "feedback.jsonl"
        with feedback_path.open("w", encoding="utf-8") as stream:
            for problem, axiom in sorted(feedback):
                stream.write(json.dumps({"conjecture": problem, "axiom": axiom}) + "\n")

        values = {
            "round": round_index,
            "feedback": feedback_path,
            "model_dir": model_dir,
            "rankings": round_dir / "rankings.json",
            "problem_dir": args.problem_dir,
            "output_dir": args.output_dir,
        }
        run_command(args.update_command.format(**values), args.shell)
        run_command(args.rank_command.format(**values), args.shell)
        if not values["rankings"].is_file():
            raise FileNotFoundError(f"Rank command did not produce {values['rankings']}")
        proof_files.append(run_proofs(round_index, values["rankings"]))

        for record in load_proved(proof_files[-1]).values():
            if record.get("proved"):
                for axiom in record.get("used_axioms", []):
                    feedback.add((record["problem"], axiom))

    rounds = []
    cumulative: set[str] = set()
    for index, proof_file in enumerate(proof_files):
        records = load_proved(proof_file)
        proved = {name for name, record in records.items() if record.get("proved")}
        cumulative |= proved
        rounds.append({
            "round": index,
            "problems": len(records),
            "proved": len(proved),
            "proof_rate": len(proved) / len(records) if records else 0.0,
            "cumulative": len(cumulative),
            "cumulative_rate": len(cumulative) / len(records) if records else 0.0,
            "proofs": str(proof_file),
        })

    summary = {
        "schema": "tw-gnn-atp-feedback-loop-v1",
        "rounds": rounds,
        "cumulative_proved": sorted(cumulative),
        "feedback_pairs": len(feedback),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
