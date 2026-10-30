# TW-GNN-LFR

Code and cached data for **Neuro-symbolic premise selection with
logic-augmented term-walk graph neural networks**.

The repository contains:

- the MPTP and CNF article-isolated splits used in the paper;
- vocabulary, symbolic caches, and rule caches for those splits;
- the TW-GNN, LCE, fusion-control, and symbolic-scoring implementations;
- entry points for the paper's classification experiments;
- a ranking evaluator for Table 6 (MAP/MRR/Recall@K);
- the Table 6 MPTP checkpoints and configs;
- the 280 ATP test problems, chainy candidate pairs, and candidate manifest;
- the ATP selector checkpoint (`data/atp/checkpoints/full_seed24_round10`) and
  its config/result files;
- an E-prover cascade runner and a feedback-loop orchestrator for Table 7;
- the `t58_tops_1` case-study rankings;
- third-party license notices.

It does **not** contain the external graph-baseline implementations.  Those
are third-party artifacts and are not vendored.

## Environment

- Linux or macOS.  The dataset loaders use `fcntl`; native Windows is not
  supported (use WSL if needed).
- Python 3.11
- PyTorch 2.4.1
- PyTorch Geometric 2.6.0
- CUDA 12.1 (CPU runs are possible with `--device cpu`)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`torch-scatter` may need the matching PyG wheel index:

```bash
pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

### E-prover 3.2.5

The proof runner calls an external E-prover executable.  Build or install
E-prover 3.2.5 and pass its path with `--eprover`:

```bash
git clone https://github.com/eprover/eprover.git
cd eprover
git checkout v3.2.5
./configure
make
PROVER/eprover
```

Typical path on a shared machine:

```text
/opt/eprover/PROVER/eprover
```

The code does not bundle E-prover; E-prover's own license applies.  See
`third_party/eprover/COPYING`.

## Data

The data directory is distributed as `data.zip` to keep the repository small.
Unpack it once from the repository root:

```bash
python scripts/unpack_data.py
```

```text
data/mptp/    train/valid/test pairs, statements, vocabulary, symbolic cache
data/cnf/     the same split in CNF form
data/atp/     ATP train/valid pairs, statements, symbolic cache, 280 test IDs
```

The MPTP/CNF splits are article-isolated: 54,928 / 7,204 / 6,892 pairs.
The ATP split has 44,258 / 4,918 pairs and 280 test problem IDs.

`data/atp/test/problems` contains the 280 TPTP problems.
`data/atp/candidate_pairs.jsonl` and `chainy_candidate_pairs.jsonl.gz` contain
the chainy candidate pairs; `candidate_manifest.json` records their counts and
hashes; `unlabeled_union_pairs.jsonl` is the source used for symbolic caches.
The ATP selector checkpoint is included at
`data/atp/checkpoints/full_seed24_round10`; `scripts/train_atp_selector.py` can
regenerate it and its config/result files.

## Paper entry points

The classification entry points previously under `scripts/legacy/` are now
under `scripts/paper_entrypoints/`:

```text
scripts/paper_entrypoints/mptp/ablation_fusion_eval.py
scripts/paper_entrypoints/cnf/ablation_fusion_eval.py
```

They are the entry points used for the paper's classification/control tables.
Run MPTP and CNF in separate processes to avoid collisions between identically
named modules.

### Classification (default: d=128, L=1, batch=32, 30 epochs, seed=24)

```bash
python scripts/paper_entrypoints/mptp/ablation_fusion_eval.py \
  --root_dir data/mptp \
  --model_save outputs/runs \
  --exp_name mptp_lfr \
  --node_out_channels 128 --layers 1 --batch_size 32 \
  --epochs 30 --seed 24

python scripts/paper_entrypoints/cnf/ablation_fusion_eval.py \
  --root_dir data/cnf \
  --model_save outputs/runs \
  --exp_name cnf_lfr \
  --node_out_channels 128 --layers 1 --batch_size 32 \
  --epochs 30 --seed 24
```

The same entry point accepts the control settings used in the paper, including
`--encoder_variant`, `--fusion_mode`, `--symbolic_baseline`,
`--use_lce_*`, `--rho_t`, and `--rho_s`.  `configs/example.json` records the
default shape; it is documentation, not read automatically.

The external graph baselines in Table 3 (GCN, GAT, GIN, GraphSAGE, Graph
Transformer, PC-GCN, ChebNet, SGC, ATW-GNN, ASTW-GNN) are not vendored.  The
TW-GNN variants and controls are implemented here.

## ATP selector training

`scripts/train_atp_selector.py` trains the initial ATP selector, selects the
best validation-ACC epoch, and writes `best_valid_acc.pt`,
`model_config.json`, and `results.json` (the provenance file consumed by the
ranking scripts).  Checkpoints and provenance are generated locally; they are
not committed to this repository.

```bash
python scripts/train_atp_selector.py \
  --root_dir data/atp \
  --output_dir outputs/atp_selector \
  --component full \
  --node_out_channels 128 --layers 1 --batch_size 32 \
  --epochs 30 --seed 24
```

Pass `--candidate_manifest`, `--rule_metadata`, and `--symbolic_metadata`
when those files exist; the continuous ranking script checks their hashes in
`results.json`.  The ATP symbolic cache can be used as provided; its source
pair list is `data/atp/unlabeled_union_pairs.jsonl`.

The ranking script then uses the generated files:

```bash
python src/atp/select_premises_continuous_v2.py \
  --model outputs/atp_selector/best_valid_acc.pt \
  --model_config outputs/atp_selector/model_config.json \
  --training_result outputs/atp_selector/results.json \
  --component full \
  --statements data/atp/statements \
  --node_dict data/atp/node_dict.pkl \
  --node_attr_dict data/atp/node_attr_dict.pkl \
  --edge_attr_dict data/atp/edge_attr_dict.pkl \
  --chainy_dir data/atp/test/problems \
  --test_problems data/atp/test_problems_provable.json \
  --candidate_manifest data/atp/candidate_manifest.json \
  --symbolic_cache data/atp/symbolic_baselines.jsonl \
  --output outputs/atp_ranking.json
```

## Table 6: MAP/MRR/Recall@K

`select_premises_continuous_v2.py` produces a ranking JSON:

```json
{"t58_tops_1": ["t3_subset", "t5_boole", "..."]}
```

Evaluate it with the article-isolated test labels:

```bash
python scripts/evaluate_ranking.py \
  --rankings outputs/ranking_full.json \
  --labels data/mptp/test/raw/test.json \
  --k 128 512 \
  --output outputs/table6_full.json
```

The label file is JSONL with rows `[conjecture, premise, label]`; only
`label == 1` is relevant.  The script reports MAP, MRR, and Recall@K and keeps
per-problem values in the output JSON.

## Table 7: proof runner and feedback loop

Single-round cascade:

```bash
python scripts/run_eprover_cascade.py \
  --rankings outputs/ranking_full.json \
  --problem-dir /path/to/atp/problems \
  --eprover /opt/eprover/PROVER/eprover \
  --output outputs/atp_single_round.json \
  --k-values 32 64 128 256 512 1024 \
  --cpu-limit 10 --memory-limit 10000 --strategy satauto
```

The runner keeps only the first K ranked axioms, keeps the conjecture, writes a
temporary TPTP file, calls E-prover, and records the proved status and used
axioms.  It does not modify the original `.p` files.

Ten-round feedback orchestration:

```bash
python scripts/run_feedback_loop.py \
  --rankings outputs/round0_rankings.json \
  --problem-dir /path/to/atp/problems \
  --eprover /opt/eprover/PROVER/eprover \
  --output-dir outputs/feedback \
  --rounds 10 \
  --update-command "python /path/to/update_selector.py --feedback {feedback} --output {model_dir}" \
  --rank-command "python src/atp/select_premises_lfr.py --model {model_dir}/best.pt --model_config {model_dir}/model_config.json --training_result {model_dir}/results.json --component full --statements data/atp/statements --node_dict data/atp/node_dict.pkl --node_attr_dict data/atp/node_attr_dict.pkl --edge_attr_dict data/atp/edge_attr_dict.pkl --chainy_dir data/atp/test/problems --test_problems data/atp/test_problems_provable.json --candidate_manifest data/atp/candidate_manifest.json --output {rankings}"
```

The update/rank commands are templates: checkpoints are not distributed, so
you must supply a selector-training command and a ranking command.  The loop
writes cumulative coverage to `outputs/feedback/summary.json`.

## Licenses

- Original code: MIT (`LICENSE`).
- MPTP2078/Mizar-derived data: GPL-3.0-or-later or CC-BY-SA-3.0-or-later,
  subject to `COPYING.interpretation`; see `third_party/mptp2078/`.
- ATPBoost: MIT; see `third_party/atpboost/LICENSE`.
- E-prover: GPL-2.0-or-later / LGPL-2.1-or-later; see
  `third_party/eprover/COPYING`.

See `THIRD_PARTY_NOTICES.md` before redistributing `data/`.

## Citation

See `CITATION.cff`.
