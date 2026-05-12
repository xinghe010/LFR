# TW-GNN-LFR: Supplementary Code

This repository accompanies the paper

> **Neuro-symbolic premise selection with logic-augmented term-walk graph neural networks.**

It provides the reference implementation of **TW-GNN-LFR** (Term-Walk Graph Neural Network with Logical Features and Rules) together with the scripts required to reproduce the premise-selection accuracies and the end-to-end E-prover results reported in the paper.

---

## 1. Overview

TW-GNN-LFR is a neuro-symbolic model for the premise-selection task in large-theory automated theorem proving (ATP). It augments a term-walk graph encoder with two ingredients drawn from the underlying first-order logic:

1. **Logical Connective Enhancement (LCE).** Connective-specific transformations are applied during message passing for conjunction, disjunction, negation, implication, biconditional, and the universal and existential quantifiers.
2. **Soft symbolic rules in the classifier.** Two rules are injected into logit space: a *rare-symbol triggering* rule built on bipartite premise–conjecture propagation, and a *rarity-weighted symbolic similarity* rule that acts as additive evidence in log-odds space.

On MPTP2078 the model attains 88.67 % (CNF: 85.44 %); the attention-based variant ASTW-GNN-LFR reaches 88.83 % / 86.19 %. End-to-end with E-prover, 222 of 280 problems are solved in a single pass and 242 after ten feedback rounds.

---

## 2. Repository layout

```
supplementary material_LFR/
├── ATP_experiment/                  End-to-end evaluation with E-prover
│   ├── code/                        Core model, dataset, parser, similarity, sInE
│   ├── dataset/D0/                  Theorem-prover-ready evaluation set
│   └── scripts/                     run.py, feedback_loop.py, evaluate_*.py
└── premise_selection/               Standalone classification benchmark
    ├── MPTP/                        Trainable on the original MPTP2078 set
    └── CNF/                         Same pipeline on the CNF-clausified variant
```

The two `premise_selection/` subdirectories share a common module layout (`model.py`, `graph.py`, `dataset.py`, `trainer.py`, `eval.py`, …) and differ only in their data loaders and connective vocabularies.

---

## 3. Requirements

A working CUDA-capable installation of PyTorch is assumed; CPU execution is supported but slow. Install the Python dependencies with

```bash
pip install -r requirements.txt
```

Pinned versions are intentionally avoided — TW-GNN-LFR has been exercised against PyTorch ≥ 1.10 and PyTorch Geometric ≥ 2.0. For end-to-end ATP evaluation, an external installation of **E-prover** (≥ 2.6) must be available on the system path; the wrapper scripts invoke it as a subprocess.

---

## 4. Datasets

The supplementary archive ships the pre-processed graph datasets used for the paper's main results:

* `premise_selection/MPTP/dataset/` — MPTP2078 formulas as DAGs.
* `premise_selection/CNF/data/` — CNF-clausified counterpart.
* `ATP_experiment/dataset/D0/` — held-out problems used in the feedback-loop experiment, together with the provable-problem manifest (`test_problems_provable.json`).

No additional download is required to reproduce the figures and tables of the paper.

---

## 5. Training a TW-GNN-LFR model

From either premise-selection sub-folder, training and evaluation are launched through `eval.py`:

```bash
cd premise_selection/MPTP
python eval.py --exp_name lfr_mptp --epochs 50 --batch_size 32 --lr 1e-3
```

The CNF variant is trained identically:

```bash
cd premise_selection/CNF
python eval.py --exp_name lfr_cnf --epochs 50 --batch_size 32 --lr 1e-3
```

Logs and checkpoints are written next to the script. The default hyper-parameters reproduce the numbers reported in Table 1 of the paper.

---

## 6. End-to-end evaluation with E-prover

The closed-loop experiment is driven from `ATP_experiment/scripts/`:

```bash
cd ATP_experiment/scripts
python run.py            --mode train          # (re-)train the model used by the prover wrapper
python feedback_loop.py  --num_rounds 10       # iterative premise refinement
python evaluate_simple.py                      # single-pass ranking baseline
python evaluate_sine.py                        # sInE comparison
python evaluate_cascade.py                     # cascade retrieval
```

`feedback_loop.py` produces the 222 / 242 numbers cited in the abstract; the `evaluate_*` family supplies the auxiliary baselines used in the comparison tables.

---

## 7. Reproducing the paper

| Experiment                          | Entry point                                  | Expected result    |
|-------------------------------------|----------------------------------------------|--------------------|
| Premise-selection accuracy (MPTP)   | `premise_selection/MPTP/eval.py`             | 88.67 %            |
| Premise-selection accuracy (CNF)    | `premise_selection/CNF/eval.py`              | 85.44 %            |
| End-to-end, single-pass             | `ATP_experiment/scripts/feedback_loop.py`    | 222 / 280          |
| End-to-end, 10 feedback rounds      | `ATP_experiment/scripts/feedback_loop.py`    | 242 / 280          |

Minor variation (≤ 0.3 pp) is expected across hardware due to non-determinism in scatter operations.

---

## 8. Citation

If you use this code, please cite the accompanying paper. A BibTeX entry will be added once the work is officially published.
