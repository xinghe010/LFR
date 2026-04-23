#!/usr/bin/env python3
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Statements
import re as _re
import collections

def _extract_symbols(formula):
    return set(_re.findall(r'[a-z][a-z0-9_]*', formula))

def _build_tri_dict(fof_dict):

    symbol_count = collections.Counter()
    for symbols in fof_dict.values():
        for s in symbols:
            symbol_count[s] += 1

    tri_dict = collections.defaultdict(set)
    for name, symbols in fof_dict.items():
        if not symbols:
            continue
        min_sym = min(symbols, key=lambda s: symbol_count.get(s, 0))
        tri_dict[min_sym].add(name)
    return tri_dict

def generate_similarity2(statements, train_data):
    fof_dict = {}
    for name, formula in statements.statements.items():
        fof_dict[name] = _extract_symbols(formula)

    tri_dict = _build_tri_dict(fof_dict)

    simi_dict = {}
    conjs = set()
    prems = set()
    for sample in train_data:
        if isinstance(sample, dict):
            conjs.add(sample['conjecture'])
            prems.add(sample['premise'])
        else:
            conjs.add(sample[0])
            prems.add(sample[1])

    for conj in conjs:
        simi_dict[conj] = {}
        conj_symbols = fof_dict.get(conj, set())

        triggered = set()
        for s in conj_symbols:
            if s in tri_dict:
                triggered.update(tri_dict[s])

        triggered2 = set(triggered)
        for t in triggered:
            t_symbols = fof_dict.get(t, set())
            for s in t_symbols:
                if s in tri_dict:
                    triggered2.update(tri_dict[s])

        for prem in prems:
            simi_dict[conj][prem] = 1 if prem in triggered2 else 0

    return simi_dict

def generate_similarity4(statements, train_data, h=0.2):
    fof_dict = {}
    for name, formula in statements.statements.items():
        import re
        symbols = set()
        for token in re.findall(r'[a-z][a-z0-9_]*', formula):
            symbols.add(token)
        fof_dict[name] = symbols

    all_symbols = set()
    for s in fof_dict.values():
        all_symbols |= s

    symbol_count = collections.Counter()
    for s in fof_dict.values():
        for sym in s:
            symbol_count[sym] += 1

    P = len(fof_dict)
    symbol_weights = {s: 1 - c / (P + 1) for s, c in symbol_count.items()}

    conjs = set()
    prems = set()
    for sample in train_data:
        if isinstance(sample, dict):
            conjs.add(sample['conjecture'])
            prems.add(sample['premise'])
        else:
            conjs.add(sample[0])
            prems.add(sample[1])

    simi_dict = {}
    for conj in conjs:
        simi_dict[conj] = {}
        conj_symbols = fof_dict.get(conj, set())

        selected = set()
        for name, symbols in fof_dict.items():
            shared = conj_symbols & symbols
            if shared:
                relevance = sum(symbol_weights.get(s, 0) for s in shared) / max(len(shared), 1)
                if relevance >= h:
                    selected.add(name)

        for prem in prems:
            simi_dict[conj][prem] = 1 if prem in selected else 0

    return simi_dict

def main():
    statements_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/statements"
    train_data_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."

    stmts = Statements(statements_path)
    print(f"Statements: {len(stmts)}")

    if train_data_path:
        with open(train_data_path) as f:
            train_data = json.load(f)
        print(f"Training samples: {len(train_data)}")
    else:

        train_data = []
        for name in stmts.statements:
            train_data.append([name, name, 0])
        print(f"Using all {len(train_data)} formulas")

    print("Generating similarity2 (two-pass triggering)...")
    simi2 = generate_similarity2(stmts, train_data)
    path2 = os.path.join(output_dir, "similarity2.json")
    with open(path2, 'w') as f:
        json.dump(simi2, f)
    print(f"Saved: {path2} ({len(simi2)} conjectures)")

    print("Generating similarity4 (symbol weight)...")
    simi4 = generate_similarity4(stmts, train_data)
    path4 = os.path.join(output_dir, "similarity4.json")
    with open(path4, 'w') as f:
        json.dump(simi4, f)
    print(f"Saved: {path4} ({len(simi4)} conjectures)")

if __name__ == '__main__':
    main()
