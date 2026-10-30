
import os, sys, re, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formula_parser import fof_formula_transformer
from graph import Graph
from utils import Statements

VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
CONNECTIVE_PATTERN = {"!", "?", "|", "&", "=>", "<=>", "~"}

def extract_attrs_from_graph(G):
    node_attrs = set()
    edge_attrs = set()

    for node in G:
        if node.name in {"!", "?"}:
            node_attrs.add(node.name)
            for child in node.children:
                if re.match(VARIABLE_PATTERN, child.name):
                    edge_attrs.add(f"{node.name}_var")
                    edge_attrs.add(f"var_{node.name}")
                elif re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.add(f"{node.name}_pred")
                    edge_attrs.add(f"pred_{node.name}")
                elif child.name == "=":
                    edge_attrs.add(f"{node.name}_=")
                    edge_attrs.add(f"=_{node.name}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.add(f"{node.name}_{child.name}")
                    edge_attrs.add(f"{child.name}_{node.name}")

        if node.name in {"&", "|", "<=>", "~"}:
            node_attrs.add(node.name)
            for child in node.children:
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.add(f"{node.name}_pred")
                    edge_attrs.add(f"pred_{node.name}")
                elif child.name == "=":
                    edge_attrs.add(f"{node.name}_=")
                    edge_attrs.add(f"=_{node.name}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.add(f"{node.name}_{child.name}")
                    edge_attrs.add(f"{child.name}_{node.name}")

        if node.name == "=>":
            node_attrs.add(node.name)
            for idx, child in enumerate(node.children, 1):
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.add(f"=>_pred_{idx}")
                    edge_attrs.add(f"pred_=>_{idx}")
                elif child.name == "=":
                    edge_attrs.add(f"=>_=_{idx}")
                    edge_attrs.add(f"=_=>_{idx}")
                elif child.name in CONNECTIVE_PATTERN:
                    edge_attrs.add(f"=>_{child.name}_{idx}")
                    edge_attrs.add(f"{child.name}_=>_{idx}")

        if node.name == "=":
            node_attrs.add("=")
            for child in node.children:
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.add("=_func")
                    edge_attrs.add("func_=")
                elif re.match(VARIABLE_PATTERN, child.name):
                    edge_attrs.add("=_var")
                    edge_attrs.add("var_=")

        if re.match(FUNCTOR_PATTERN, node.name):
            parents_tokens = [p.name for p in node.parents]
            is_pred = len(parents_tokens) == 0 or parents_tokens[0] in CONNECTIVE_PATTERN
            role = "pred" if is_pred else "func"
            node_attrs.add(role)
            for idx, child in enumerate(node.children, 1):
                if re.match(FUNCTOR_PATTERN, child.name):
                    edge_attrs.add(f"{role}_func_{idx}")
                    edge_attrs.add(f"func_{role}_{idx}")
                elif re.match(VARIABLE_PATTERN, child.name):
                    edge_attrs.add(f"{role}_var_{idx}")
                    edge_attrs.add(f"var_{role}_{idx}")

        if node.name == "$true":
            node_attrs.add("$true")
        if re.match(VARIABLE_PATTERN, node.name):
            node_attrs.add("var")

    return node_attrs, edge_attrs

def main():
    statements_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/statements"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Loading statements from {statements_path}")
    stmts = Statements(statements_path)
    print(f"Total formulas: {len(stmts)}")

    all_node_attrs = set()
    all_edge_attrs = set()
    errors = 0

    for i, (name, formula) in enumerate(stmts.statements.items()):
        try:
            parsed = fof_formula_transformer(formula)
            graph = Graph(parsed, rename=True)
            na, ea = extract_attrs_from_graph(graph)
            all_node_attrs |= na
            all_edge_attrs |= ea
        except Exception as e:
            errors += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(stmts)}, node_attrs={len(all_node_attrs)}, edge_attrs={len(all_edge_attrs)}")

    print(f"\nDone. Errors: {errors}")
    print(f"Node attr types: {sorted(all_node_attrs)}")
    print(f"Edge attr types ({len(all_edge_attrs)}): {sorted(all_edge_attrs)[:20]}...")

    node_attr_dict = {attr: idx for idx, attr in enumerate(sorted(all_node_attrs))}
    edge_attr_dict = {attr: idx for idx, attr in enumerate(sorted(all_edge_attrs))}

    na_path = os.path.join(output_dir, "node_attr_dict.pkl")
    ea_path = os.path.join(output_dir, "edge_attr_dict.pkl")

    with open(na_path, 'wb') as f:
        pickle.dump(node_attr_dict, f)
    with open(ea_path, 'wb') as f:
        pickle.dump(edge_attr_dict, f)

    print(f"\nSaved: {na_path} ({len(node_attr_dict)} types)")
    print(f"Saved: {ea_path} ({len(edge_attr_dict)} types)")

if __name__ == '__main__':
    main()
