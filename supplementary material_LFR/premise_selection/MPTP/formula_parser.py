from lark import Lark, Transformer

fof_parser = Lark(r"""
    annotated_formula: "fof(" name "," formula_role ","  fof_formula ")."

    name: NAME
    NAME: UPPER_LOW_ALPHA_NUMERIC+

    formula_role: FORMULA_ROLE
    FORMULA_ROLE: "axiom"

    ?fof_formula: unitary_formula | binary_formula

    ?unitary_formula: literal | type_bool | quantified_formula | negative "(" fof_formula ")" | constant

    ?binary_formula: assoc_formula | non_assoc_formula

    non_assoc_formula: "(" fof_formula binary_connective fof_formula ")"
    binary_connective: BINARY_CONNECTIVE

    assoc_formula: "(" fof_formula assoc_connective fof_formula ")"
    assoc_connective: ASSOC_CONNECTIVE

    quantified_formula: "(" quantifier variable_list fof_formula ")"
    quantifier: QUANTIFIER
    variable_list: "[" variable ( "," variable )* "]" ":"

    ?literal: atom | negative "(" atom ")"
    negative: NEGATIVE

    atom: predicate "(" term_argument ")" | term equal term
    predicate: PREDICATE
    equal: EQUAL

    ?term: functional_term | variable | constant
    functional_term: functor "(" term_argument ")"
    term_argument: term ("," term)*
    functor: FUNCTOR
    variable: VARIABLE
    constant: CONSTANT

    PREDICATE: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    FUNCTOR: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    VARIABLE: UPPER_ALPHA UPPER_ALPHA_NUMERIC*
    CONSTANT: NUMERIC+ | LOWER_ALPHA LOW_ALPHA_NUMERIC*

    ?type_bool: type_true | type_false
    type_true: TYPE_TRUE
    type_false: TYPE_FALSE
    TYPE_TRUE: "$true"
    TYPE_FALSE: "$false"

    EQUAL: "="
    QUANTIFIER: "!" | "?"
    NEGATIVE: "~"
    BINARY_CONNECTIVE: "<=>" | "=>"
    ASSOC_CONNECTIVE : "&" | "|"

    LOW_ALPHA_NUMERIC : LOWER_ALPHA | NUMERIC | "_"
    UPPER_ALPHA_NUMERIC: UPPER_ALPHA | NUMERIC | "_"
    UPPER_LOW_ALPHA_NUMERIC : UPPER_ALPHA | LOWER_ALPHA | NUMERIC | "_"
    LOWER_ALPHA : "a" .. "z"
    UPPER_ALPHA : "A" .. "Z"
    NUMERIC : "0" .. "9" 
    %ignore " "
    """, start='annotated_formula')

class Transform(Transformer):
    annotated_formula = lambda self, a: a[2]
    name = lambda self, a: a[0][:]
    formula_role = lambda self, a: a[0][:]
    fof_formula = lambda self, a: a
    unitary_formula = lambda self, a: a
    binary_formula = lambda self, a: a
    assoc_formula = lambda self, a: a
    non_assoc_formula = lambda self, a: a
    quantified_formula = lambda self, a: a
    literal = lambda self, a: a
    atom = lambda self, a: a
    term = lambda self, a: a
    term_argument = lambda self, a: a
    functional_term = lambda self, a: a
    variable_list = lambda self, a: a
    type_bool = lambda self, a: a
    constant = lambda self, a: a[0][:]
    variable = lambda self, a: a[0][:]
    predicate = lambda self, a: a[0][:]
    functor = lambda self, a: a[0][:]
    quantifier = lambda self, a: a[0][:]
    negative = lambda self, a: a[0][:]
    binary_connective = lambda self, a: a[0][:]
    assoc_connective = lambda self, a: a[0][:]
    equal = lambda self, a: a[0][:]
    type_true = lambda self, a: a[0][:]
    type_false = lambda self, a: a[0][:]

def fof_formula_transformer(formula):
    fomula_pharse = fof_parser.parse(formula)
    formula_tree = Transform().transform(fomula_pharse)
    return formula_tree

def traversal_demo(tree, list_1):

    for a in tree:
        if isinstance(a, list):
            traversal_demo(a, list_1)
        else:

            list_1.append(a)

def retain_scope(a):
    if isinstance(a, list):
        while a[0] == '!' or a[0] == '?':
            a = a[2]

    if not is_form(a):
        list_1 = []
        list_1.append(a)
        return list_1
    return a

def remove_negation(a):
    if isinstance(a, list):
        while a[0] == '~':
            a = a[1]

    if not is_form(a):
        list_1 = []
        list_1.append(a)
        return list_1
    return a

def remove_log_connectives(a):
    if isinstance(a, list) and len(a) > 1:
        while a[1] == '&' or a[1] == '|' or a[1] == '=>' or a[1] == '<=>':
            a = a[0::2]
    return a

def is_form(a):
    if len(a) <2 :
        return False
    if a[0] == '!' or a[0] == '?' or a[0] =='~':
        return True
    if a[1] == '&' or a[1] =='|' or a[1] == '=>' or a[1] == '<=>':
        return True
    return False

def remove_logical(a):
    a = retain_scope(a)
    a = remove_negation(a)
    a = remove_log_connectives(a)
    return a

def get_atomic_formula(tree):
    temp1 = []
    temp2 = []
    while len(tree) != 0:
        form = is_form(tree)

        while form:
            tree = remove_logical(tree)
            form = is_form(tree)

        for a in tree:
            if not is_form(a):
                temp1.append(a)
            else:
                temp2.append(a)

        if len(temp2) != 0:
            tree = temp2[0]
            del temp2[0]
        else:
            tree = []
    return temp1

def similarity_form(conclusion, premise):

    conclusion = fof_formula_transformer(conclusion)
    premise = fof_formula_transformer(premise)

    conclusion = get_atomic_formula(conclusion)
    premise = get_atomic_formula(premise)

    print(conclusion)
    for a in conclusion:
        print(a)

    print(premise)
    for a in premise:
        print(a)

    sum = 0
    for i in range(len(conclusion)):
        value = -1
        for j in range(len(premise)):
            value = max(similarity_atomic_formula(conclusion[i], premise[j]), value)
        sum += value

    sum = sum / len(conclusion)
    return sum

def similarity_atomic_formula(conclusion, premise):

    if len(conclusion) != len(premise):
        return 0.5
    elif len(premise) == 3:
        return 1
    elif len(premise) == 2:
        if premise[0] != conclusion[0]:
            return 0
        else:

            sum = 0
            for i in range(len(premise[1])):
                sum += similarity_term(conclusion[1][i], premise[1][i])
            sum = sum/len(premise[1])
            return sum

def get_term_category(term):
    if isinstance(term, list):
        return 2
    elif term[0].isupper():
        return 1
    else:
        return 0

def similarity_constant_variable(conclusion, premise):
    category_1 = get_term_category(conclusion)
    category_2 = get_term_category(premise)
    if category_1 == 0 and category_2 == 0:
        if conclusion == premise:
            return 1
        else:
            return 0
    elif category_1 == 1 or category_2 == 1:
        return 1
    else:
        return 0

def similarity_term(conclusion, premise):
    category_1 = get_term_category(conclusion)
    category_2 = get_term_category(premise)
    if category_1 == 0 and category_2 == 0:
        if conclusion == premise:
            return 1
        else:
            return 0
    elif category_1 == 1 or category_2 == 1:
        return 1
    elif category_1 == 2 and category_2 == 2:
        if conclusion[0] != premise[0]:
            return 0
        else:
            sum = 0
            for i in range(len(premise[1])):
                sum += similarity_constant_variable(conclusion[1][i], premise[1][i])
            sum = sum/len(premise[1])
            return sum
    else:
        return 0

if __name__ == "__main__":
    fof = "fof(t4_catalan1, axiom,  (! [A] :  (v7_ordinal1(A) =>  ( ~ (r1_xxreal_0(A, 1))  =>" \
          " k2_nat_1(k7_nat_d(A, 2), 1)=k7_nat_d(A, 1)) ) ) )."

    conclusion = "fof(l1_analort, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v13_algstr_0(A) &  (v2_rlvect_1(A) &  (v3_rlvect_1(A) &  (v4_rlvect_1(A) &  (v5_rlvect_1(A) &  (v6_rlvect_1(A) &  (v7_rlvect_1(A) &  (v8_rlvect_1(A) & l1_rlvect_1(A)) ) ) ) ) ) ) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(A)) =>  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (! [E] :  (m1_subset_1(E, u1_struct_0(A)) =>  (! [F] :  (m1_subset_1(F, k1_numbers) =>  (! [G] :  (m1_subset_1(G, k1_numbers) =>  (! [H] :  (m1_subset_1(H, k1_numbers) =>  (! [I] :  (m1_subset_1(I, k1_numbers) =>  ( (B=k3_rlvect_1(A, k1_rlvect_1(A, C, F), k1_rlvect_1(A, D, G)) & E=k3_rlvect_1(A, k1_rlvect_1(A, C, H), k1_rlvect_1(A, D, I)))  =>  (k3_rlvect_1(A, B, E)=k3_rlvect_1(A, k1_rlvect_1(A, C, k7_real_1(F, H)), k1_rlvect_1(A, D, k7_real_1(G, I))) & k5_algstr_0(A, B, E)=k3_rlvect_1(A, k1_rlvect_1(A, C, k9_real_1(F, H)), " \
                 "k1_rlvect_1(A, D, k9_real_1(G, I)))) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )."

    premise = "fof(l43_geomtrap, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v13_algstr_0(A) &  (v2_rlvect_1(A) &  (v3_rlvect_1(A) &  (v4_rlvect_1(A) &  (v5_rlvect_1(A) &  (v6_rlvect_1(A) &  (v7_rlvect_1(A) &  (v8_rlvect_1(A) & l1_rlvect_1(A)) ) ) ) ) ) ) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(A)) =>  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (! [E] :  (m1_subset_1(E, u1_struct_0(A)) =>  (! [F] :  (m1_subset_1(F, k1_numbers) =>  (! [G] :  (m1_subset_1(G, k1_numbers) =>  (! [H] :  (m1_subset_1(H, k1_numbers) =>  (! [I] :  (m1_subset_1(I, k1_numbers) =>  ( (B=k3_rlvect_1(A, k1_rlvect_1(A, C, F), k1_rlvect_1(A, D, G)) & E=k3_rlvect_1(A, k1_rlvect_1(A, C, H), k1_rlvect_1(A, D, I)))  =>  (k3_rlvect_1(A, B, E)=k3_rlvect_1(A, k1_rlvect_1(A, C, k7_real_1(F, H)), k1_rlvect_1(A, D, k7_real_1(G, I))) & k5_algstr_0(A, B, E)=k3_rlvect_1(A, k1_rlvect_1(A, C, k9_real_1(F, H)), " \
              "k1_rlvect_1(A, D, k9_real_1(G, I)))) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )."

    conclusion = fof_formula_transformer(conclusion)
    premise = fof_formula_transformer(premise)

    conclusion = get_atomic_formula(conclusion)
    premise = get_atomic_formula(premise)

    list_1 = []
    traversal_demo(conclusion, list_1)
    print(list_1)

    list_2 = []
    traversal_demo(premise, list_2)
    print(list_2)

    for i in range(len(list_1)):
        if list_1[i] != list_2[i]:
            print("false")
            print(list_1[i])
            print(list_2[i])
    print("true")
