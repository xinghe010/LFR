from formula_parser import fof_formula_transformer, traversal_demo
from utils import read_file

import torch
import numpy
import sInE
import math

def get_dict_weigtht(dict_fof, w0=1.0):

    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]


    occ_symbol = dict()
    for symbol in set_symbol:
        count = 0
        for premise in dict_fof:
            if symbol in dict_fof[premise]:
                count += 1
        occ_symbol[symbol] = count

    dict_weight = dict()
    for symbol in occ_symbol:
        occ = float(occ_symbol[symbol])
        softplus_occ = occ + math.log1p(math.exp(-occ))
        w = 1.0 + float(w0) / softplus_occ
        dict_weight[symbol] = w

    return dict_weight

def premise_select(h, conclusion, dict_fof):

    name = sInE.get_formula_name(conclusion)

    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    print(name)

    print(symbol_conclusion)

    w_dict = get_dict_weigtht(dict_fof)

    res_set = set()
    premise_set = set()
    for premise in dict_fof:
        up = 0
        down = 0
        for formula in dict_fof[premise]:
            down += w_dict[formula]
            if formula in symbol_conclusion:
                up += w_dict[formula]
        if (up / down) >= h:
            premise_set.add(premise)

    while len(premise_set) > len(res_set):
        res_set = res_set | premise_set

        for name in premise_set:
            symbol_conclusion = dict_fof[name] | symbol_conclusion

        premise_set = set()
        for premise in dict_fof:
            up = 0
            down = 0
            for formula in dict_fof[premise]:
                down += w_dict[formula]
                if formula in symbol_conclusion:
                    up += w_dict[formula]
            if (up / down) >= h:
                premise_set.add(premise)

    return res_set

if __name__ == '__main__':

    file_path = "./l1_pepin"
    [list_conclusion, list_premise] = sInE.get_conclusion_premise_list(file_path)
    dict_fof = sInE.formula_list_to_dict(list_premise)

    for i in dict_fof:
        print(i + ":", end='    ')
        print(dict_fof[i])

    w_dict = get_dict_weigtht(dict_fof)
    print("dict_weight")
    for i in w_dict:
        print(i, end=':  ')
        print(w_dict[i])

    h = 0.5
    res = premise_select(h, list_conclusion[0], dict_fof)
    print("Relevance threshold h is %.2f, premises selected by symbol relevance:" %h)
    for name in res:
        print(name)
