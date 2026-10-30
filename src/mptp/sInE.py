from formula_parser import fof_formula_transformer, traversal_demo
from utils import read_file

import torch
import numpy



def get_conclusion_premise_list(file_path):
    lines = read_file(file_path)
    list_conclusion = []
    list_premise = []
    for line in lines:
        if line[0] == 'C':
            list_conclusion.append((line[2:]))
        else:
            list_premise.append(line[2:])
    
    
    

    return list_conclusion, list_premise




def formula_to_set(formula):      
    list_1 = []
    traversal_demo(formula, list_1)
    

    set_formula = set()
    for i in list_1:
        if i[0].islower() or i[0].isdigit():
            set_formula.add(i)
    
    return set_formula



def get_formula_name(formula):     
    for i in range(len(formula)):
        if formula[i] == '(':
            start = i+1
        if formula[i] == ',':
            formula_name = formula[start:i]
            break
    return formula_name



def formula_list_to_dict(list_fof):      
    dict_fof = dict()
    for fof in list_fof:
        formula = fof_formula_transformer(fof)
        set_formula = formula_to_set(formula)
        name = get_formula_name(fof)
        dict_fof[name] = set_formula
    return dict_fof



def get_tri_dict(dict_fof):      
    
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

    
    print("occ ")
    for i in occ_symbol:
        print(i, end=':  ')
        print(occ_symbol[i])

    
    tri_premise = dict()
    for premise in dict_fof:
        count = 10000

        
        
        
        
        
        

        
        for symbol in dict_fof[premise]:
            if occ_symbol[symbol] < count:
                tri_symbol = [symbol]
                count = occ_symbol[symbol]
            elif occ_symbol[symbol] == count:
                tri_symbol.append(symbol)
        tri_premise[premise] = tri_symbol
    return tri_premise


if __name__ == "__main__":


    
    
    
    

    
    file_path = "data/input"
    [list_conclusion, list_premise] = get_conclusion_premise_list(file_path)
    
    
    dict_fof = formula_list_to_dict(list_premise)

    for i in dict_fof:
        print(i + ":", end='    ')
        print(dict_fof[i])

    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = get_tri_dict(dict_fof)
    print(" ")
    for i in tri_premise:
        print(i + ":", end='    ')
        print(tri_premise[i])


    
    
    

    
    conclusion = list_conclusion[0]
    name = get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = formula_to_set(conclusion)
    print(name)
    
    print(symbol_conclusion)
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    if len(trigger_premise_set) == 0:
        print(" ")
    else:
        print(" ")
        for premise in trigger_premise_set:
            print(premise)

        
        
        set_symbol = set()
        for premise in trigger_premise_set:
            set_symbol = set_symbol | dict_fof[premise]

        
        for symbol in set_symbol:
            for i in tri_premise:
                if symbol in tri_premise[i]:
                    trigger_premise_set.add(i)
        print(" ")
        for premise in set_symbol:
            print(premise, end='   ')
        print()
        if len(trigger_premise_set) == 0:
            print(" ")
        else:
            print(" ")
            for premise in trigger_premise_set:
                print(premise)


