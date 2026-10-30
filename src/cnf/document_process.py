import collections
import json
import numpy as np
import torch

from formula_parser import fof_formula_transformer, traversal_demo
from utils import read_file
import sInE
import symbol_weight





def get_conclusion_p_dict(path: str):
    with open(path, 'r', encoding='utf8') as file:
        paper = []
        for line in file.readlines():
            dic = json.loads(line)
            paper.append(dic)
    print(len(paper))
    dict_conclusion = dict()
    for line in paper:
        dict_conclusion[line[0]] = dict()

    for line in paper:
        temp = dict_conclusion[line[0]]
        temp[line[1]] = line[2]

    
    
    
    
    
    

    return dict_conclusion





def get_fof_dict(fof_path: str):
    lines = read_file(fof_path)
    fof_dict = dict()
    for line in lines:
        name = sInE.get_formula_name(line)
        fof_dict[name] = line
    print(len(fof_dict))
    return fof_dict





def get_premise_set(dict_conclusion, fof_dict):
    for name in dict_conclusion:
        prefix = "data/input"  
        filename = prefix + name + ".json"
        
        premise_dict = dict()
        for line in dict_conclusion[name]:
            premise_dict[fof_dict[line]] = dict_conclusion[name][line]
        with open(filename, 'w') as file_obj:
            json.dump(premise_dict, file_obj, indent=1)

    print(" ")



def get_correct(name: str, dict_conclusion, fof_dict):
    
    
    list_premise = []
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = sInE.get_tri_dict(dict_fof)
    
    
    
    
    

    
    
    

    
    
    name = sInE.get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    
    
    
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    if len(trigger_premise_set) == 0:
        
        return 1
    else:
        
        correct = 0
        count = 0
        for premise in trigger_premise_set:
            
            
            count += 1
            if dict_conclusion[name][premise] == 1:
                correct += 1
        res = correct / count
        print(" ", end=' ')
        print(count)
        print(" ", end=' ')
        print(correct)
        print(" ", end=' ')
        print(res)
        return res



def get_correct2(name: str, dict_conclusion, fof_dict):
    
    
    list_premise = []
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = sInE.get_tri_dict(dict_fof)
    
    
    
    
    

    
    
    

    
    
    name = sInE.get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    
    
    
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    if len(trigger_premise_set) == 0:
        
        return 1
    else:
        
        
        
        set_symbol = set()
        for premise in trigger_premise_set:
            set_symbol = set_symbol | dict_fof[premise]

        
        for symbol in set_symbol:
            for i in tri_premise:
                if symbol in tri_premise[i]:
                    trigger_premise_set.add(i)
        
        
        
        if len(trigger_premise_set) == 0:
            
            return 1
        else:
            
            
            

            correct = 0
            count = 0
            for premise in trigger_premise_set:
                
                
                count += 1
                if dict_conclusion[name][premise] == 1:
                    correct += 1
            res = correct / count
            print(" ", end=' ')
            print(count)
            print(" ", end=' ')
            print(correct)
            print(" ", end=' ')
            print(res)
            return res



def get_correct3(name: str, dict_conclusion, fof_dict):
    
    
    list_premise = []
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)
    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = sInE.get_tri_dict(dict_fof)
    
    
    name = sInE.get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    if len(trigger_premise_set) == 0:
        
        return 1
    else:
        
        
        
        set_symbol = set()
        for premise in trigger_premise_set:
            set_symbol = set_symbol | dict_fof[premise]

        
        for symbol in set_symbol:
            for i in tri_premise:
                if symbol in tri_premise[i]:
                    trigger_premise_set.add(i)
        
        
        
        if len(trigger_premise_set) == 0:
            
            return 1
        else:
            
            
            
            set_symbol = set()
            for premise in trigger_premise_set:
                set_symbol = set_symbol | dict_fof[premise]

            
            for symbol in set_symbol:
                for i in tri_premise:
                    if symbol in tri_premise[i]:
                        trigger_premise_set.add(i)
            
            
            
            if len(trigger_premise_set) == 0:
                
                return 1
            else:
                
                
                

                correct = 0
                count = 0
                for premise in trigger_premise_set:
                    
                    
                    count += 1
                    if dict_conclusion[name][premise] == 1:
                        correct += 1
                res = correct / count
                print(" ", end=' ')
                print(count)
                print(" ", end=' ')
                print(correct)
                print(" ", end=' ')
                print(res)
                return res



def get_correct_sw(name, h, dict_conclusion, fof_dict):
    
    list_premise = []      
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    res_set = symbol_weight.premise_select(h, conclusion, dict_fof)
    
    correct = 0
    count = 0
    for premise in res_set:
        
        
        count += 1
        if dict_conclusion[name][premise] == 1:
            correct += 1
    if count==0:
        return 1
    res = correct / count
    print(" ", end=' ')
    print(count)
    print(" ", end=' ')
    print(correct)
    print(" ", end=' ')
    print(res)
    return res






def get_similarity_dict(name: str, dict_conclusion, fof_dict):
    
    
    list_premise = []
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = sInE.get_tri_dict(dict_fof)
    
    
    
    
    

    
    
    

    
    
    name = sInE.get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    
    
    
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    
    
    simi_dict = collections.defaultdict(int)
    if len(trigger_premise_set) == 0:
        
        return simi_dict
    else:
        
        
        
        for premise in trigger_premise_set:
            
            
            simi_dict[premise] = 1

        return simi_dict






def get_similarity_dict2(name: str, dict_conclusion, fof_dict):
    
    
    list_premise = []
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    set_symbol = set()
    for premise in dict_fof:
        set_symbol = set_symbol | dict_fof[premise]

    tri_premise = sInE.get_tri_dict(dict_fof)
    
    
    
    
    

    
    
    

    
    
    name = sInE.get_formula_name(conclusion)
    
    conclusion = fof_formula_transformer(conclusion)
    symbol_conclusion = sInE.formula_to_set(conclusion)
    
    
    
    trigger_premise_set = set()
    for symbol in symbol_conclusion:
        for i in tri_premise:
            if symbol in tri_premise[i]:
                trigger_premise_set.add(i)

    
    
    simi_dict = collections.defaultdict(int)
    if len(trigger_premise_set) == 0:
        
        return simi_dict
    else:
        
        
        

        
        
        
        set_symbol = set()
        for premise in trigger_premise_set:
            set_symbol = set_symbol | dict_fof[premise]

        
        for symbol in set_symbol:
            for i in tri_premise:
                if symbol in tri_premise[i]:
                    trigger_premise_set.add(i)
        
        
        

        
        
        

        for premise in trigger_premise_set:
            
            
            simi_dict[premise] = 1

        return simi_dict






def get_simi_sw_dict(name, h, dict_conclusion, fof_dict):
    
    list_premise = []  
    print(name)
    for line in dict_conclusion[name]:  
        list_premise.append(fof_dict[line])
    
    conclusion = fof_dict[name]
    
    
    dict_fof = sInE.formula_list_to_dict(list_premise)

    
    
    

    
    res_set = symbol_weight.premise_select(h, conclusion, dict_fof)

    
    
    simi_dict = collections.defaultdict(int)
    if len(res_set) == 0:
        
        return simi_dict
    else:
        
        for premise in res_set:
            
            
            simi_dict[premise] = 1

        return simi_dict


def generate_simi_dict(dict_conclusion, fof_dict, path="data/input"):
    
    concl_simi_dict = dict()
    for name in dict_conclusion:
        concl_simi_dict[name] = get_similarity_dict(name, dict_conclusion, fof_dict)

    json_str = json.dumps(concl_simi_dict, indent=4)
    with open(path, 'w') as f:
        f.write(json_str)
    print(" ")


def generate_simi_dict2(dict_conclusion, fof_dict, path="data/input"):
    
    concl_simi_dict = dict()
    for name in dict_conclusion:
        concl_simi_dict[name] = get_similarity_dict2(name, dict_conclusion, fof_dict)

    json_str = json.dumps(concl_simi_dict, indent=4)
    with open(path, 'w') as f:
        f.write(json_str)
    print(" ")



def generate_simi_sw_dict(h, dict_conclusion, fof_dict, path="data/input"):
    
    concl_simi_dict = dict()
    for name in dict_conclusion:
        concl_simi_dict[name] = get_simi_sw_dict(name, h, dict_conclusion, fof_dict)

    json_str = json.dumps(concl_simi_dict, indent=4)
    with open(path, 'w') as f:
        f.write(json_str)
    print(" ")


def get_simi_from_json(path="dataset/similarity.json"):
    
    

    with open(path) as f:
        data = json.load(f)

    for i in data:
        data[i] = collections.defaultdict(lambda: 0, data[i])  

    return data


def get_simi_from_json2(path="dataset/similarity2.json"):
    
    

    with open(path) as f:
        data = json.load(f)

    for i in data:
        data[i] = collections.defaultdict(lambda: 0, data[i])  

    return data


def get_simi_sw_from_json(path="dataset/similarity4.json"):      
    
    
    with open(path) as f:
        data = json.load(f)
    for i in data:
        data[i] = collections.defaultdict(lambda: 0, data[i])  
    return data


if __name__ == "__main__":
    path = "data/input"
    fof_path = "data/input"
    
    dict_conclusion = get_conclusion_p_dict(path)
    print(" ")
    print(len(dict_conclusion))
    
    fof_dict = get_fof_dict(fof_path)
    
    

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    

    

    
    
    
    
    
    
    
    
    

    
    
    
    
    data = get_simi_sw_from_json()
    print(type(data))
    print(type(data["t14_funct_1"]))
    print(data["t14_funct_1"]["aa"])
    print(data["t14_funct_1"]["cc1_funct_1"])






    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

