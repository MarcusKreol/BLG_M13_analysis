"""
functions for controlled generation of random population structures
with kinship structure characteristics (in the form of ped-sim def files)

see supplementary note 3 for a detailed explanation of parameters
"""

import numpy as np
import random
import os

def get_number_list(struct_str): 
    
    """get number lists from strings like '1-5,7,9'"""
    
    number_list = []
    for x in list(filter(None,struct_str.split(','))):
        start_end = x.split('-')
        if len(start_end)==1:
            number_list.append(int(start_end[0]))
        else:
            number_list.extend(list(range(int(start_end[0]),int(start_end[1])+1)))
    return number_list

def find_key(dictionary, target_tuple):
    for key, value_list in dictionary.items():
        for value_tuple in value_list:
            if target_tuple == value_tuple:
                return key
    return None

def calculate_lengths(dictionary):
    max_length = 0
    total_length = 0

    for key, value_list in dictionary.items():
        if len(value_list)!=1:
            total_length += len(value_list)
        max_length = max(max_length, len(value_list))

    return max_length, total_length

def get_lineage_populations(def_text):
    
    """
    get the lineage composition and centrality of a population (in the form of ped-sim def text)
    usage: get_lineage_populations(def_text)
    """
    
    lineage_members = {}
    founders = []
    
    lines = list(filter(None,def_text.split("\n")))
    
    for i,line in enumerate(lines[1:]):
        content = list(filter(None,line.split(' ')))
        gen = i+1
        if gen == 1:
            for i in range(int(content[2])):
                founder = (1,i+1)
                founders.append(founder)
                lineage_members[founder] = [founder]
        else:
            for substr in content[3:]:
                if 's' not in substr:
                    children, parent = substr.split(':')
                    children = [(gen,x) for x in get_number_list(children)]
                    if parent=='':
                        for child in children:
                            founders.append(child)
                            lineage_members[child] = [child]
                    else:
                        parent = (gen-1, int(parent))
                        if parent in founders:
                            the_founder = parent
                        else:
                            the_founder = find_key(lineage_members, parent)
                        if parent+('s',) not in lineage_members[the_founder]:
                            lineage_members[the_founder].append(parent+('s',)) # add spouse
                        lineage_members[the_founder].extend(children)
                        
    max_num, total_num = calculate_lengths(lineage_members)
    centrality = max_num/total_num
    
    return lineage_members,total_num,centrality


def generate_def(name = 'test', save_folder = './defs', reps = 1, size = (100,150), generations = 10, mean_children = 2, outsider_rate = 0, rule = 'patrilineal', purity = 1, die_rate = 0.1,
                 compensate_outsider = False, founder_num = 1, centrality_requirement = None):
    
    """
    randomly generate a population (in the form of ped-sim def files) with custom kinship structure characteristics 
    usage: generate_def(name = 'test', reps = 1, size = (100,150), generations = 10, mean_children = 2, outsider_rate = 0, rule = 'patrilineal', purity = 1, die_rate = 0.1,
                 compensate_outsider = False, founder_num = 1, centrality_requirement = None)
                 
    Brief explanation of non-straightforward parameters:
    name: save name for def file. files will be saved to ./defs/{name}.def
    purity: proportion of paternal descendance in 'patrilineal' rule or maternal descendance in 'matrilineal' rule. Purity=1 corresponds to strict patrilineal / matrilineal.
    compensate_outsider: if set to True, the mean individual number of each generation will be kept identical whether with or without outsiders. 
    centrality requirement: tuple (lower,higher). Restrict generated populations to a centrality range.
    
    """
    
    copies = 1 # identical replicates, which is generally not needed
    sex_i1 = 'M' if rule=='patrilineal' else ('F' if rule=='matrilineal' else '')
    
    rep_num = 0
    write_list = []
    
    lineage_populations = {} # works with outsider mode
    lineage_number = 1
    
    
    while rep_num < reps: 
        break_flag = 0
        write_content = f"def {name}{rep_num} {copies} {generations} {sex_i1}\n"
        
        population = 0
        children_num_sum = founder_num
        reproducing_children_num_sum = founder_num
        children_num_list = []
        
        for gen in range(1,generations+1):
            
            if compensate_outsider:
                mean_outsider_num = outsider_rate*children_num_sum/(1+outsider_rate)
            else:
                mean_outsider_num = outsider_rate*children_num_sum
            outsider_num_raw = np.random.normal(mean_outsider_num, np.sqrt(mean_outsider_num))
            outsider_num = 0 if outsider_num_raw<=0 else int(np.floor(outsider_num_raw))
            
            # intialize pedigree statistics
            for i in range(1,outsider_num+1):
                lineage_populations[lineage_number+i] = 1
            lineage_number += outsider_num
            
            tot_gen_size = outsider_num + children_num_sum
            gen_size = outsider_num + reproducing_children_num_sum # Reproducing individuals
            
            if gen_size == 0:
                break_flag = 1 # abort this family if extincted
                break
            
            if gen == 1:
                write_content += f"{gen} {1} {tot_gen_size}\n" # without parent specification
                # no need for sex specification, since there's no difference anyway
                
            else: 
                par_spec = ''
                dum = 1
                dum_1 = gen_size+1
                for par_id, chil_num in enumerate(reproducing_children_num_list, start = 1): # reproducing children
                    die_chil_num = die_children_num_list[par_id-1]
                    old_dum, dum = dum, dum + chil_num # reproducing
                    old_dum_1, dum_1 = dum_1, dum_1 + die_chil_num # died
                    if chil_num ==1:
                        par_spec += f"{old_dum}"
                    elif chil_num >1 :
                        par_spec += f"{old_dum}-{dum-1}"
                    
                    if chil_num != 0:
                        par_spec += ','
                        
                    if die_chil_num ==1:
                        par_spec += f"{old_dum_1}"
                    elif die_chil_num >1 :
                        par_spec += f"{old_dum_1}-{dum_1-1}"
                    
                    if chil_num + die_chil_num >0 :
                        par_spec += f":{par_id} "
                
                if outsider_num == 1:   # outsider
                    par_spec += f"{dum}: "
                    outsider_range = (dum-1,dum+outsider_num-1)
                elif outsider_num > 1:
                    par_spec += f"{dum}-{dum+outsider_num-1}: "
                    outsider_range = (dum-1,dum+outsider_num-1)
                dum += outsider_num
                
                if rule: # assign sexes
                    minor_sex = 'sF' if rule=='patrilineal' else 'sM'
                    for chil_id in range(1,dum):
                        if random.random()>purity:
                            par_spec += f"{chil_id}{minor_sex} "
                            
                    for chil_id in range(dum,dum_1): 
                        if random.random()>0.5:
                            par_spec += f"{chil_id}{minor_sex} "
                
                write_content += f"{gen} {1} {tot_gen_size} "+par_spec+"\n"
            
            if compensate_outsider:
                children_num_list = np.floor(np.random.exponential(scale=mean_children/(1+outsider_rate), size = gen_size)).astype(int)
            else:
                children_num_list = np.floor(np.random.exponential(scale=mean_children, size = gen_size)).astype(int)
            if gen ==1:
                children_num_list[children_num_list==0] = 1 # at lest one children for founder generation
            
            die_children_num_list = np.random.binomial(children_num_list, die_rate)
            reproducing_children_num_list = children_num_list - die_children_num_list
            
            population += tot_gen_size
            if gen!= generations:
                population += np.sum(children_num_list>0) # one spouse for each person with children
                if gen>1:
                    try:
                        fuck = children_num_list==0
                        population -= sum(fuck[outsider_range[0]:outsider_range[1]]) # outsiders without offspring is excluded
                    except:
                        pass
            else:
                population -= outsider_num # outsiders from the last generation are illegal
            
            children_num_sum = np.sum(children_num_list)
            reproducing_children_num_sum = np.sum(reproducing_children_num_list)
                
        if break_flag == 1:
            continue
        
        if centrality_requirement is None:
            if size[0]<=population<=size[1]:
                write_list.append(write_content)
                rep_num += 1
                print(f"generated {rule} family tree with {population} valid individuals")
        else:
            _,_,centrality = get_lineage_populations(write_content)
            if centrality_requirement[0] <= centrality <= centrality_requirement[1]:
                if size[0]<=population<=size[1]:
                    write_list.append(write_content)
                    rep_num += 1
                    print(f"generated {rule} family tree with {population} valid individuals")
    
    os.makedirs(save_folder, exist_ok = True)
    with open(f"{save_folder}/{name}.def", "w") as def_file:
        for write_content in write_list:
            def_file.write(write_content)
            def_file.write("\n")
    
    return write_list[0]
