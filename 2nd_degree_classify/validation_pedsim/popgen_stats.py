'''
Functions for pedigree simulation and multi-way IBD sharing statistics on simulated datasets.
'''

import numpy as np
import random


def generate_def_simple(name = 'test', size = (100,150), generations = 10, mean_children = 2, rule = 'patrilineal', purity = 1, mean_spouse_num = 1,
                        max_generation_size = 500):
    
    copies = 1 # identical replicates, which is generally not needed
    sex_i1 = 'M' if rule=='patrilineal' else ('F' if rule=='matrilineal' else '')
    
    while True: 
        generation_sizes = []
        
        break_flag = 0
        write_content = f"def {name} {copies} {generations} {sex_i1}\n"
        
        population = 0
        children_num_sum = 1
        children_num_list = []
        
        for gen in range(1,generations+1):
            gen_size = children_num_sum # Reproducing individuals
            generation_sizes.append(gen_size)
            
            if gen_size == 0:
                break_flag = 1 # abort this family if extincted
                break
            
            if gen == 1:
                write_content += f"{gen} {1} {gen_size}\n" # without parent specification
                # no need for sex specification, since there's no difference anyway
                
            else: 
                par_spec = ''
                chil_inds =np.cumsum(children_num_list)
                chil_inds = np.insert(chil_inds,0,0)
                for par_id, chil_num in enumerate(children_num_list, start = 1): # reproducing children
                    if chil_num == 1:
                        par_spec += f"{chil_inds[par_id]}:{par_id} "
                    elif chil_num >1 :
                        spouse_num = spouse_num_list[par_id-1]
                        if spouse_num == 1:
                            par_spec += f"{chil_inds[par_id-1]+1}-{chil_inds[par_id]}:{par_id} "
                        else:
                            start_ind = chil_inds[par_id-1]+1
                            remaining = chil_num
                            for sp_id in range(spouse_num):
                                if (sp_id+1<spouse_num) & (remaining>0):
                                    chil_num_for_spouse = np.random.binomial(remaining, 1/(spouse_num-sp_id-1))
                                    if chil_num_for_spouse>0:
                                        if chil_num_for_spouse==1:
                                            par_spec += f"{start_ind}:{par_id} "
                                        else:    
                                            par_spec += f"{start_ind}-{start_ind+chil_num_for_spouse-1}:{par_id} "
                                        start_ind += chil_num_for_spouse
                                        remaining -= chil_num_for_spouse
                                elif (sp_id+1==spouse_num) & (remaining>0):
                                    par_spec += f"{start_ind}-{chil_inds[par_id]}:{par_id} "      
                        
                if rule: # assign sexes
                    minor_sex = 'sF' if rule=='patrilineal' else 'sM'
                    for chil_id in range(1,len(children_num_list)+1):
                        if random.random()>purity:
                            par_spec += f"{chil_id}{minor_sex} "
                
                write_content += f"{gen} {1} {gen_size} "+par_spec+"\n"
            
            spouse_num_list = np.ceil(np.random.exponential(scale=mean_spouse_num, size = gen_size)).astype(int)
            children_num_list = np.floor(np.random.exponential(scale=mean_children*spouse_num_list, size = gen_size)).astype(int)
            if gen ==1:
                children_num_list[children_num_list==0] = 1 # at lest one children for founder generation
            
            population += gen_size
            if gen != generations:
                population += np.sum(spouse_num_list*(children_num_list>0)) # one spouse for each person with children
                generation_sizes[gen-1] += np.sum(spouse_num_list*(children_num_list>0))
                children_num_sum = np.sum(children_num_list)
                
        if break_flag == 1:
            continue
        if size[0]<=population<=size[1]:
            if max(generation_sizes) <= max_generation_size:
                print(f"generated {rule} family tree with {population} valid individuals")
                # print(write_content)
                break
    
    return write_content


