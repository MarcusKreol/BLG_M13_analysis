import numpy as np
import os as os
import sys as sys

from seg_process import matrix_simmilarity


np.random.seed(12)

def get_flat_similarities(flat_matrices, ref_matrix):
    # calculate similarity df
    samp_rep = 50
    
    flat_similarity = np.zeros((samp_rep, flat_matrices.shape[3]))

    for j in range(flat_matrices.shape[3]):
        if j%1000==0:
            print(f"similarity: {j}/{flat_matrices.shape[3]} done")
        for i in range(samp_rep):
            flat_similarity[i,j] = matrix_simmilarity(flat_matrices[:,:,i,j],ref_matrix)
                     
    return flat_similarity
    

def get_data_small(centrality_backup,matrix_backup,tree_backup,
                                         matrix_lists = [],tree_lists = [], centrality_lists = [],
                                         first_run = True, sample = False, sample_num = 100, avoid_list = [], custom_pop_range = None):
    # rearrange according to centrality
    
    np.random.seed(12) # for reproducibility
    samp_rep = 50
    pop_ranges = [(50,100),(100,150),(150,200),(200,300),(300,400),(400,500)]
    
    intervals_small = [(0,0.2),(0.2,0.25),(0.25,0.3),(0.3,0.35),(0.35,0.4),(0.4,0.45),(0.45,0.5),(0.5,0.55),
                    (0.55,0.6),(0.6,0.65),(0.65,0.7),(0.7,0.75),(0.75,0.8),(0.8,0.85),(0.85,0.9),(0.9,0.95),(0.95,0.99),(0.99,1)]
    
    if custom_pop_range:
        pop_ranges = custom_pop_range
    
    for pop_i in range(len(pop_ranges)):
        if first_run:
            matrix_lists.append([])
            tree_lists.append([])
            centrality_lists.append([])
        
        centrality_df = centrality_backup[:,:,pop_i]
        matrix_df = matrix_backup[:,:,:,:,:,pop_i]
        tree_df = tree_backup[:,:,pop_i]
        
        for int_i, interval in enumerate(intervals_small):
            print((pop_i,int_i))
            lower, upper = interval
            
            mask = (centrality_df > lower) & (centrality_df <= upper) & (~np.isin(tree_df, avoid_list))
            ind1, ind2 = np.where(mask)
            
            print((ind1.shape,ind2.shape))
            
            matrix_list = None
            centrality_list = []
            population_list = []
            tree_list = []
            
            for iii in range(len(ind1)):
                centrality_list.append(centrality_df[ind1[iii],ind2[iii]])
                tree_list.append(tree_df[ind1[iii],ind2[iii]])
                if matrix_list is None:
                    matrix_list = matrix_df[:,:,:,ind1[iii],ind2[iii]]
                    matrix_list = matrix_list[:,:,:,np.newaxis]
                else:
                    ml_new = matrix_df[:,:,:,ind1[iii],ind2[iii]]
                    ml_new = ml_new[:,:,:,np.newaxis]
                    matrix_list = np.concatenate((matrix_list,ml_new),axis = 3)
            
            tree_list = np.array(tree_list)
            centrality_list = np.array(centrality_list)
            population_list = np.array(population_list)
            
            if first_run:
                matrix_lists[pop_i].append(matrix_list)
                tree_lists[pop_i].append(tree_list)
                centrality_lists[pop_i].append(centrality_list)
            else:
                matrix_list_old = matrix_lists[pop_i][int_i]
                if matrix_list_old is not None:
                    if matrix_list is not None:
                        matrix_lists[pop_i][int_i] = np.concatenate((matrix_list_old,matrix_list),axis = 3)
                else:
                    matrix_lists[pop_i][int_i] = matrix_list
                tree_list_old = tree_lists[pop_i][int_i]
                tree_lists[pop_i][int_i] = np.concatenate((tree_list_old,tree_list),axis = 0)
                centrality_list_old = centrality_lists[pop_i][int_i]
                centrality_lists[pop_i][int_i] = np.concatenate((centrality_list_old,centrality_list),axis = 0)

    # sample so that centrality distributions are uniform
    if sample:
        for pop_i in range(len(pop_ranges)):
            for interval_i in range(len(intervals_small)):
                matrix_list = matrix_lists[pop_i][interval_i]
                tree_list = tree_lists[pop_i][interval_i]
                centrality_list = centrality_lists[pop_i][interval_i]
                
                # sanity check
                n = matrix_list.shape[3]
                print((pop_i,interval_i))
                print(n,centrality_list.shape[0])
                assert n == centrality_list.shape[0]
                
                length_interval = intervals_small[interval_i][1] - intervals_small[interval_i][0]
                if length_interval < 0.0499:
                    sample_num1 = int(sample_num*length_interval/0.05)
                else:
                    sample_num1 = sample_num
                if n > sample_num1:
                    selected_columns = np.random.choice(n, size=sample_num1, replace=False)
                    matrix_lists[pop_i][interval_i] = matrix_list[:,:,:,selected_columns]
                    tree_lists[pop_i][interval_i] = tree_list[selected_columns]
                    centrality_lists[pop_i][interval_i] = centrality_list[selected_columns]
    
    return matrix_lists, tree_lists, centrality_lists

def flatten_data_list(matrix_lists, tree_lists, centrality_lists):
    flat_tree = np.concatenate([np.concatenate(inner) for inner in tree_lists])
    flat_centrality = np.concatenate([np.concatenate(inner) for inner in centrality_lists])
    flat_matrices = np.concatenate([np.concatenate(inner, axis = 3) for inner in matrix_lists], axis = 3)
    
    return flat_matrices, flat_tree, flat_centrality