'''
2nd-degree classification through multi-way IBD sharing.
A rebuild base on the basic principles of CREST (Qiao et al., AJHG 2019)

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mutual_relative(pair, coef, order_thres = (3,6)):
    all_individuals =  list(set(coef[0].unique()).union(set(coef[1].unique())))
    mutual_relative_list = []
    order_level_list = []
    
    for individual in all_individuals:
        rel_level_1 = coef[((coef[0]==individual)&(coef[1]==pair[0]))|((coef[1]==individual)&(coef[0]==pair[0]))][5].tolist()
        rel_level_2 = coef[((coef[0]==individual)&(coef[1]==pair[1]))|((coef[1]==individual)&(coef[0]==pair[1]))][5].tolist()
        if len(rel_level_1)==1 and len(rel_level_2)==1:
            if (order_thres[0] <= rel_level_1[0] <= order_thres[1]) & (order_thres[0] <= rel_level_2[0] <= order_thres[1]):
                mutual_relative_list.append(individual)
                order_level_list.append((rel_level_1[0],rel_level_2[0]))
    return mutual_relative_list, order_level_list

def filter_segments(seg, pair):
    return seg[((seg[0]==pair[0]) & (seg[1]==pair[1])) | ((seg[0]==pair[1]) & (seg[1]==pair[0]))]

def add_noise(seg, noise):
    seg_new = seg.copy()
    for ind, _ in seg.iterrows():
        seg_new.loc[ind,6] = seg.loc[ind,6] + np.random.normal(0,noise)
        seg_new.loc[ind,7] = seg.loc[ind,7] + np.random.normal(0,noise)
    return seg_new
    

def calculate_length_and_intersections(spec_seg):
    total_length = 0
    intersections = []

    for _, row in spec_seg.iterrows():
        length = row[7] - row[6]
        total_length += length

        # 检查是否有交集
        intersecting_rows = spec_seg[(spec_seg.index != row.name) &
                                      (row[2] == spec_seg[2]) &
                                      ((row[7] > spec_seg[6]) & (row[6] < spec_seg[7]))]

        for _, intersect_row in intersecting_rows.iterrows():
            intersections.append((row.name, intersect_row.name, max(row[6], intersect_row[6]), min(row[7], intersect_row[7])))

    if len(intersections)>0:
        print(f'Detected {len(intersections)} intersections.')
        
    return total_length, intersections

def find_intersection_total(spec_seg1, spec_seg2):
    intersection_total = 0
    intersection_rows = []

    for _, row1 in spec_seg1.iterrows():
        for _, row2 in spec_seg2.iterrows():
            if row1[2] == row2[2] and ((row1[7] > row2[6]) and (row1[6] < row2[7])):
                intersection_rows.append((row1[2], max(row1[6], row2[6]), min(row1[7], row2[7])))
                intersection_total += min(row1[7], row2[7]) - max(row1[6], row2[6])

    return intersection_total, intersection_rows

def get_R1_R2(seg, coef, order_thres = (5,5), reorder = False, return_length = False):
    sec_pairs = coef[coef[5]==2]
    all_R1_R2s = []
    all_y_names = []
    
    for index,row in sec_pairs.iterrows():
        pair = (row[0],row[1])
        mutual_relative_list, order_level_list = get_mutual_relative(pair,coef,order_thres)
        R1_R2_summary = np.zeros((len(mutual_relative_list),2))
        l1_l2_summary = np.zeros((len(mutual_relative_list),2))
        all_y_names.append(list(zip(mutual_relative_list,order_level_list)))
        
        for i, individ in enumerate(mutual_relative_list):
            spec_seg_1 = filter_segments(seg, (pair[0],individ))
            spec_seg_2 = filter_segments(seg, (pair[1],individ))
            IBD_len_1, _ = calculate_length_and_intersections(spec_seg_1)
            IBD_len_2, _ = calculate_length_and_intersections(spec_seg_2)
            IBD_len_mutual, _ = find_intersection_total(spec_seg_1, spec_seg_2)
            
            if (IBD_len_1>0) & (IBD_len_2>0):
                if reorder:
                    R1_R2_summary[i,0] = min(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                    R1_R2_summary[i,1] = max(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                else:
                    R1_R2_summary[i,0] = IBD_len_mutual/IBD_len_1
                    R1_R2_summary[i,1] = IBD_len_mutual/IBD_len_2
                
                if return_length:
                    l1_l2_summary[i,0] = IBD_len_1
                    l1_l2_summary[i,1] = IBD_len_2
        
        if return_length:
            all_R1_R2s.append((pair,R1_R2_summary,l1_l2_summary))
        else:
            all_R1_R2s.append((pair,R1_R2_summary))
        
    return all_R1_R2s, all_y_names

### extensions for ped-sim
    
def get_R1_R2_stat(seg, coef, fam, order_thres = (5,5), reorder = False, return_length = False, noise = 0):
    sec_pairs = coef[coef[5]==2].copy()
    sec_pairs.reset_index(inplace = True)
    all_R1_R2s = []
    all_y_names = []
    
    if noise!=0:
        seg = add_noise(seg,noise)
    for index,row in sec_pairs.iterrows():
        pair_raw = (row[0],row[1])
        pair_spec = classify_2nd_degree(fam,pair_raw)
        
        pair = pair_spec[0]
        if (index%50 == 0) | (index+1==len(sec_pairs)):
            print(f'{pair}, {index+1}/{len(sec_pairs)} pairs done')
        mutual_relative_list = get_mutual_relative(pair,coef,order_thres)
        
        R1_R2_summary = np.zeros((len(mutual_relative_list),2))
        l1_l2_summary = np.zeros((len(mutual_relative_list),2))
        y_type_summary = np.zeros((len(mutual_relative_list),))
        all_y_names.append(mutual_relative_list)
        
        for i, individ in enumerate(mutual_relative_list):
            spec_seg_1 = filter_segments(seg, (pair[0],individ))
            spec_seg_2 = filter_segments(seg, (pair[1],individ))
            IBD_len_1, _ = calculate_length_and_intersections(spec_seg_1)
            IBD_len_2, _ = calculate_length_and_intersections(spec_seg_2)
            IBD_len_mutual, _ = find_intersection_total(spec_seg_1, spec_seg_2)
            
            if (IBD_len_1>0) & (IBD_len_2>0):
                y_type = classify_mutual_relative(fam, pair_spec, individ)
                y_type_summary[i] = y_type
                if reorder:
                    R1_R2_summary[i,0] = min(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                    R1_R2_summary[i,1] = max(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                else:
                    R1_R2_summary[i,0] = IBD_len_mutual/IBD_len_1
                    R1_R2_summary[i,1] = IBD_len_mutual/IBD_len_2
                
                if return_length:
                    l1_l2_summary[i,0] = IBD_len_1
                    l1_l2_summary[i,1] = IBD_len_2
        
        if return_length:
            all_R1_R2s.append((*pair_spec,R1_R2_summary,l1_l2_summary,y_type_summary))
        else:
            all_R1_R2s.append((*pair_spec,R1_R2_summary,y_type_summary))
        
    return all_R1_R2s, all_y_names

def get_R1_R2_stat_sample(seg, coef1, fam, order_thres = (5,5), reorder = False, return_length = False, noise = 0, sample_num = 30):
    individ_list = coef1[0].tolist() + coef1[1].tolist()
    individ_list = list(set(individ_list))
    sampled = random.sample(individ_list, min(sample_num, len(individ_list)))
    
    coef = coef1[coef1[0].isin(sampled) & coef1[1].isin(sampled)]
    
    sec_pairs = coef[coef[5]==2].copy()
    sec_pairs.reset_index(inplace = True)
    all_R1_R2s = []
    
    if noise!=0:
        seg = add_noise(seg,noise)
    for index,row in sec_pairs.iterrows():
        pair_raw = (row[0],row[1])
        pair_spec = classify_2nd_degree(fam,pair_raw)
        
        pair = pair_spec[0]
        if (index%50 == 0) | (index+1==len(sec_pairs)):
            print(f'{pair}, {index+1}/{len(sec_pairs)} pairs done')
        mutual_relative_list = get_mutual_relative(pair,coef,order_thres)
        
        R1_R2_summary = np.zeros((len(mutual_relative_list),2))
        l1_l2_summary = np.zeros((len(mutual_relative_list),2))
        y_type_summary = np.zeros((len(mutual_relative_list),))
        
        for i, individ in enumerate(mutual_relative_list):
            spec_seg_1 = filter_segments(seg, (pair[0],individ))
            spec_seg_2 = filter_segments(seg, (pair[1],individ))
            IBD_len_1, _ = calculate_length_and_intersections(spec_seg_1)
            IBD_len_2, _ = calculate_length_and_intersections(spec_seg_2)
            IBD_len_mutual, _ = find_intersection_total(spec_seg_1, spec_seg_2)
            
            if (IBD_len_1>0) & (IBD_len_2>0):
                y_type = classify_mutual_relative(fam, pair_spec, individ)
                y_type_summary[i] = y_type
                if reorder:
                    R1_R2_summary[i,0] = min(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                    R1_R2_summary[i,1] = max(IBD_len_mutual/IBD_len_1, IBD_len_mutual/IBD_len_2)
                else:
                    R1_R2_summary[i,0] = IBD_len_mutual/IBD_len_1
                    R1_R2_summary[i,1] = IBD_len_mutual/IBD_len_2
                
                if return_length:
                    l1_l2_summary[i,0] = IBD_len_1
                    l1_l2_summary[i,1] = IBD_len_2
        
        if return_length:
            all_R1_R2s.append((*pair_spec,R1_R2_summary,l1_l2_summary,y_type_summary))
        else:
            all_R1_R2s.append((*pair_spec,R1_R2_summary,y_type_summary))
        
    return all_R1_R2s
