"""
This module contains functions for processing segment files and generating matrices for IBD results.  

"""

import pandas as pd
import os
import random 
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

def sample_individuals_mf(seg_path, sex_ref = None, sex_rep = (1,2), male_sample_num = 25,female_sample_num = 12, save = True, suffix = '_new'):
    """
    Randomly sample individuals according to given male and female individual numbers in a pedsim output segment file, and save to a new file. 
    A sex reference dataframe with fields 'iid' (matching those in the seg file) and 'sex' should be provided.

    Parameters:
    seg_path (str): Path to the segment file.
    sex_ref (pd.DataFrame, optional): DataFrame with fields 'iid' (matching those in the seg file) and 'sex'. Default is None.
    sex_rep (tuple, optional): Tuple representing the sex identifiers for male and female. Default is (1, 2).
    male_sample_num (int, optional): Number of male individuals to sample. Default is 25.
    female_sample_num (int, optional): Number of female individuals to sample. Default is 12.
    save (bool, optional): Whether to save the filtered DataFrame to a new file. Default is True.
    suffix (str, optional): Suffix to add to the new file name. Default is '_new'.
    
    Returns:
    tuple: A tuple containing:
        - filtered_df (pd.DataFrame): The filtered DataFrame with sampled individuals.
        - random_elements (list): List of sampled individual IDs.
        - flag (int): Flag indicating the completion of the function (1 if successful).
    """
    
    flag = 0
    df = pd.read_csv(seg_path, header=None, delim_whitespace=True)

    elements_list = df[0].tolist() + df[1].tolist()

    unique_elements = list(set(elements_list))
    
    m_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == sex_rep[0]].tolist()]
    f_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == sex_rep[1]].tolist()]
            
    m_elements_1 = random.sample(m_elements, male_sample_num)
    f_elements_1 = random.sample(f_elements, female_sample_num)
    random_elements = m_elements_1+f_elements_1

    filtered_df = df[df[0].isin(random_elements) & df[1].isin(random_elements)]
    
    seg_basename = os.path.splitext(os.path.basename(seg_path))[0]
    new_seg_path = os.path.join(os.path.dirname(seg_path), seg_basename + suffix + ".seg")
    
    if save:
        filtered_df.to_csv(new_seg_path, sep="\t", header=False, index=False)
    
    flag = 1
    
    return filtered_df, random_elements, flag


def generate_mt_df(fam_info):
    
    """
    Generate a DataFrame with mitochondrial haplogroup (mt) values based on family information.
    This function processes a DataFrame containing family information, typically from a .fam file.
    It assigns a unique mitochondrial transmission (mt) value to each individual based on their 
    maternal lineage.
    Parameters:
    fam_info (pd.DataFrame): A DataFrame with family information. The DataFrame should have the 
                            following columns: [0, 1, 2, 3, 4, 5], where:
                            - Column 0: Individual ID (iid)
                            - Column 1: Father ID (father)
                            - Column 2: Mother ID (mother)
                            - Column 3: Sex (sex)
                            - Column 4: Unused
                            - Column 5: Unused
    Returns:
    pd.DataFrame: A DataFrame with the following columns: ['iid', 'sex', 'mt'], where 'mt' 
                represents the mitochondrial transmission value for each individual.
    """
    
    fam_info = fam_info.drop([0,5], axis = 1)
    fam_info.columns = ['iid','father','mother','sex']
    fam_info['iid_number'] = fam_info['iid'].str.split('_').str.get(-1).str.split('-').str.get(0).str.split('g').str.get(-1).astype(int)
    fam_info.sort_values(by='iid_number', inplace=True)

    fam_info['mt'] = 0
    counter = 1

    for index, row in fam_info.iterrows():
        mother_value = row['mother']
        if mother_value == '0':
            fam_info.at[index, 'mt'] = counter
            counter += 1
        else:
            mt_value = fam_info.loc[fam_info['iid'] == mother_value, 'mt'].values
            assert(len(fam_info.loc[fam_info['iid'] == mother_value, 'mt'])==1)
            fam_info.at[index, 'mt'] = mt_value

    fam_info.drop(columns=['iid_number','father','mother'], inplace=True)
    return fam_info


def determine_kinship(value):
    """IBD ranges for different kinship degrees, For sorting of IBD matrices later on"""
    
    IBD_ref_len = [(4000,2800),(2800,2000),(2000,1000),(1000,450),(450,150),(150,20),(20,0)]
    IBD_ref_len = list(reversed(IBD_ref_len))
    
    for i, (lower, upper) in enumerate(IBD_ref_len):
        if lower > value >= upper:
            return i
    return 0

def generate_matrix(df, sex_ref, clustering = True, order = 'orig', mt_ref = None, keep_sex = True, plot = True):
    
    """ 
    Generate a matrix representing the sum of IBD lengths > 20 between individuals, with options for ordering, clustering, and plotting.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing IBD information with columns 'iid1', 'iid2', 'sum_IBD>20'.
    order (str, optional): Ordering method for individuals. 'orig' for original order, 'level' for kinship level order. Default is 'orig'.
    mt_ref (pd.DataFrame, optional): DataFrame containing mitochondrial haplogroup information with columns 'iid' and 'mt'. Default is None.
    keep_sex (bool, optional): Whether to keep sex-specific ordering (females after males). Default is True.
    clustering (bool, optional): Whether to apply hierarchical clustering to the matrix. Default is True.
    plot (bool, optional): Whether to plot the resulting matrix as a heatmap. Default is True.
    
    Returns:
    np.ndarray: A matrix representing the sum of IBD lengths > 20 between individuals.
    """
    
    unique_elements = list(set(df['iid1'].unique()).union(set(df['iid2'].unique())))
    quant_df = df.copy()
    quant_df['kinship'] = quant_df['sum_IBD>20'].apply(determine_kinship)
    
    if keep_sex: # male ratio
        m_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == 'M'].tolist()]
        f_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == 'F'].tolist()]
        
        if len(m_elements)+len(f_elements)==0:
            m_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == 1].tolist()]
            f_elements = [element for element in unique_elements if element in sex_ref['iid'][sex_ref['sex'] == 2].tolist()]

        # print(len(m_elements))
        # print(len(f_elements))
        assert(len(m_elements)+len(f_elements) == len(unique_elements))
        
        if not clustering:
            if order == 'orig':
                m_elements.sort(key=lambda x: df[df['iid1'] == x]['sum_IBD>20'].sum() + df[df['iid2'] == x]['sum_IBD>20'].sum(), reverse=True)
                f_elements.sort(key=lambda x: df[df['iid1'] == x]['sum_IBD>20'].sum() + df[df['iid2'] == x]['sum_IBD>20'].sum(), reverse=True)
            elif order == 'level':
                m_elements.sort(key=lambda x: quant_df[quant_df['iid1'] == x]['kinship'].sum() + quant_df[quant_df['iid2'] == x]['kinship'].sum(), reverse=True)
                f_elements.sort(key=lambda x: quant_df[quant_df['iid1'] == x]['kinship'].sum() + quant_df[quant_df['iid2'] == x]['kinship'].sum(), reverse=True)
            
        lst = m_elements+f_elements
    else:
        lst = unique_elements
        
        if not clustering:
            if order == 'orig':
                lst = sorted(unique_elements, key=lambda x: df[df['iid1'] == x]['kinship'].sum() + df[df['iid2'] == x]['kinship'].sum(), reverse=True)
            elif order == 'level':
                lst = sorted(unique_elements, key=lambda x: quant_df[quant_df['iid1'] == x]['kinship'].sum() + quant_df[quant_df['iid2'] == x]['kinship'].sum(), reverse=True)

    n = len(lst)
    result_matrix = np.zeros((n, n))

    for index, row in df.iterrows():
        iid1_index = lst.index(row['iid1'])
        iid2_index = lst.index(row['iid2'])
        result_matrix[iid1_index][iid2_index] = row['sum_IBD>20']
        result_matrix[iid2_index][iid1_index] = row['sum_IBD>20']
        if mt_ref is not None:
            if mt_ref.loc[mt_ref['iid'] == row['iid1'], 'mt'].values == mt_ref.loc[mt_ref['iid'] == row['iid2'], 'mt'].values:
                result_matrix[iid1_index][iid2_index] = -row['sum_IBD>20']
                result_matrix[iid2_index][iid1_index] = -row['sum_IBD>20']

    if clustering:
        new_dist = 4200-np.abs(result_matrix)
        np.fill_diagonal(new_dist, 0)
        if keep_sex:
            new_dist_m = new_dist[:len(m_elements),:len(m_elements)]
            new_dist_f = new_dist[len(m_elements):,len(m_elements):]
            
            linkage_matrix_m = linkage(squareform(new_dist_m), method='average', optimal_ordering=True) #UPGMA
            linkage_matrix_f = linkage(squareform(new_dist_f), method='average', optimal_ordering=True)
            optimal_order_m = np.array(leaves_list(linkage_matrix_m))
            optimal_order_f = np.array(leaves_list(linkage_matrix_f))+len(m_elements)
            
            optimal_order = optimal_order_m.tolist()+optimal_order_f.tolist()
        else:
            linkage_matrix = linkage(squareform(new_dist), method='average', optimal_ordering=True) #UPGMA
            optimal_order = leaves_list(linkage_matrix)
        
        result_matrix = result_matrix[optimal_order][:, optimal_order]
        lst = np.array(lst)[optimal_order]
        lst = lst.tolist()
        
    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(result_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='sum IBD length > 20')
        plt.xticks(np.arange(n), lst, rotation = 90)
        plt.yticks(np.arange(n), lst)
        plt.show()  
        
        # plt.figure(figsize=(10, 8))
        # new_dist = 4200-result_matrix
        # np.fill_diagonal(new_dist, 0)
        # new_dist = squareform(new_dist)
        # linkage_matrix = linkage(new_dist, method='average')
        # sns.clustermap(result_matrix, row_linkage = linkage_matrix, col_linkage = linkage_matrix)
        # plt.show() 

    return result_matrix

def generate_2x2_matrix(df, sex_ref, plot = True, namename = 'MF',title = ''):
    """ 
    Generate a 2x2 matrix representing the mean sum of IBD lengths > 20 between different sex combinations.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing IBD information with columns 'iid1', 'iid2', and 'sum_IBD>20'.
    sex_ref (pd.DataFrame): DataFrame containing sex information with columns 'iid' and 'sex'.
    plot (bool, optional): Whether to plot the resulting matrix. Default is True.
    namename (str, optional): Naming convention for sex identifiers. 'MF' for 'M' and 'F', '12' for 1 and 2. Default is 'MF'.
    title (str, optional): Title for the plot. Default is an empty string.
    
    Returns:
    np.ndarray: A 2x2 matrix with mean sum of IBD lengths > 20 for different sex combinations.
    """
    
    
    df1 = df.merge(sex_ref, left_on = 'iid1', right_on = 'iid',how='left')
    df1 = df1.rename(columns={'sex': 'sex1'})
    df1 = df1.drop('iid', axis=1)

    df1 = df1.merge(sex_ref, left_on = 'iid2', right_on = 'iid',how='left')
    df1 = df1.rename(columns={'sex': 'sex2'})
    df1 = df1.drop('iid', axis=1)
    
    if namename == 'MF':
        df_mf = df1[((df1['sex1'] == 'M') & (df1['sex2'] == 'F')) |
                    ((df1['sex1'] == 'F') & (df1['sex2'] == 'M'))]
        df_mm, df_ff = df1[(df1['sex1'] == 'M') & (df1['sex2'] == 'M')], df1[(df1['sex1'] == 'F') & (df1['sex2'] == 'F')]
    elif namename == '12':
        df_mf = df1[((df1['sex1'] == 1) & (df1['sex2'] == 2)) |
                    ((df1['sex1'] == 2) & (df1['sex2'] == 1))]
        df_mm, df_ff = df1[(df1['sex1'] == 1) & (df1['sex2'] == 1)], df1[(df1['sex1'] == 2) & (df1['sex2'] == 2)]
    
    result_matrix = np.zeros((2,2))
    
    result_matrix[0,0] = np.mean(df_mm['sum_IBD>20'])
    result_matrix[0,1] = np.mean(df_mf['sum_IBD>20'])
    result_matrix[1,0] = np.mean(df_mf['sum_IBD>20'])
    result_matrix[1,1] = np.mean(df_ff['sum_IBD>20'])
    
    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(result_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='sum IBD length > 20')
        plt.title(title)
        plt.show()    

    return result_matrix

def matrix_simmilarity(mat1,mat2, method = 'fro'):
    """
    Calculate the similarity between two matrices using the specified method.
    
    Parameters:
    mat1 (np.ndarray): The first matrix.
    mat2 (np.ndarray): The second matrix.
    method (str, optional): The method to use for calculating similarity. 'fro' for Frobenius norm, 'pearson' for Pearson correlation. Default is 'fro'.
    
    Returns:
    float: The similarity score between the two matrices.
    """
    
    if method == 'fro':
        differ = mat1 - mat2
        dist = np.linalg.norm(differ, ord = method)
        len1 = np.linalg.norm(mat1, ord = method)
        len2 = np.linalg.norm(mat2, ord = method)
        denom = len1+len2
        sim = 1-(dist/denom)
    elif method == 'pearson':
        flat_matrix1 = mat1.flatten()
        flat_matrix2 = mat2.flatten()
        sim = np.corrcoef(flat_matrix1, flat_matrix2)[0, 1]
    return sim



