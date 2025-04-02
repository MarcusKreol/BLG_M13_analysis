'''
visualization functions for multi-way IBD based 2nd degree classification
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

def plot_scatter(data_list, show_size = False, y_names = None):
    num_rows = len(data_list) // 3 + (len(data_list) % 3 > 0)
    num_cols = min(len(data_list), 3)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # 将axes展平以便于迭代
    axes = axes.flatten()

    for i, obj in enumerate(data_list):
        
        row_idx = i // 3
        col_idx = i % 3
        
        if show_size:
            pair, array, lens = obj
            axes[i].scatter(array[:, 0], array[:, 1], s = 0.12*(np.mean(lens,axis = 1)), alpha = 0.8)
        else:
            try:
                pair, array = obj
            except ValueError:
                pair, array, _ = obj
            axes[i].scatter(array[:, 0], array[:, 1], alpha = 0.8)
        axes[i].set_title(f'{pair[0]}, {pair[1]}')
        axes[i].set_xlabel('R1')
        axes[i].set_ylabel('R2')
        
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

        # 画一条虚线对角线
        axes[i].plot([0, 1], [0, 1], '--', color='gray')

        # 添加网格
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        if y_names is not None:
            for j in range(array.shape[0]):
                axes[i].text(array[j, 0], array[j, 1], y_names[i][j][0], fontsize=7)

    plt.tight_layout()
    return fig
    
def plot_hist(data_list):
    num_rows = len(data_list) // 4 + (len(data_list) % 4 > 0)
    num_cols = min(len(data_list), 4)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

    # 将axes展平以便于迭代
    axes = axes.flatten()

    for i, (pair, array) in enumerate(data_list):
        row_idx = i // 4
        col_idx = i % 4

        axes[i].hist(array[:, 0]-array[:, 1])
        axes[i].set_title(f'{pair[0]}, {pair[1]}')
        axes[i].set_xlabel('R1-R2')
        axes[i].set_ylabel('Probability')
        
        axes[i].set_xlim(-0.5,0.5)
        # axes[i].set_ylim(0, 1)

        mean1 = np.mean(array[:, 0]-array[:, 1])
        axes[i].plot([mean1, mean1], [0, 20], '--', color='gray')

        # # 添加网格
        # axes[i].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

def plot_scatter_overlap(data_list, use_inds = None, show_size = False):

    fig = plt.figure(figsize=(6,6))

    legend_texts = []

    if (len(data_list[0])==3) & show_size:
        for i, (pair, array, lens) in enumerate(data_list):
            if use_inds is not None:
                if i not in use_inds:
                    continue
            plt.scatter(array[:, 0], array[:, 1], s = 0.2*(np.mean(lens,axis = 1)))
            legend_texts.append(f'{pair[0]}, {pair[1]}')
    else:
        try:
            for i, (pair, array) in enumerate(data_list):
                if use_inds is not None:
                    if i not in use_inds:
                        continue
                plt.scatter(array[:, 0], array[:, 1], s = 36)
                legend_texts.append(f'{pair[0]}, {pair[1]}')
        except:
            for i, (pair, array,_) in enumerate(data_list):
                if use_inds is not None:
                    if i not in use_inds:
                        continue
                plt.scatter(array[:, 0], array[:, 1], s = 36)
                legend_texts.append(f'{pair[0]}, {pair[1]}')
    
    
        
    plt.xlabel('R1',fontsize=18)
    plt.ylabel('R2',fontsize=18)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 画一条虚线对角线
    plt.plot([0, 1], [0, 1], '--', color='gray')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)   
    plt.legend(legend_texts, loc='upper left',fontsize=13) 
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.tight_layout()
    return fig

def plot_scatter_split_color(data_dict, show_size = False):
    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    order = ['GP','AV','HS']
    palette = sns.color_palette("CMRmap")
    palette[0] = (0.12, 0.15, 0.6)

    
    for key, (R1R2_dat, l1l2_dat) in data_dict.items():
        i = order.index(key[0])

        if not show_size:
            axes[i].scatter(R1R2_dat[:, 0], R1R2_dat[:, 1], s = 4, alpha = 0.02, marker = '.', color = palette[key[1]-1])
        else:
            axes[i].scatter(R1R2_dat[:, 0], R1R2_dat[:, 1], s = 0.02*(np.mean(l1l2_dat,axis = 1)), alpha = min(500/(R1R2_dat.shape[0]+1),0.8), marker = '.')
            
        axes[i].set_title(key[0], fontsize=24)
        axes[i].set_xlabel('R1', fontsize=24)
        axes[i].set_ylabel('R2', fontsize=24)
        
        axes[i].set_xlim(0, 1.1)
        axes[i].set_ylim(0, 1.1)

        axes[i].plot([0, 1.1], [0, 1.1], '--', color='gray')

        # 添加网格
        axes[i].grid(True, linestyle='--', alpha=0.7)
        i+=1
        
        # if j>15:
        #     break

    plt.tight_layout()
    return fig

## KDE for artificial data validation

def plot_kde(data_dict, bandwidth = 0.1, use_weights = False):
    num_rows = 3
    num_cols = 5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 9))

    # 将axes展平以便于迭代
    axes = axes.flatten()
    kde_dict = {}

    i=0
    for key, (R1R2_dat, l1l2_dat) in data_dict.items():
        
        try:
            if use_weights:
                kde = gaussian_kde(R1R2_dat.T,bw_method = bandwidth, weights=np.mean(l1l2_dat,axis = 1))
            else:
                kde = gaussian_kde(R1R2_dat.T,bw_method = bandwidth)
            
            kde_dict[key] = kde
            
            x_grid = np.linspace(0, 1, 100)
            y_grid = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])

            # 计算在网格点上的核密度估计值
            Z = np.reshape(kde(positions).T, X.shape)
            
            
            axes[i].contourf(X, Y, Z, cmap='Blues')  # 绘制核密度估计的等高线图
            # plt.colorbar(label='Density')
        except:
            pass
        # axes[i].scatter(R1R2_dat[:, 0], R1R2_dat[:, 1], s = 0.02*(np.mean(l1l2_dat,axis = 1)), alpha = min(100/(R1R2_dat.shape[0]+1),0.2), marker = '.', color='red')    
        
        axes[i].set_title(key)
        axes[i].set_xlabel('R1')
        axes[i].set_ylabel('R2')
        
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

        # 画一条虚线对角线
        axes[i].plot([0, 1], [0, 1], '--', color='gray')

        # 添加网格
        axes[i].grid(True, linestyle='--', alpha=0.7)
        i += 1

    plt.tight_layout()
    return fig, kde_dict

def plot_kde_specificity(kde_dict,y_type_probs):
    types = ['GP','AV','HS']
    
    ticks = 100
    prob_vals = np.zeros((ticks,ticks,3))
    
    x_grid = np.linspace(0, 1, ticks)
    y_grid = np.linspace(0, 1, ticks)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    i=0
    for key, kde in kde_dict.items():
        probs = np.reshape(kde(positions).T, X.shape)
        # print(probs.shape)
        ytype_ind = key[1]-1
        type_ind = types.index(key[0])
        prob_vals[:,:,type_ind] += y_type_probs[type_ind,ytype_ind]*probs
        prob_vals[:,:,type_ind] += y_type_probs[type_ind,ytype_ind]*probs.T
    
    specs = np.max(prob_vals,axis = 2)/np.sum(prob_vals,axis = 2)
    
    fig = plt.figure(figsize=(8,8))
    plt.contourf(X, Y, specs, cmap='Blues')
    
    return prob_vals,specs,fig
