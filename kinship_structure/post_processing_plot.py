""" 
Plotting functions to visualize results.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pdf2image import convert_from_path
import subprocess
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def plot_results(df, xticks, xlabel, ylabel, ref_value = None, filename = f'../plots/test.png', figtype = 'violin', ref = True, title = ''
                 ,ylim = (0.15,0.65), ttest = False):
    
    plt.figure(figsize=(8, 8))
    
    if figtype == 'violin':
        ax = sns.violinplot(data=df, inner="quartiles", cut=0)
    elif figtype == 'box':
        ax = sns.boxplot(data=df)
    if ref:
        plt.axhline(y=ref_value, color='red', linestyle='--', label='BLG value')

    ax.set_xticklabels(xticks)  
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel) 
    ax.set_ylim((ylim))
    plt.title(title)
    
    if ttest:
        col_data = df.to_numpy()
        for j in range(1, col_data.shape[1]):
            _, ttest_p_value = stats.ttest_ind(col_data[:, 0], col_data[:, j])
            
            if ttest_p_value<0.05:
                p_str = '*'
            if ttest_p_value<0.01:
                p_str = '**'
            if ttest_p_value<0.001:
                p_str = '***'
            if ttest_p_value<0.0001:
                p_str = '****'
            if ttest_p_value>0.05:
                p_str = 'ns'
            ax.text(j, 0.9*ylim[1], p_str, fontsize=15, ha='center', va='center')
    
    plt.savefig(filename)
    
def plot_results_sep(df, xticks, xlabel, ylabel, ref_value = None, figtype = 'violin', ref = False, sep_line = False, title = ''
                 ,ylim = (0.15,0.65), ttest = False):
    # accepts a list of numpy arrays
    
    fig = plt.figure(figsize=(10, 8))
    
    if figtype == 'violin':
        ax = sns.violinplot(data=df, inner="quartiles", cut=0, scale='width', palette = "colorblind", width = 0.6)
    elif figtype == 'box':
        ax = sns.boxplot(data=df)
    if ref:
        plt.axhline(y=ref_value, color='red', linestyle='--', label='BLG value')
    
    if sep_line:
        plt.axvline(x=1.5, color='black')
    
    ax.set_xticklabels(xticks) 
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)
    ax.set_ylim((ylim))
    plt.title(title)
    
    if ttest:
        for j in range(1, len(df)):
            _, ttest_p_value = stats.ttest_ind(df[j], df[0])
            
            if ttest_p_value<0.05:
                p_str = '*'
            if ttest_p_value<0.01:
                p_str = '**'
            if ttest_p_value<0.001:
                p_str = '***'
            if ttest_p_value<0.0001:
                p_str = '****'
            if ttest_p_value>0.05:
                p_str = 'ns'
            ax.text(j, 0.9*ylim[1], p_str, fontsize=15, ha='center', va='center')
    
    plt.show()
    return fig

def plot_results_sep_new(df, xticks, xlabel, ylabel, ref_value = None, figtype = 'violin', ref = False, title = ''
                 ,ylim = (0.15,0.65), ttest = False, test_method = 'ttest'):
    plt.rcParams['axes.linewidth'] = 2  
    plt.rcParams['xtick.major.width'] = 2 
    plt.rcParams['ytick.major.width'] = 2  
    plt.rcParams['xtick.minor.width'] = 1 
    plt.rcParams['ytick.minor.width'] = 1 
    plt.rcParams['xtick.labelsize'] = 16  
    plt.rcParams['ytick.labelsize'] = 16 
    plt.rcParams['axes.labelsize'] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [len(df)-1, 1]})
    
    
    if figtype == 'violin':
        sns.violinplot(data=df[:-1], ax=ax1, cut=0, inner = 'box', density_norm='area', width = 0.6, linewidth = 2,
                       inner_kws=dict(box_width=12, whis_width = 2))
        sns.violinplot(data=[df[-1]], ax=ax2, cut=0, inner = 'box', density_norm='area', width = 0.6, linewidth = 2,
                       inner_kws=dict(box_width=12, whis_width = 2))
    elif figtype == 'box':
        sns.boxplot(data=df[:-1], ax=ax1)
        sns.boxplot(data=[df[-1]], ax=ax2)
        
    if ref:
        ax1.axhline(y=ref_value, color='red', linestyle = '--', label='BLG value')
        ax2.axhline(y=ref_value, color='red', linestyle = '--', label='BLG value')
    
    ax1.grid(axis='y',linestyle='--', color='black', linewidth=1, alpha = 0.5)
    ax2.grid(axis='y',linestyle='--', color='black', linewidth=1, alpha = 0.5)
    
    
    ax1.set_xticklabels(xticks[:-1])
    ax2.set_xticklabels([xticks[-1]]) 
    ax1.set_xlabel(xlabel) 
    ax1.set_ylabel(ylabel) 
    ax1.set_ylim((ylim))
    fig.suptitle(title)
    
    # draw percentile box
    
    rect_height = 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    rect_bot = ax1.get_ylim()[0]+0.1*rect_height
    palette = sns.color_palette()[:2]
    rect1 = Rectangle((-0.1, rect_bot), 0.2, rect_height, color=palette[0], alpha=1)
    rect2 = Rectangle((0.9, rect_bot), 0.2, rect_height, color=palette[1], alpha=1)
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    
    outer_rect_path = Path([
        (1.3, rect_bot), 
        (-0.1, rect_bot),  
        (-0.1, rect_bot + rect_height), 
        (1.3, rect_bot + rect_height)
    ])
    outer_rect = PathPatch(outer_rect_path, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(outer_rect)
    ax1.text(-0.42, rect_bot + 0.5*rect_height, "(top)", verticalalignment='center', fontsize=16)
    ax1.spines['bottom'].set_visible(False)
    
    
    plt.subplots_adjust(wspace=0.2)
    
    if ttest:
        for j in range(1, len(df)):
            if test_method == "ttest":
                tval, ttest_p_value = stats.ttest_ind(df[j], df[0])
                symbol = 't'
            elif test_method == "mlu":
                tval, ttest_p_value = stats.mannwhitneyu(df[j], df[0])
                symbol = 'u'
            
            if ttest_p_value<0.05:
                p_str = '*'
            if ttest_p_value<0.01:
                p_str = '**'
            if ttest_p_value<0.001:
                p_str = '***'
            if ttest_p_value<0.0001:
                p_str = '****'
            if ttest_p_value>0.05:
                p_str = 'ns'
            if j != len(df)-1:
                ax1.text(j, 0.9*ylim[1], p_str, fontsize=15, ha='center', va='center')
            # else:
            #     ax2.text(0, 0.9*ylim[1], p_str, fontsize=15, ha='center', va='center')
            
            print(f"{symbol} = {tval}, p = {ttest_p_value}, n{j},n0 = {len(df[j])},{len(df[0])}")
    
    
    plt.show()
    return fig

def get_pvals(similarity_df):
    normality_pvals = np.zeros((similarity_df.shape[1],similarity_df.shape[2]))
    test_pvals = np.zeros((similarity_df.shape[1]-1,similarity_df.shape[2]))+1
    test_symbols = np.empty((similarity_df.shape[1]-1,similarity_df.shape[2]), dtype = '<U255')

    for i in range(similarity_df.shape[2]):
        col_data = similarity_df[:, :, i]
        
        for j in range(col_data.shape[1]):
            _, normality_p_value = stats.normaltest(col_data[:, j])
            normality_pvals[j,i] = normality_p_value
        
        for j in range(1, col_data.shape[1]):
            _, ttest_p_value = stats.ttest_ind(col_data[:, 0], col_data[:, j])
            test_pvals[j-1,i] = ttest_p_value
            
            if ttest_p_value<0.05:
                p_str = '*'
            if ttest_p_value<0.01:
                p_str = '**'
            if ttest_p_value<0.001:
                p_str = '***'
            if ttest_p_value<0.0001:
                p_str = '****'
            if ttest_p_value>0.05:
                p_str = 'ns'
            test_symbols[j-1,i] = p_str
    return test_symbols

def get_pvals_list(similarity_list):
    m,n = len(similarity_list), len(similarity_list[0])
    
    normality_pvals = np.zeros((n,m))
    test_pvals = np.ones((n-1,m))
    test_symbols = np.empty((n-1,m), dtype = '<U255')

    for i, col_data in enumerate(similarity_list): 
        for j in range(n):
            _, normality_p_value = stats.normaltest(col_data[j])
            normality_pvals[j,i] = normality_p_value
        
        for j in range(n-1):
            _, ttest_p_value = stats.ttest_ind(col_data[-1], col_data[j])
            test_pvals[j-1,i] = ttest_p_value
            
            if ttest_p_value<0.05:
                p_str = '*'
            if ttest_p_value<0.01:
                p_str = '**'
            if ttest_p_value<0.001:
                p_str = '***'
            if ttest_p_value<0.0001:
                p_str = '****'
            if ttest_p_value>0.05:
                p_str = 'ns'
            test_symbols[j-1,i] = p_str
    return normality_pvals, test_pvals, test_symbols

def plot_heatmap(df,name,xtick,ytick,xlabel = 'population range', ylabel='family structure', quantity = 'ratio matrix similarity>0.4'):
    width = 2*df.shape[1]
    height = 2*df.shape[0] if df.shape[0]<5 else 1.5*df.shape[0]
        
    fig = plt.figure(figsize=(width, height))
    ax = sns.heatmap(df, cmap='viridis', annot=True, fmt=".2f", linewidths=1)
    cbar = ax.collections[0].colorbar
    cbar.set_label(quantity, size=18, labelpad=15)
    
    plt.xticks(np.arange(df.shape[1])+0.5,xtick)
    plt.yticks(np.arange(df.shape[0])+0.5,ytick)
    plt.xlabel(xlabel, size=18, labelpad=11)
    plt.ylabel(ylabel, size=18, labelpad=11)
    if name:
        plt.title(name)
    return fig

def plot_bar3d(df, xtick, ytick, xlabel = 'population range', ylabel='family structure', quantity = 'ratio matrix similarity>0.4', pval = None, yoff=45, force_zpos = None):
    fig = plt.figure(figsize=(10,10))
    rect = [0.1, 0.1, 0.75, 0.75]
    ax = fig.add_subplot(rect, projection='3d')

    xpos, ypos = np.meshgrid(np.arange(len(xtick)), np.arange(len(ytick)), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    if force_zpos:
        zpos = force_zpos
    else:
        zpos = max(np.min(df)-0.05,0)

    dx = dy = 0.8
    dz = df.T.ravel()-zpos

    pll= sns.color_palette()
    colors = pll[:len(ytick)]
    bar_colors = [colors[i] for _ in range(len(xtick)) for i in range(len(ytick))]
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=bar_colors, alpha=0.75)
    
    
    
    if pval is not None:
        for i in range(len(xtick)):
            for j in range(len(ytick)-1):
                ax.text(i + 0.5, j + 0.5, dz[i * len(ytick) + j]+zpos+0.003, pval[i,j], fontsize=12, ha='center', va='center')

    # # 设置坐标轴标签
    ax.set_xticks(np.arange(len(xtick)) + 0.5,xtick, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(ytick)) + 0.5,ytick, ha='left')
    ax.set_xlabel(xlabel, size=18, labelpad=50)
    ax.set_ylabel(ylabel, size=18, labelpad=yoff)
    ax.set_zlabel(quantity, size=18, labelpad=25)
    
    original_ticks = ax.get_zticks()
    original_labels = ["{:.2f}".format(num) for num in original_ticks]
    ax.set_zticks(original_ticks, original_labels, ha='left')
    
    # set figure size
    ax2 = fig.add_axes([0, 0, 1, 1], frameon=False)
    print(ax.get_position())
    ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax2.set_xticks([])  # 清除 x 轴刻度
    ax2.set_yticks([])  # 清除 y 轴刻度
    ax2.spines['right'].set_visible(True)  # 显示右边框

    plt.savefig('tight.png', bbox_inches='tight')
    plt.show()
    
    return fig

def plot_line(df, xtick, ytick, xlabel = 'population range', ylabel='pedigree centrality range', quantity = 'ratio matrix similarity>0.4', pval = None):
    fig = plt.figure(figsize=(10, 6)) # line plot
    sns.set_style("ticks")

    ax = sns.lineplot(data = df.T, dashes=False, linewidth=2, markers=True, markersize=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(xtick) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(quantity)
    ax.legend(handles, ytick, title = ylabel)

    if pval is not None:
        for i in range(pval.shape[1]):
            for j in range(pval.shape[0]):
                ax.text(i, df[j,i], pval[j, i], ha='center', va='bottom')
                ax.text(i, df[j,i], pval[j, i], ha='center', va='bottom')
    
    return fig

def extract_paragraph_after_size(input_string):
    size_index = input_string.find("size")

    if size_index != -1:
        numeric_index = size_index
        while numeric_index > 0 and input_string[numeric_index - 1].isdigit():
            numeric_index -= 1

        result_paragraph = input_string[numeric_index:]

        return result_paragraph

def summarize_trees(tree_array, skip = False, force1 = False):

    # 创建存储PDF文件的文件夹
    pdf_folder = "temp_pdf"
    os.makedirs(pdf_folder, exist_ok=True)
    
    # 调用R脚本并生成图形
    if not skip:
        if force1:
            file_path=tree_array
            # 调用R脚本并传递文件路径
            # print(file_path)
            subprocess.run(["Rscript", "./plot-fam-slave.R", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            for row in tree_array:
                for file_path in row:
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")
                    
                    if not os.path.exists(pdf_path):
                        subprocess.run(["Rscript", "./plot-fam-slave.R", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
    # 创建阵列图
    if force1:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")
        if os.path.exists(pdf_path):
            images = convert_from_path(pdf_path)
            img = images[0]
            fig = plt.figure(figsize = (12,8), dpi=300)
            plt.imshow(img)
            plt.axis('off')
            plt.title(extract_paragraph_after_size(file_name)[:-10], fontsize=12)
            return fig
    else:
        m = tree_array.shape[0]
        n = tree_array.shape[1]
        fig, axs = plt.subplots(m, n, figsize=(2.5*n, 1.8*m) ,dpi=1200)

        # 显示每个生成的图形
        for i in range(m):
            for j in range(n):
                # 获取不带扩展名的文件名
                file_name = os.path.splitext(os.path.basename(tree_array[i, j]))[0]
                pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")

                # 检查文件是否存在
                try:
                    if os.path.exists(pdf_path):
                        images = convert_from_path(pdf_path)
                        img = images[0]
                        axs[i, j].imshow(img)
                        axs[i, j].axis('off')
                        axs[i, j].set_title(extract_paragraph_after_size(file_name)[:-10], fontsize=8//m)
                except Exception as e:
                    file_path = tree_array[i,j]
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")

                    subprocess.run(["Rscript", "./plot-fam-slave.R", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        if os.path.exists(pdf_path):
                            images = convert_from_path(pdf_path)
                            img = images[0]
                            axs[i, j].imshow(img)
                            axs[i, j].axis('off')
                            axs[i, j].set_title(extract_paragraph_after_size(file_name)[:-10], fontsize=8//m)
                    except Exception as e:
                        print('fuck!',e)

        plt.show()
        return fig

def summarize_trees_simple(tree_list, skip = False, force_titles = []):

    # 创建存储PDF文件的文件夹
    pdf_folder = "temp_pdf"
    os.makedirs(pdf_folder, exist_ok=True)
    
    # 调用R脚本并生成图形
    if not skip:
        for file_path in tree_list:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")
            
            if not os.path.exists(pdf_path):
                subprocess.run(["Rscript", "./plot-fam-slave.R", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
    # 创建阵列图
    m = -((-len(tree_list))//4)
    n = 4
    fig, axs = plt.subplots(m, n, figsize=(4*n, 3*m) ,dpi=1200)

    axs = axs.flatten()
    
    for ax in axs:
        ax.axis('off')
    
    for dum in range(len(tree_list)):
        file_name = os.path.splitext(os.path.basename(tree_list[dum]))[0]
        pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")

        # 检查文件是否存在
        try:
            images = convert_from_path(pdf_path)
            img = images[0]
            axs[dum].imshow(img)
            axs[dum].axis('off')
            title = force_titles[dum] if force_titles else extract_paragraph_after_size(file_name)[:-10]
            axs[dum].set_title(title, fontsize=7)
        except Exception as e:
            file_path = tree_list[dum]
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            pdf_path = os.path.join(pdf_folder, f"{file_name}_plot.pdf")

            subprocess.run(["Rscript", "./plot-fam-slave.R", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                images = convert_from_path(pdf_path)
                img = images[0]
                axs[dum].imshow(img)
                axs[dum].axis('off')
                title = force_titles[dum] if force_titles else extract_paragraph_after_size(file_name)[:-10]
                axs[dum].set_title(title, fontsize=8//m)
            except Exception as e:
                print('fuck!',e)

    plt.show()
    return fig

def plot_bar_label_max(df, xtick, xlabel = 'population range', ylabel='pedigree centrality range', quantity = 'ratio matrix similarity>0.4'
                       , test_method = 'ttest'):

    plt.rcParams.update({'pdf.fonttype': 42})
    plt.rcParams.update({'ps.fonttype': 42})

    fig = plt.figure(figsize=(7,5))
    sns.barplot(df, linewidth = 2, edgecolor='black',width = 0.8, color = sns.color_palette()[0])
    plt.xticks(np.arange(len(xtick)),xtick, rotation=45, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(quantity)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', color='black', alpha=0.3)
    
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.rcParams.update({'ps.fonttype': 42})
    
    # calculate pvalues with peak
    means = np.squeeze(np.mean(df,axis= 0))
    plt.ylim((np.min(means)-0.1),np.max(means)+0.05)
    
    peak_pos = np.argmax(means)
    for j in range(df.shape[1]):
        if j != peak_pos:
            if test_method == "ttest":
                tval, p_value = stats.ttest_ind(df[:,peak_pos], df[:,j])
                symbol = 't'
            elif test_method == "mlu":
                tval, p_value = stats.mannwhitneyu(df[:,peak_pos], df[:,j])
                symbol = 'u'
            if p_value<0.05:
                p_str = '*'
            if p_value<0.01:
                p_str = '**'
            if p_value<0.001:
                p_str = '***'
            if p_value<0.0001:
                p_str = '****'
            if p_value>0.05:
                p_str = 'ns'
            plt.text(j, means[j]+0.01, p_str, ha='center', va='bottom')
            
            print(f"{symbol} = {tval}, p = {p_value}, n{j},n{peak_pos} = {len(df[:,peak_pos])},{len(df[:,j])}")

    plt.show()
    
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.rcParams.update({'ps.fonttype': 42})
    return fig


### for the artificial validation of individual number estimation
### rearrange data, calculate posterior probabilities, plot

def unwrap_nested_list(nested_list):
    # 如果当前元素是列表，递归调用函数
    if isinstance(nested_list, list):
        # 确保列表只有一个元素
        if len(nested_list) != 1:
            raise ValueError("The nested list must contain exactly one element.")
        return unwrap_nested_list(nested_list[0])
    # 如果当前元素是整数，返回它
    elif isinstance(nested_list, int):
        return nested_list

def my_ix(*args):
    new_args = []
    for arg in args:
        if isinstance(arg,int):
            new_args.append([arg])
        else:
            new_args.append(arg)
    return np.ix_(*new_args)

def get_population_probabilities(ref_sizes,ref_pos,argmax_dists,title = '',multiple = False, write_txt = '../plots/test.txt', reps=400):
    assert len(ref_sizes) == argmax_dists.shape[0]
    labels = [f'>{ref_size[0]}' for ref_size in ref_sizes[1:]]
    
    size_lens = np.array([thesize[1]-thesize[0] for thesize in ref_sizes])
    
    for _ in range(len(argmax_dists.shape)-1):
        size_lens = size_lens[:, np.newaxis]
    
    if multiple:
        probs = [np.sum(size_lens[i:,:] * argmax_dists[my_ix(range(i, argmax_dists.shape[0]), *ref_pos)])
             / np.sum(size_lens * argmax_dists[my_ix(range(argmax_dists.shape[0]), *ref_pos)]) for i in range(1, argmax_dists.shape[0])]
        
        with open(write_txt,'w') as datfile:
            datfile.write('Simulated individual number range\t')
            datfile.write('Number simulated pedigrees satisfying peak ranges Nps & Npp in current individual number range\t')
            datfile.write('Number simulated pedigrees satisfying peak ranges Nps & Npp in all individual number ranges\t')
            datfile.write('Length of current range\t')
            datfile.write('Posterior probability for N being in current range\t')
            datfile.write('Posterior probability for N exceeding current range\n')
            for i in range(1, argmax_dists.shape[0]+1):
                datfile.write(f'{ref_sizes[i-1]}\t')
                datfile.write(f'{reps*np.sum(argmax_dists[my_ix(i-1, *ref_pos)])}\t')
                datfile.write(f'{reps*np.sum(argmax_dists[my_ix(range(argmax_dists.shape[0]), *ref_pos)])}\t')
                datfile.write(f'{unwrap_nested_list(size_lens[i-1])}\t')
                prob = np.sum(size_lens[i-1,:] * argmax_dists[my_ix(i-1, *ref_pos)]) / np.sum(size_lens * argmax_dists[my_ix(range(argmax_dists.shape[0]), *ref_pos)])
                acc_prob = np.sum(size_lens[i:,:] * argmax_dists[my_ix(range(i, argmax_dists.shape[0]), *ref_pos)]) / np.sum(size_lens * argmax_dists[my_ix(range(argmax_dists.shape[0]), *ref_pos)])
                datfile.write(f'{prob}\t{acc_prob}\n')
                
    else:
        size_lens = np.array([thesize[1]-thesize[0] for thesize in ref_sizes])[:,np.newaxis] 
        ind = ref_pos if not isinstance(ref_pos, int) else [ref_pos]
        probs = [np.sum(size_lens[i:,:]*argmax_dists[i:,ind])/np.sum(size_lens*argmax_dists[:,ind]) for i in range(1,argmax_dists.shape[0])]
        
        with open(write_txt,'w') as datfile:
            datfile.write('Simulated individual number range\t')
            datfile.write('Number simulated pedigrees satisfying peak ranges Nps & Npp in current individual number range\t')
            datfile.write('Number simulated pedigrees satisfying peak ranges Nps & Npp in all individual number ranges\t')
            datfile.write('Length of current range\t')
            datfile.write('Posterior probability for N being in current range\t')
            datfile.write('Posterior probability for N exceeding current range\n')
            for i in range(1, argmax_dists.shape[0]+1):
                datfile.write(f'{ref_sizes[i-1]}\t')
                datfile.write(f'{reps*np.sum(argmax_dists[i-1,ind])}\t')
                datfile.write(f'{reps*np.sum(argmax_dists[i:,ind])}\t')
                datfile.write(f'{unwrap_nested_list(size_lens[i-1])}\t')
                prob = np.sum(size_lens[i-1,:] * argmax_dists[i-1,ind]) / np.sum(size_lens*argmax_dists[:,ind])
                acc_prob = np.sum(size_lens[i:,:]*argmax_dists[i:,ind])/np.sum(size_lens*argmax_dists[:,ind])
                datfile.write(f'{prob}\t{acc_prob}\n')
    
    fig = plt.figure(figsize=(7,5))
    sns.barplot(probs, linewidth = 2, edgecolor='black',width = 0.8)
    plt.xticks(np.arange(len(labels)),labels, rotation=45, ha='right')
    plt.xlabel('Predicted population range')
    plt.ylabel('Probability')
    plt.title(title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', color='black', alpha=0.3)

    plt.show()
    
    return labels,probs,fig

