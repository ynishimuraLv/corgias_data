# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import subprocess
import numpy as np
import pandas as pd
import polars as pl
from itertools import combinations
from scipy import stats
from multiprocessing import Pool
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import ete3 as et

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
all_methods = ['naive', 'rle', 'cwa', 'asa', 'asawo', 'cotr', 'sev',
           'PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance']
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
df = {}
df_sorted = {}
df_ppv = {}
for lin in lineages:
    df[lin] = pl.read_csv(f'{lin}/scaled_pvalues.csv').rename({'asa_ML':'asa', 'sev_MP':'sev'})
    df_sorted[lin] = {}
    df_ppv[lin] = {}
    for meth in all_methods:
        df_sorted[lin][meth] = df[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True).with_row_index('rank')

    for meth in all_methods:
        df_sorted[lin][meth] = df_sorted[lin][meth].with_columns(
                                 (pl.col('truth').cum_sum() / (pl.col('rank') + 1)).alias('ppv')
                               )
        df_ppv[lin][meth] = {}
        for th in thresholds:
            tp = df_sorted[lin][meth].filter(pl.col('ppv') >= th)
            if not tp.is_empty():
                rank = tp[-1]['rank']
                df_ppv[lin][meth][th] = df_sorted[lin][meth].filter(pl.col('rank') <= rank)


# %%
def eval_pvalues(values, vector):
    fpr, tpr, thresholds = roc_curve(values, vector)
    pre, rec, thresholds = precision_recall_curve(values, vector)
    roc_auc = auc(fpr, tpr)
    rec_pre_auc = auc(rec, pre)
    
    return fpr, tpr, pre, rec, roc_auc, rec_pre_auc


# %%
num_tp = {}
for lin in lineages:
    num_tp[lin] = []
    for meth in all_methods:
        for th in thresholds:
            tp = df_sorted[lin][meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))
            if not tp.is_empty():
                num_tp[lin].append([th, meth, tp.shape[0]])
            else:
                num_tp[lin].append([th, meth, 0])
    num_tp[lin] = pd.DataFrame(num_tp[lin])
    num_tp[lin].rename(columns={0:'th', 1:'method', 2:'TP'}, inplace=True)
    num_tp[lin] = num_tp[lin].pivot(index="th", columns="method", values="TP")
    num_tp[lin] = num_tp[lin].loc[thresholds, all_methods]

# %%
# Fig. 2a

cmap = plt.get_cmap('Set3', 12)
corgias_methods = ['naive', 'rle',  'cwa', 'asa', 'cotr', 'sev', 'asawo']
fig, ax = plt.subplots(1, 3, figsize = (8.27, 2.5))
for i, lin in enumerate(lineages):
    label_dict = {}
    for j, meth in enumerate(corgias_methods):
        if meth == 'asawo':
            label = 'asa-'
        else:
            label = meth
        result = eval_pvalues(df_sorted[lin][meth]['truth'], df_sorted[lin][meth][meth])
        score = result[5]
        lable = f'{label}({score:.3f})'
        label_dict[meth] = lable
        ax[i].plot(result[3], result[2],  label = lable,
                    linewidth = 1, color = cmap([j]))
    
    ax[i].set_title(lin, fontsize=8)
    ax[i].set_xlabel('Recall', fontsize=7)
    ax[i].tick_params(axis='both', labelsize=4)
    ax[i].grid(linewidth=0.5)
    if i != 0:
        ax[i].set_ylabel('')
    handles, labels = ax[i].get_legend_handles_labels()
    order = [labels.index(label_dict[meth]) for meth in ['naive', 'rle', 'cwa', 'asa', 'asawo', 'cotr', 'sev']]
    ax[i].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=5.5)

fig.savefig('Fig2a.svg', dpi=300)

# %%
# Fig. 2b
fig, axes = plt.subplots(3, 1, figsize=(8.27, 7))
fig.subplots_adjust(hspace=0.4)
cmap = plt.get_cmap('Set3', 12)
corgias_methods = ['naive', 'rle',  'cwa', 'asa', 'cotr', 'sev']

for i, lin in enumerate(lineages):
    bar_width = 0.25
    x = np.arange(len(num_tp[lin].index))
    for j, sub_cat in enumerate(corgias_methods):
        axes[i].bar(x*2 + j * bar_width, num_tp[lin][sub_cat], width=bar_width, label=sub_cat, color=cmap([j]))
        axes[i].legend(ncol=2)
        axes[i].set_title(lin)
        axes[i].set_xticks(x*2 + bar_width*2.5)
        axes[i].set_ylabel('No. positive pairs')
        axes[i].set_xticklabels(num_tp[lin].index)
        if i == 2:
            axes[i].set_xticklabels(num_tp[lin].index)
            axes[i].set_xlabel('True Positive Rate')
        else:
            axes[i].set_xticklabels("")
fig.savefig('Fig2b.svg', dpi=300)            

# %%
coverage = {}
count = {}
for lin in lineages:
    tp_sets = df_sorted[lin]
    coverage[lin] = []
    count[lin] = []
    for th in thresholds:
        detected_all = set()
        weighted = set()
        transition = set()
        for meth in corgias_methods:
            detected_all |= set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
            if meth in ['naive', 'rle', 'cwa', 'asa']:
                weighted |= set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
            else:
                transition |= set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
        coverage[lin].append(['weighted', th, len(weighted) / len(detected_all)])
        count[lin].append(['weighted', th, len(weighted)])
        coverage[lin].append(['transition', th, len(transition) / len(detected_all)])
        count[lin].append(['transition', th, len(transition)])
        count[lin].append(['total', th, len(detected_all)])
        for meth in corgias_methods:
            tp =  set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
            coverage[lin].append([meth, th, len(tp) / len(detected_all)])
            count[lin].append([meth, th, len(tp)])

    
    coverage[lin] = pd.DataFrame(coverage[lin])
    coverage[lin].rename(columns={0:'method', 1:'th', 2:'coverage'}, inplace=True)
    coverage[lin] = coverage[lin].pivot(index="th", columns="method", values="coverage")
    coverage[lin] = coverage[lin].loc[thresholds, corgias_methods + ['weighted', 'transition']]

    count[lin] = pd.DataFrame(count[lin])
    count[lin].rename(columns={0:'method', 1:'th', 2:'count'}, inplace=True)
    count[lin] = count[lin].pivot(index="th", columns="method", values="count")
    count[lin] = count[lin].loc[thresholds, corgias_methods + ['weighted', 'transition', 'total']]

# %%
# Table S2-S4
count

# %%
# Table S2-S4
coverage

# %%
count = {}
coverage = {}
for lin in lineages:
    tp_sets = df_sorted[lin]
    count[lin] = []
    coverage[lin] = []
    for th in thresholds:
        detected_all = set()
        for meth1 in corgias_methods:
            others = set()
            detected_all |= set(tp_sets[meth1].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
            for meth2 in corgias_methods:
                if meth1 == meth2:
                    tp =  set(tp_sets[meth1].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
                else:
                    others |= set(tp_sets[meth2].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
            only = tp.difference(others)
            count[lin].append([meth1, th, len(only)])
            coverage[lin].append([meth1, th, len(only) / len(detected_all)])

    for th in thresholds:
       weighted = set()
       transtion = set()
       detected_all = set()
       for meth in corgias_methods:
           detected_all |=  set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
           if meth in ['naive', 'rle', 'cwa', 'asa']:
               weighted |= set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
           else:
               transition |=  set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
       only_weighted = weighted.difference(transition)
       only_transition = transition.difference(weighted)
       count[lin].append(['weighted', th, len(only_weighted)])
       coverage[lin].append(['weighted', th, len(only_weighted) / len(detected_all)])
       count[lin].append(['transition', th, len(only_transition)])
       coverage[lin].append(['transition', th, len(only_transition) / len(detected_all)])        

# %%
for lin in lineages:
    coverage[lin] = pd.DataFrame(coverage[lin])
    coverage[lin].rename(columns={0:'method', 1:'th', 2:'coverage'}, inplace=True)
    coverage[lin] = coverage[lin].pivot(index="th", columns="method", values="coverage")
    coverage[lin] = coverage[lin].loc[thresholds, corgias_methods + ['weighted', 'transition']]
    
    count[lin] = pd.DataFrame(count[lin])
    count[lin].rename(columns={0:'method', 1:'th', 2:'count'}, inplace=True)
    count[lin] = count[lin].pivot(index="th", columns="method", values="count")
    count[lin] = count[lin].loc[thresholds, corgias_methods + ['weighted', 'transition']]

# %%
# Table S5-S7
coverage

# %%
# Table S5-S7
count

# %%
# TableS5-S7
coverage

# %%
fig, axes = plt.subplots(3, 1, figsize=(8.27, 7))
cmap = plt.get_cmap('Set3', 12)
fig.subplots_adjust(hspace=0.3)
for i, lin in enumerate(lineages):
    bar_width = 0.3
    print(lin, type(count[lin]))
    x = np.arange(len(count[lin].index))
    for j, sub_cat in enumerate(corgias_methods):
        axes[i].bar(x*2 + j * bar_width, count[lin][sub_cat], width=bar_width, label=sub_cat, color=cmap([j]))
        axes[i].legend(fontsize=6, loc='upper left')
        axes[i].set_title(lin)
        axes[i].set_xticks(x*2 + bar_width*2.5)
        axes[i].set_ylabel('No. unique pairs')
        axes[i].set_xticklabels(count[lin].index)
        if i == 2:
            axes[i].set_xticklabels(count[lin].index)
            axes[i].set_xlabel('True Positive Rate')
        else:
            axes[i].set_xticklabels("")
fig.savefig('FigS2.png', dpi=300)

# %%
th = 0.5
for lin in lineages:
    tp_sets = df_sorted[lin]
    detected_all = set()
    for meth in corgias_methods:
        detected_all |= set(tp_sets[meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))['COG_pair'])
        
    with open(f'{lin}/tp_pairs.txt', encoding='utf-8', mode='w') as outfile:
        for pair in detected_all:
            outfile.write(pair + '\n')

# %%
for lin in lineages:
    if not os.path.exists(f'{lin}/pastML_tp/'):
        os.mkdir(f'{lin}/pastML_tp/')
    cog = pl.read_csv(f'{lin}/COG_table99.csv')
    cog = cog.rename({'':'genome'})
    for pair in open(f'{lin}/tp_pairs.txt'):
        cog1, cog2 = pair.strip().split('_')
        tmp = cog.select('genome', cog1, cog2)
        tmp = tmp.with_columns(
            pl.when((pl.col(cog1) == 1) & (pl.col(cog2) == 1)).then(pl.lit('1'))
            .when((pl.col(cog1) == 1) & (pl.col(cog2) == 0)).then(pl.lit('2'))
            .when((pl.col(cog1) == 0) & (pl.col(cog2) == 1)).then(pl.lit('3'))
            .when((pl.col(cog1) == 0) & (pl.col(cog2) == 0)).then(pl.lit('0'))
            .otherwise(pl.lit('4')).alias('trait')
        )
        tmp.select('genome', 'trait').write_csv(f'{lin}/pastML_tp/{cog1}_{cog2}.csv')

# %%
