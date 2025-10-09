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
import os,shutil,sys
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']

results = {}
for lin in lineages:
    results[lin] = pl.read_csv(f'{lin}/scaled_pvalues.csv')
    results[lin] = results[lin].rename({'asa_ML':'asa', 'sev_MP':'sev'})

# %%
df_sorted = {}
df_ppv = {}
methods = ['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev',
           'PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance']
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
for lin in lineages:
    df_sorted[lin] = {}
    df_ppv[lin] = {}
    for meth in methods:
        df_sorted[lin][meth] = results[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True).with_row_index('rank')
        df_sorted[lin][meth] = df_sorted[lin][meth].with_columns(
                             (pl.col('truth').cum_sum() / (pl.col('rank') + 1)).alias('ppv')
                          )
        df_ppv[lin][meth] = {}
        for th in thresholds:
            tp = df_sorted[lin][meth].filter(pl.col('ppv') >= th)
            if tp.shape[0] >= 1:
                rank = tp[-1]['rank']
                df_ppv[lin][meth][th] = df_sorted[lin][meth].filter((pl.col('rank') <= rank) & (pl.col('truth') == 1) )


# %%
def eval_pvalues(values, vector):
    fpr, tpr, thresholds = roc_curve(values, vector)
    pre, rec, thresholds = precision_recall_curve(values, vector)
    roc_auc = auc(fpr, tpr)
    rec_pre_auc = auc(rec, pre)
    
    return fpr, tpr, pre, rec, roc_auc, rec_pre_auc


# %%
methods = ['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev',
           'PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance']

num_tp = {}
for lin in lineages:
    num_tp[lin] = []
    for meth in methods:
        for th in thresholds:
            tp = df_sorted[lin][meth].filter((pl.col('ppv') >= th) & (pl.col('truth') == 1))
            if not tp.is_empty():
                num_tp[lin].append([th, meth, tp.shape[0]])
            else:
                num_tp[lin].append([th, meth, 0])
    num_tp[lin] = pd.DataFrame(num_tp[lin])
    num_tp[lin].rename(columns={0:'th', 1:'method', 2:'TP'}, inplace=True)
    num_tp[lin] = num_tp[lin].pivot(index="th", columns="method", values="TP")
    num_tp[lin] = num_tp[lin].loc[thresholds, methods]

# %%
methods = ['PAJaccard', 'PAOverlap', 'GLMI', 'asa', 'GLDistance', 'sev']
replace = {'PAJaccard':'P/A Jaccard', 'PAOverlap':'P/A Overlap', 'GLMI':'G/L MI',
           'asa':'ASA', 'GLDistance':'G/L Distance', 'sev':'SEV'}

# %%
# Fig. S7

# A4サイズ（横: 8.27インチ, 縦: 11.69インチ）
fig = plt.figure(figsize=(8.27, 2.5)) 

# GridSpecで1行2列、横幅比率を [1, 2] に設定
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

# 左右のサブプロット
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

cmap = plt.get_cmap('Set3', 12)

lin = 'pseudomonadales'

label_dict = {}
for j, meth in enumerate(methods):
    label = meth
    test = df_sorted[lin][meth].filter(~pl.col('truth').is_null())
    result = eval_pvalues(test['truth'], test[meth])
    score = result[5]
    label = f'{replace[meth]}({score:.3f})'
    label_dict[meth] = label
    ax1.plot(result[3], result[2],  label = label,
                   linewidth = 1, color=cmap([j]))
ax1.set_title(lin, fontsize=8)
ax1.set_xlabel('Recall', fontsize=7)
ax1.set_ylabel('Presicion', fontsize=7)
ax1.tick_params(axis='both', labelsize=4)
ax1.legend(fontsize=5.5)
ax1.grid(linewidth=0.5)
handles, labels = ax1.get_legend_handles_labels()
order = [labels.index(label_dict[meth]) for meth in ['PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance', 'asa', 'sev']]
ax1.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=5.5)

ax1.set_xlabel('Recall', fontsize=7)
ax1.set_ylabel('Presicion', fontsize=7)
ax1.tick_params(axis='both', labelsize=4)
ax1.legend(fontsize=5.5)
ax1.grid(linewidth=0.5)
handles, labels = ax1.get_legend_handles_labels()
order = [labels.index(label_dict[meth]) for meth in ['PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance', 'asa', 'sev']]
ax1.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=5.5)

bar_width = 0.25
x = np.arange(len(num_tp[lin].index))
for j, sub_cat in enumerate(['PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance', 'asa', 'sev']):
    if sub_cat == 'GLDistance':
        c = j + 1
    elif sub_cat == 'asa':
        c = j - 1
    else:
        c = j
    ax2.bar(x*2 + j * bar_width, num_tp[lin][sub_cat], width=bar_width, 
                    label=replace[sub_cat], color=cmap([c]))
ax2.legend(ncol=1, fontsize=5.5)
ax2.set_xticks(x*2 + bar_width*2.5)
ax2.set_ylabel('No. positive pairs', fontsize=7)
ax2.set_xticklabels(num_tp[lin].index, fontsize=5.5)
ax2.set_yticklabels([0, 500, 1000, 1500, 2000, 2500], fontsize=5.5)
ax2.set_xlabel('True Positive Rate', fontsize=7)

fig.savefig('Fig7.png', dpi=300, bbox_inches="tight")

# %%
# Fig. S17


cmap = plt.get_cmap('Set3', 12)
fig, ax = plt.subplots(1, 3, figsize = (8.27, 2.5))
for i, lin in enumerate(lineages):
    label_dict = {}
    for j, meth in enumerate(methods):
        label = meth
        test = df_sorted[lin][meth].filter(~pl.col('truth').is_null())
        result = eval_pvalues(test['truth'], test[meth])
        score = result[5]
        label = f'{replace[meth]}({score:.3f})'
        label_dict[meth] = label
        ax[i].plot(result[3], result[2],  label = label,
                   linewidth = 1, color=cmap([j]))
    ax[i].set_title(lin, fontsize=8)
    ax[i].set_xlabel('Recall', fontsize=7)
    ax[i].set_ylabel('Presicion', fontsize=7)
    ax[i].tick_params(axis='both', labelsize=4)
    ax[i].legend(fontsize=5.5)
    ax[i].grid(linewidth=0.5)
    handles, labels = ax[i].get_legend_handles_labels()
    order = [labels.index(label_dict[meth]) for meth in ['PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance', 'asa', 'sev']]
    ax[i].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=5.5)
#plt.savefig('FigS17.png', dpi=300)

# %%
# Fig. S18
methods = ['PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance', 'asa', 'sev']
replace = {'PAJaccard':'P/A Jaccard', 'PAOverlap':'P/A Overlap', 'GLMI':'G/L MI',
           'asa':'ASA', 'GLDistance':'G/L Distance', 'sev':'SEV'}
fig, axes = plt.subplots(3, 1, figsize=(8.27, 7))
fig.subplots_adjust(hspace=0.4)
cmap = plt.get_cmap('Set3', 12)
for i, lin in enumerate(lineages):
    bar_width = 0.25
    x = np.arange(len(num_tp[lin].index))
    for j, sub_cat in enumerate(methods):
        if sub_cat == 'GLDistance':
            c = j + 1
        elif sub_cat == 'asa':
            c = j - 1
        else:
            c = j
        axes[i].bar(x*2 + j * bar_width, num_tp[lin][sub_cat], width=bar_width, 
                    label=replace[sub_cat], color=cmap([c]))
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
        axes[i].legend_.remove()
        
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", 
           fontsize=12, ncol=1, bbox_to_anchor=(1.12, 0.5))
fig.savefig('FigS18.png', dpi=300, bbox_inches="tight")

# %%
heatmaps = {}
for lin in lineages:
    heatmaps[lin] = []
    for th in thresholds:
        tmp = df_ppv[lin]
        sev = set(tmp['sev'][th]['COG_pair'])
        GLDistance = set(tmp['GLDistance'][th]['COG_pair'])
        GLDistance_sev = GLDistance.intersection(sev)
        only_GLDdistance = GLDistance.difference(sev)
        only_sev = sev.difference(GLDistance)
        subsets = [GLDistance_sev, only_GLDdistance, only_sev ]

        heatmaps[lin].append([ len(subset) for subset in subsets])
    heatmaps[lin] = pd.DataFrame(heatmaps[lin]).T
    heatmaps[lin] = heatmaps[lin].rename(
                      index = {0:'SEV & GLDistance', 1:'GLDistance only', 2:'SEV only'},
                      columns = {0:'0.9', 1:'0.8', 2:'0.7', 3:'0.6', 4:'0.5'})

# %%
heatmaps_ratio = {}
for lin in lineages:
    heatmaps_ratio[lin] = []
    for th in thresholds:
        tmp = df_ppv[lin]
        sev = set(tmp['sev'][th]['COG_pair'])
        GLDistance = set(tmp['GLDistance'][th]['COG_pair'])
        GLDistance_sev = GLDistance.intersection(sev)
        only_GLDdistance = GLDistance.difference(sev)
        only_sev = sev.difference(GLDistance)
        subsets = [GLDistance_sev, only_GLDdistance, only_sev ]
        total = 0
        for subset in subsets:
            total += len(subset)
        heatmaps_ratio[lin].append([ len(subset) /total for subset in subsets])
    heatmaps_ratio[lin] = pd.DataFrame(heatmaps_ratio[lin]).T
    heatmaps_ratio[lin] = heatmaps_ratio[lin].rename(
                      index = {0:'SEV & GLDistance', 1:'GLDistance only', 2:'SEV only'},
                      columns = {0:'0.9', 1:'0.8', 2:'0.7', 3:'0.6', 4:'0.5'})

# %%
# Fig. S19

fig, ax = plt.subplots(3, 2, figsize=(11, 10))
fig.subplots_adjust(hspace=0.3)
for i, lin in enumerate(lineages):
    sns.heatmap(
        heatmaps[lin], cmap="YlGnBu", annot=True, fmt="d",
        norm=LogNorm(vmin=max(1, heatmaps[lin].values.min()), vmax=heatmaps[lin].values.max()),
        cbar_kws={'label': 'count (log scale)'},
        linewidths=0.5, linecolor="gray", ax=ax[i, 0]
    )
    if lin == 'pseudomonas':
        taxonomy = 'pseudomonadales'
    else:
        taxonomy = lin
    ax[i, 0].set_title(taxonomy)
    ax[i, 0].set_xlabel('True Positive Rate')

for i, lin in enumerate(lineages):
    sns.heatmap(
        heatmaps_ratio[lin], cmap="YlGnBu", annot=True,
        cbar_kws={'label': 'ratio'},
        linewidths=0.5, linecolor="gray", ax=ax[i ,1]
    )
    if lin == 'pseudomonas':
        taxonomy = 'pseudomonadales'
    else:
        taxonomy = lin
    ax[i, 1].set_title(taxonomy)
    ax[i, 1].set_xlabel('True Positive Rate')

    if i != 2:
        ax[i, 0].set_xlabel('')
        ax[i, 1].set_xlabel('')
    ax[i, 1].set_yticklabels([])
fig.savefig('FigS19.png', dpi=300)
