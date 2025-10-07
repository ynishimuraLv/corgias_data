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
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
methods = ['naive', 'rle', 'cwa', 'asa_MPPA', 'asa_ACCTRAN', 'asa_DELTRAN', 'asa_DOWNPASS',
           'cotr', 'sev_ACCTRAN', 'sev_DELTRAN', 'sev_DOWNPASS']
df = {}
df_sorted = {}
for lin in lineages:
    df[lin] = pl.read_csv(f'{lin}/scaled_pvalues.csv')
    df[lin] = df[lin].rename({'asa_ML':'asa_MPPA', 'sev_MP':'sev_ACCTRAN'})
    df_sorted[lin] = {}
    for meth in methods:
        df_sorted[lin][meth] = df[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True).with_row_index('rank')

for lin in lineages:
    for meth in methods:
        df_sorted[lin][meth] = df_sorted[lin][meth].with_columns(
                                 (pl.col('truth').cum_sum() / (pl.col('rank') + 1)).alias('ppv')
                               )


# %%
def eval_pvalues(values, vector):
    fpr, tpr, thresholds = roc_curve(values, vector)
    pre, rec, thresholds = precision_recall_curve(values, vector)
    roc_auc = auc(fpr, tpr)
    rec_pre_auc = auc(rec, pre)
    
    return fpr, tpr, pre, rec, roc_auc, rec_pre_auc


# %%
order = ['asa_MPPA', 'asa_DOWNPASS', 'asa_ACCTRAN', 'asa_DELTRAN']
fig, ax = plt.subplots(1, 3, figsize = (10, 3.5))
for i, lin in enumerate(lineages):
    for meth in order:
        label = meth
        result = eval_pvalues(df_sorted[lin][meth]['truth'], df_sorted[lin][meth][meth])
        score = result[5]
        ax[i].plot(result[3], result[2],  label = f'{label}({score:.3f})',
                   linewidth = 1)
    ax[i].set_title(lin, fontsize=8)
    ax[i].set_xlabel('Recall', fontsize=7)
    ax[i].set_ylabel('Presicion', fontsize=7)
    ax[i].tick_params(axis='both', labelsize=4)
    ax[i].legend(fontsize=6)
    ax[i].grid(linewidth=0.5)

# %%
order = ['sev_DOWNPASS', 'sev_ACCTRAN', 'sev_DELTRAN']
fig, ax = plt.subplots(1, 3, figsize = (10, 3.5))
for i, lin in enumerate(lineages):
    for meth in order:
        label = meth
        result = eval_pvalues(df_sorted[lin][meth]['truth'], df_sorted[lin][meth][meth])
        score = result[5]
        ax[i].plot(result[3], result[2],  label = f'{label}({score:.3f})',
                   linewidth = 1)
    ax[i].set_title(lin, fontsize=8)
    ax[i].set_xlabel('Recall', fontsize=7)
    ax[i].set_ylabel('Presicion', fontsize=7)
    ax[i].tick_params(axis='both', labelsize=4)
    ax[i].legend(fontsize=6)
    ax[i].grid(linewidth=0.5)

# %%
ppv05 = {}
methods = ['naive', 'rle', 'cwa', 'asa_MPPA', 'cotr', 'sev_ACCTRAN']
top_pairs = {}
for lin in lineages:
    top_pairs[lin] = set()
    for meth in methods:
        tp = df_sorted[lin][meth].filter(pl.col('ppv') >= 0.5)
        rank = tp[-1]['rank']
        tp_pairs = tp.filter((pl.col('rank') <= rank))['COG_pair']
        top_pairs[lin] |= set(tp_pairs)

# %%
fig, ax = plt.subplots(3, 2, figsize = (8.27, 9))

for i, lin in enumerate(lineages):
    ax[i, 0].hist(df[lin].filter(pl.col('score') >= 0)['score'], range=(0, 1000), bins=20)
    count, _, _ = ax[i, 1].hist(df[lin].filter((pl.col('score') >= 0) & 
                                               (pl.col('COG_pair').is_in(top_pairs[lin])))['score'],
                                range=(0, 1000), bins=20)
    
    ax[i, 1].vlines(x=900, linestyle='--', ymin=0, ymax=max(count)*1.1, color='red')
    ax[i, 1].set_ylim(0, max(count)*1.1)
    ax[i, 0].set_title(f'{lin}: all scored pairs')
    ax[i, 1].set_title(f'{lin}: Top pairs (TPR ≥ 50%)')
    if i != 2:
        ax[i, 0].set_xticklabels('')
        ax[i, 1].set_xticklabels('')
    else:
        ax[i, 0].set_xlabel('STRING score')
        ax[i, 1].set_xlabel('STRING score')
fig.savefig('FigS1.png', dpi=300)

# %%
