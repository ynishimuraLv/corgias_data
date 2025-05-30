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
from collections import Counter
import upsetplot
from venny4py.venny4py import *

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
