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
methods = ['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev']
df = {}
df_sorted = {}
for lin in lineages:
    df[lin] = pl.read_csv(f'{lin}/scaled_pvalues.csv').rename({'asa_ML':'asa', 'sev_MP':'sev'})
    df_sorted[lin] = {}
    for meth in methods:
        df_sorted[lin][meth] = df[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True).with_row_index('rank')

for lin in lineages:
    for meth in methods:
        df_sorted[lin][meth] = df_sorted[lin][meth].with_columns(
                                 (pl.col('truth').cum_sum() / (pl.col('rank') + 1)).alias('ppv')
                               )

# %%
list(df_sorted[lin].keys())

# %%
df_ppv07 = {}
for lin in lineages:
    df_ppv07[lin] = {}
    for meth in methods:
        th = df_sorted[lin][meth].filter(pl.col('ppv') >= 0.7)[-1]['rank']
        df_ppv07[lin][meth] = df_sorted[lin][meth].filter(pl.col('rank') <= th)


# %%
def eval_pvalues(values, vector):
    fpr, tpr, thresholds = roc_curve(values, vector)
    pre, rec, thresholds = precision_recall_curve(values, vector)
    roc_auc = auc(fpr, tpr)
    rec_pre_auc = auc(rec, pre)
    
    return fpr, tpr, pre, rec, roc_auc, rec_pre_auc


# %%
order = ['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev']
fig, ax = plt.subplots(1, 3, figsize = (8.27, 2.5))
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
    ax[i].legend(fontsize=5.5)
    ax[i].grid(linewidth=0.5)
plt.savefig('Fig2_1.png', dpi=300)

# %%
df_ppv07[lin][meth].head()

# %%
tp_sets = {}
for lin in lineages:
    tp_sets[lin] = {}
    for meth in methods:
        tp_sets[lin][meth] = df_ppv07[lin][meth].filter(pl.col('truth') == 1).select('COG_pair')

# %%
for lin in lineages:
    for meth in methods:
        print(lin, meth, tp_sets[lin][meth].shape)

# %%
lin = 'pseudomonadales'
naive = tp_sets[lin]['naive']['COG_pair']
naive = pd.Series(index = naive, data = [True] * len(naive))
rle = tp_sets[lin]['rle']['COG_pair']
rle = pd.Series(index = rle, data = [True] * len(rle))
cwa = tp_sets[lin]['cwa']['COG_pair']
cwa = pd.Series(index = cwa, data = [True] * len(cwa))
asa = tp_sets[lin]['asa']['COG_pair']
asa = pd.Series(index = asa, data = [True] * len(asa))
cotr = tp_sets[lin]['cotr']['COG_pair']
cotr = pd.Series(index = cotr, data = [True] * len(cotr))
sim = tp_sets[lin]['sev']['COG_pair']
sim = pd.Series(index = sim, data = [True] * len(sim))

upset_data = pd.concat((naive, rle, cwa, asa, cotr, sim), axis=1).fillna(False)
upset_data.rename(columns={0:'naive', 1:'rle', 2:'cwa', 3:'asa', 4:'cotr', 5:'sev'}, inplace = True)
upset_data = upset_data.set_index(list(upset_data.columns))
upsetplot.plot(upset_data, subset_size="count", show_counts="%d", sort_by="cardinality")

# %%
naivesets = {
    'RLE': set(tp_sets[lin]['rle'].to_series()),
    'ASA': set(tp_sets[lin]['asa'].to_series()),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'weighted': set(tp_sets[lin]['rle'].to_series()).union(set(tp_sets[lin]['cwa'].to_series())).union(set(tp_sets[lin]['asa'].to_series()).union(set(tp_sets[lin]['naive'].to_series()))),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'naive': set(tp_sets[lin]['naive'].to_series()),
    'rle': set(tp_sets[lin]['rle'].to_series()),
    'cwa': set(tp_sets[lin]['cwa'].to_series()),
    'asa': set(tp_sets[lin]['asa'].to_series())
}

venny4py(sets=naivesets)

# %%
lin = 'mycobacteriales'
naive = tp_sets[lin]['naive']['COG_pair']
naive = pd.Series(index = naive, data = [True] * len(naive))
rle = tp_sets[lin]['rle']['COG_pair']
rle = pd.Series(index = rle, data = [True] * len(rle))
cwa = tp_sets[lin]['cwa']['COG_pair']
cwa = pd.Series(index = cwa, data = [True] * len(cwa))
asa = tp_sets[lin]['asa']['COG_pair']
asa = pd.Series(index = asa, data = [True] * len(asa))
cotr = tp_sets[lin]['cotr']['COG_pair']
cotr = pd.Series(index = cotr, data = [True] * len(cotr))
sim = tp_sets[lin]['sev']['COG_pair']
sim = pd.Series(index = sim, data = [True] * len(sim))

upset_data = pd.concat((naive, rle, cwa, asa, cotr, sim), axis=1).fillna(False)
upset_data.rename(columns={0:'naive', 1:'rle', 2:'cwa', 3:'asa', 4:'cotr', 5:'sev'}, inplace = True)
upset_data = upset_data.set_index(list(upset_data.columns))
upsetplot.plot(upset_data, subset_size="count", show_counts="%d", sort_by="cardinality")

# %%
naivesets = {
    'RLE': set(tp_sets[lin]['rle'].to_series()),
    'ASA': set(tp_sets[lin]['asa'].to_series()),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'weighted': set(tp_sets[lin]['rle'].to_series()).union(set(tp_sets[lin]['cwa'].to_series())).union(set(tp_sets[lin]['asa'].to_series()).union(set(tp_sets[lin]['naive'].to_series()))),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'naive': set(tp_sets[lin]['naive'].to_series()),
    'rle': set(tp_sets[lin]['rle'].to_series()),
    'cwa': set(tp_sets[lin]['cwa'].to_series()),
    'asa': set(tp_sets[lin]['asa'].to_series())
}

venny4py(sets=naivesets)

# %%
lin = 'archaea'
naive = tp_sets[lin]['naive']['COG_pair']
naive = pd.Series(index = naive, data = [True] * len(naive))
rle = tp_sets[lin]['rle']['COG_pair']
rle = pd.Series(index = rle, data = [True] * len(rle))
cwa = tp_sets[lin]['cwa']['COG_pair']
cwa = pd.Series(index = cwa, data = [True] * len(cwa))
asa = tp_sets[lin]['asa']['COG_pair']
asa = pd.Series(index = asa, data = [True] * len(asa))
cotr = tp_sets[lin]['cotr']['COG_pair']
cotr = pd.Series(index = cotr, data = [True] * len(cotr))
sim = tp_sets[lin]['sev']['COG_pair']
sim = pd.Series(index = sim, data = [True] * len(sim))

upset_data = pd.concat((naive, rle, cwa, asa, cotr, sim), axis=1).fillna(False)
upset_data.rename(columns={0:'naive', 1:'rle', 2:'cwa', 3:'asa', 4:'cotr', 5:'sev'}, inplace = True)
upset_data = upset_data.set_index(list(upset_data.columns))
upsetplot.plot(upset_data, subset_size="count", show_counts="%d", sort_by="cardinality")

# %%
naivesets = {
    'RLE': set(tp_sets[lin]['rle'].to_series()),
    'ASA': set(tp_sets[lin]['asa'].to_series()),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'naive': set(tp_sets[lin]['naive'].to_series()),
    'rle': set(tp_sets[lin]['rle'].to_series()),
    'cwa': set(tp_sets[lin]['cwa'].to_series()),
    'asa': set(tp_sets[lin]['asa'].to_series())
}

venny4py(sets=naivesets)

# %%
naivesets = {
    'weighted': set(tp_sets[lin]['rle'].to_series()).union(set(tp_sets[lin]['cwa'].to_series())).union(set(tp_sets[lin]['asa'].to_series()).union(set(tp_sets[lin]['naive'].to_series()))),
    'cotransition': set(tp_sets[lin]['cotr'].to_series()),
    'SEV': set(tp_sets[lin]['sev'].to_series())
}

venny4py(sets=naivesets)

# %%
df_tp = {}
for lin in lineages:
    df_tp[lin] = tp_sets[lin]['naive'].to_pandas()
    df_tp[lin].loc[:, 'naive'] = True
    df_tp[lin].set_index('COG_pair', inplace=True)
    for meth in methods[1:]:
        tmp = tp_sets[lin][meth].to_pandas()
        tmp.loc[:, meth] = True
        tmp.set_index('COG_pair', inplace=True)
        df_tp[lin] = df_tp[lin].join(tmp, how='outer')

# %%
for lin in df_tp.keys():
    print(df_tp[lin].shape)

# %%
for lin, df in df_tp.items():
    with open(f'{lin}/tpr07_pairs.txt', encoding='utf-8', mode='w') as outfile:
        for pair in df.index:
            outfile.write(pair + '\n')

# %%
for lin in ['pseudomonadales', 'mycobacteriales', 'archaea']:
    if not os.path.exists(f'{lin}/pastML_tpr07/'):
        os.mkdir(f'{lin}/pastML_tpr07/')
    cog = pl.read_csv(f'{lin}/COG_table99.csv')
    cog = cog.rename({'':'genome'})
    for pair in open(f'{lin}/tpr07_pairs.txt'):
        cog1, cog2 = pair.strip().split('_')
        tmp = cog.select('genome', cog1, cog2)
        tmp = tmp.with_columns(
            pl.when((pl.col(cog1) == 1) & (pl.col(cog2) == 1)).then(pl.lit('1'))
            .when((pl.col(cog1) == 1) & (pl.col(cog2) == 0)).then(pl.lit('2'))
            .when((pl.col(cog1) == 0) & (pl.col(cog2) == 1)).then(pl.lit('3'))
            .when((pl.col(cog1) == 0) & (pl.col(cog2) == 0)).then(pl.lit('0'))
            .otherwise(pl.lit('4')).alias('trait')
        )
        tmp.select('genome', 'trait').write_csv(f'{lin}/tpr_pairs/{cog1}_{cog2}.csv')

# %%
