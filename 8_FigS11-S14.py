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
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import networkx as nx
from collections import Counter
from scipy.stats import brunnermunzel
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib import lines as lines
from matplotlib.gridspec import GridSpec
from Bio import Phylo


# %%
def extract_category(both, weighted, transition, category):
    both_category = both.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    weighted_category = weighted.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    transition_category = transition.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    
    return both_category, weighted_category, transition_category


# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

df = {}
df_weighted = {}
df_transition = {}
sev = {}
for th in thresholds:
    df[th] = {}
    df_weighted[th] = {}
    df_transition[th] = {}
    for lin in lineages:
        df[th][lin] = pl.read_csv(f'{lin}/tp_stats{th}.csv').fill_null(False)
        if th == 0.9:
            sev[lin] =  pl.read_csv(f'{lin}/sev_MP.csv')
            sev[lin] = sev[lin].with_columns(
                            larger = pl.max_horizontal('OG1', 'OG2'),
                            smaller = pl.min_horizontal('OG1', 'OG2')
                       ).select(pl.concat_str(pl.col('larger'), pl.col('smaller'), separator='_').alias('COG_pair'),
                       (((pl.col('num_change1') - pl.col('num_change2')).abs())).alias('diff_change'))
            df[th][lin] = df[th][lin].join(sev[lin], on='COG_pair')
        df_weighted[th][lin] = df[th][lin].filter(
                           (pl.col('naive')) | (pl.col('rle')) | (pl.col('cwa')) | (pl.col('asa'))
                           )
        df_transition[th][lin] = df[th][lin].filter( (pl.col('cotr')) | (pl.col('sev')) ) 

# %%
both = {}
only_weighted = {}
only_transition = {}
for th in thresholds:
    both[th] = {}
    only_transition[th] = {}
    only_weighted[th] = {}
    for lin in lineages:
        both[th][lin] = df_weighted[th][lin].filter(pl.col('COG_pair').is_in(set(df_transition[th][lin]['COG_pair']))).filter(pl.col('coeff') >= 0)
        only_weighted[th][lin] = df_weighted[th][lin].filter(~pl.col('COG_pair').is_in(set(both[th][lin]['COG_pair']))).filter(pl.col('coeff') >= 0)
        only_transition[th][lin] = df_transition[th][lin].filter(~pl.col('COG_pair').is_in(set(both[th][lin]['COG_pair']))).filter(pl.col('coeff') >= 0)

# %%
cog_class = {
    'A':'RNA PROCESSING AND MODIFICATION',
#    'B':'CHROMATIN STRUCTURE AND DYNAMICS',
    'C':'Energy production and conversion',
    'D':'Cell cycle control, cell division, chromosome partitioning',
    'E':'Amino acid transport and metabolism',
    'F':'Nucleotide transport and metabolism',
    'G':'Carbohydrate transport and metabolism',
    'H':'Coenzyme transport and metabolism',
    'I':'Lipid transport and metabolism',
    'J':'TRANSLATION, RIBOSOMAL STRUCTURE AND BIOGENESIS',
    'K':'TRANSCRIPTION',
    'L':'REPLICATION, RECOMBINATION AND REPAIR',
    'M':'Cell wall/membrane/envelope biogenesis',
    'N':'Cell motility',
    'O':'Posttranslational modification, protein turnover, chaperones',
    'P':'Inorganic ion transport and metabolism',
    'Q':'Secondary metabolites biosynthesis, transport and catabolism',
    'R':'General function prediction only',
    'S':'Function unknown',
    'V':'Defense mechanisms',
    'T':'Signal transduction mechanisms',
    'U':'Intracellular trafficking, secretion, and vesicular transport',
    'W':'Extracellular structures',
    'X':'Mobilome: prophages, transposons',
#    'Y':'Nuclear structure',
#    'Z':'Cytoskeleton'
}

# %%
# Fig. S11

fig, ax = plt.subplots(5, 3, figsize=(15, 10), sharex='col')

for i, th in enumerate(thresholds):
    for j, lin in enumerate(lineages):
        if i == 0 and j == 1:
            ax[i, j].remove()
        else:
            sns.kdeplot(only_transition[th][lin].to_pandas(),
                        x = 'coeff', fill=True, alpha=0.5, ax=ax[i, j], label='only transition')
            sns.kdeplot(only_weighted[th][lin].to_pandas(),
                        x = 'coeff', fill=True, alpha=0.5, ax=ax[i, j], label='only weighted')
        ax[i, j].set_ylabel('')
        if i == 0:
            if j != 1:
                ax[i, j].set_title(lin)
            else:
                ax[1, j].set_title(lin)

pos = ax[0, 0].get_position()   # 隣の位置を基準に
# → 1つ右の列分シフトして、ほぼ同じサイズの領域を作る
legend_pos = [pos.x1 + 0.01, pos.y0, pos.width, pos.height]

ax_leg = fig.add_axes(legend_pos, frameon=False)  # 枠なしの Axes
ax_leg.set_xticks([]); ax_leg.set_yticks([])

# どこかのプロットから handles を取って legend 表示
handles, labels = ax[0, 0].get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", fontsize=12)

for i in range(5):
    # 右端のサブプロットの位置を取得
    pos = ax[i, 2].get_position()
    # 透明Axesを追加（右に少しスペースを空ける）
    ax_text = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.1, pos.height], frameon=False)
    ax_text.set_xticks([]); ax_text.set_yticks([])
    
    # 文字列表示
    ax_text.text(0.5, 0.5, f"TPR = {0.9-0.1*i}", ha='center', va='center', fontsize=15)

fig.savefig('FigS11', dpi=300)
plt.show()

# %%
# Fig. S12

fig = plt.figure(figsize=(5, 5))
th = 0.9
category = 'N'
lin = 'pseudomonadales'

both_category = both[th][lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
weighted_category = only_weighted[th][lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
transition_category = only_transition[th][lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))

plt.scatter(transition_category.select('coeff'),
            transition_category.select('diff_change'),
            label='Transition only', s=15, alpha=0.5)
#ax4.scatter(weighted_category.select('coeff'),
#            weighted_category.select((pl.col('t1_sim') - pl.col('t2_sim')).abs()),
#            label='Weighted only', s=15)
plt.scatter(both_category.select('coeff'),
            both_category.select('diff_change'),
            label='Both', s=15, c='green', alpha=0.5)
plt.legend()
plt.xlabel('Coefficient')
plt.ylabel('Difference between t1 and t2')
plt.savefig('FigS12.png', dpi=300)

# %%
th = 0.6

a4_width = 8.27
aspect = 18/40

fig, ax = plt.subplots(1, 3, figsize = (8.27, 12), sharey='all')

for i, lin in enumerate(lineages):
    offset = [ i for i in range(1, 47, 2)]
    width = 0.5
    categories = [category for category in cog_class.keys() if category not in 'BYZ']
    for j, category in zip(offset, categories):
        both_category, weighted_category, sim_category = \
           extract_category(both[th][lin], only_weighted[th][lin], only_transition[th][lin], category)
        jitter_both = j - 0.3 + np.random.normal(0, 0.05, size=both_category.shape[0])
        jitter_weighted = j + np.random.normal(0, 0.05, size=weighted_category.shape[0])
        jitter_sim = j + 0.3 + np.random.normal(0, 0.05, size=sim_category.shape[0])
        
        if category == 'X' and lin == 'archaea':
            ax[i].scatter(weighted_category['coeff'], jitter_weighted, alpha=0.6, s=2, color='orange', label='only weighted')
            ax[i].scatter(sim_category['coeff'],jitter_sim,  alpha=0.5, s=2, color='blue', label='only transition')
            ax[i].scatter(both_category['coeff'], jitter_both, alpha=0.2, s=2, color='green', label='both')
        else:
            ax[i].scatter(weighted_category['coeff'], jitter_weighted, alpha=0.6, s=2, color='orange')
            ax[i].scatter(sim_category['coeff'],jitter_sim,  alpha=0.5, s=2, color='blue')
            ax[i].scatter(both_category['coeff'], jitter_both, alpha=0.2, s=2, color='green')
    ax[i].invert_yaxis()
    ax[i].set_title(lin)
    ax[i].set_xlabel('coefficient')
ax[0].set_yticks([ i for i in range(1, 47, 2)], [i for i in categories],
                 fontsize=10)
fig.legend(loc="lower center", ncol=3, fontsize=12, markerscale=4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.075) 
plt.savefig('FigS13.png', dpi=300)

# %%
# Fig. S14

th = 0.5

a4_width = 8.27
aspect = 18/40

fig, ax = plt.subplots(1, 3, figsize = (8.27, 12), sharey='all')

for i, lin in enumerate(lineages):
    offset = [ i for i in range(1, 47, 2)]
    width = 0.5
    categories = [category for category in cog_class.keys() if category not in 'BYZ']
    for j, category in zip(offset, categories):
        both_category, weighted_category, sim_category = \
           extract_category(both[th][lin], only_weighted[th][lin], only_transition[th][lin], category)
        jitter_both = j - 0.3 + np.random.normal(0, 0.05, size=both_category.shape[0])
        jitter_weighted = j + np.random.normal(0, 0.05, size=weighted_category.shape[0])
        jitter_sim = j + 0.3 + np.random.normal(0, 0.05, size=sim_category.shape[0])
        
        if category == 'X' and lin == 'archaea':
            ax[i].scatter(weighted_category['coeff'], jitter_weighted, alpha=0.6, s=2, color='orange', label='only weighted')
            ax[i].scatter(sim_category['coeff'],jitter_sim,  alpha=0.5, s=2, color='blue', label='only transition')
            ax[i].scatter(both_category['coeff'], jitter_both, alpha=0.2, s=2, color='green', label='both')
        else:
            ax[i].scatter(weighted_category['coeff'], jitter_weighted, alpha=0.6, s=2, color='orange')
            ax[i].scatter(sim_category['coeff'],jitter_sim,  alpha=0.5, s=2, color='blue')
            ax[i].scatter(both_category['coeff'], jitter_both, alpha=0.2, s=2, color='green')
    ax[i].invert_yaxis()
    ax[i].set_title(lin)
    ax[i].set_xlabel('coefficient')
ax[0].set_yticks([ i for i in range(1, 47, 2)], [i for i in categories],
                 fontsize=10)
fig.legend(loc="lower center", ncol=3, fontsize=12, markerscale=4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.075) 
plt.savefig('FigS14.png', dpi=300)

# %%
