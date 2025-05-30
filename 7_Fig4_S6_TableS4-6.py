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
import os,sys,shutil
import numpy as np
import pandas as pd
import polars as pl
import ete3 as et
import seaborn as sns
from scipy.stats import brunnermunzel
from itertools import permutations
from matplotlib import pyplot as plt


# %%
def divide_tree_by_ancestral_state(tree):
    result = {}
    for node in tree.traverse(strategy='postorder'):
        node.state = node.trait
        if node.is_leaf():
            node.connected = {node}
        else:
            node.connected = set()
            if node.state in '0123':
                for child in node.get_children():
                    if node.state == child.state:
                        node.connected |= child.connected
                    else:
                        if child.connected:
                            result[child] = child.connected
            else:
                for child in node.get_children():
                    if child.connected:
                        result[child] = child.connected
                       
    if tree.connected:
        result[tree] = tree.connected
                       
    return result


# %%
def count_lca_transition(tree):
    result = { f'{state[0]}->{state[1]}':0 for state in all_direction }
    for key, value in divide_tree_by_ancestral_state(tree).items():
        if not key.is_root():
            parent_trait = key.get_ancestors()[0].trait
            if '|' in parent_trait:
                parent_trait = '?'
            result[f'{parent_trait}->{key.trait}'] += 1
    
    return result


# %%
## Ancestral state reconstruction of the detected pairs
# %%bash
for dir in pseudomonadales mycobacteriales archaea; 
do
(
    cd "$dir" &&  mkdir pastml_out &&
    ls pastML_tpr07/ | sed -E 's@^pastml_in/Pa_@@; s/\.csv//' | xargs -I@ -P 12 pastml -t hq_tree.tre -d pastML_tpr07/@.csv -s , --threads 4 --work_dir pastml_out/@ 
)
done

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
schema = {'naive':pl.Boolean, 'rle':pl.Boolean, 'cwa':pl.Boolean,
          'asa':pl.Boolean, 'cotr':pl.Boolean, 'sev':pl.Boolean,
          'COG_pair':pl.String, 'COG1':pl.String, 'COG2':pl.String,
          'category1':pl.String, 'annot1':pl.String, 'pathway1':pl.String,
          'category2':pl.String, 'annot2':pl.String, 'pathway2':pl.String,
          'alpha1':pl.Float64, 'alpha2':pl.Float64, 'beta1':pl.Float64, 'beta2':pl.Float64, 
          'coeff':pl.Float64, 'pvalue':pl.Float64}

df = {}
df_weighted = {}
df_transition = {}
both = {}
only_weighted = {}
only_transition = {}
for lin in lineages:
    df[lin] = pl.read_csv(f'{lin}/tpr07_stat.csv', schema=schema).fill_null(False)
    df_weighted[lin] = df[lin].filter(
                       (pl.col('naive')) | (pl.col('rle')) | (pl.col('cwa')) | (pl.col('asa'))
                       )
    df_transition[lin] = df[lin].filter( (pl.col('cotr')) | (pl.col('sev')) ) 

    both[lin] = df_weighted[lin].filter(pl.col('COG_pair').is_in(df_transition[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)
    only_weighted[lin] = df_weighted[lin].filter(~pl.col('COG_pair').is_in(both[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)
    only_transition[lin] = df_transition[lin].filter(~pl.col('COG_pair').is_in(both[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)

# %%
only_transition[lin].head()

# %%
only_weighted[lin].head()

# %%
all_direction = list(permutations(['0', '1', '2', '3', '?'], 2))

# %%
count_weighted = {}
for lin in lineages:
    flag = 0
    for row in only_weighted[lin].iter_rows():
        pair = row[6]
        tree = et.Tree(f'{lin}/pastml_out/{pair}/named.tree_hq_tree.nwk', format = 1)
        if flag == 0:
            count_weighted[lin] = pd.Series(count_lca_transition(tree), name = pair)
            flag = 1
        else:
            tmp = pd.Series(count_lca_transition(tree), name = pair)
            count_weighted[lin] = pd.concat([count_weighted[lin], tmp], axis = 1)

# %%
count_transition = {}
for lin in lineages:
    flag = 0
    for row in only_transition[lin].iter_rows():
        pair = row[6]
        tree = et.Tree(f'{lin}/pastml_out/{pair}/named.tree_hq_tree.nwk', format = 1)
        if flag == 0:
            count_transition[lin] = pd.Series(count_lca_transition(tree), name = pair)
            flag = 1
        else:
            tmp = pd.Series(count_lca_transition(tree), name = pair)
            count_transition[lin] = pd.concat([count_transition[lin], tmp], axis = 1)

# %%
fig, axes = plt.subplots(1, 3, figsize=(8.27, 8.27 / 3)) 

for i, lin in enumerate(lineages):
    transit = count_weighted[lin]
    weighted_ordered = transit.loc['0->3'] + transit.loc['3->0'] + transit.loc['1->3'] + transit.loc['3->1'] + \
              transit.loc['2->1'] + transit.loc['1->2'] + transit.loc['0->2'] + transit.loc['2->0']
    transit = count_transition[lin]
    transition_ordered = transit.loc['0->3'] + transit.loc['3->0'] + transit.loc['1->3'] + transit.loc['3->1'] + \
              transit.loc['2->1'] + transit.loc['1->2'] + transit.loc['0->2'] + transit.loc['2->0']
    parts = axes[i].violinplot([transition_ordered, weighted_ordered], 
                              showmeans=False, showextrema=False, showmedians=False)
    
    jitter_weighted = 2 + np.random.normal(0, 0.07, size=weighted_ordered.shape[0])
    jitter_transition = 1 + np.random.normal(0, 0.07, size=transition_ordered.shape[0])
    if i == 0:
        axes[i].scatter(jitter_transition, transition_ordered, alpha = 0.2, s=5, label='Transition only')
        axes[i].scatter(jitter_weighted, weighted_ordered, alpha = 0.5, s=5, label='Weighted only')
    else:
        axes[i].scatter(jitter_transition, transition_ordered, alpha = 0.2, s=5)
        axes[i].scatter(jitter_weighted, weighted_ordered, alpha = 0.5, s=5)
    axes[i].set_title(lin, fontsize=10)
    axes[i].tick_params(axis='both', which='major', labelsize=5)  
    axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[0].set_ylabel('Transition from/to intermediate state', fontsize=8)
fig.legend(loc="lower center", ncol=3, fontsize=6, markerscale=2)
plt.savefig('FigS6.png', dpi=300)

# %%
fig, axes = plt.subplots(1, 3, figsize=(8.27, ((10 / 3)))) 

for i, lin in enumerate(lineages):
    transit = count_weighted[lin]
    direction_1 = transit.loc['0->3'] + transit.loc['3->0'] + transit.loc['1->3'] + transit.loc['3->1']
    direction_2 = transit.loc['2->1'] + transit.loc['1->2'] + transit.loc['0->2'] + transit.loc['2->0']
    tmp1 = pd.concat([direction_1, direction_2], axis=1)
    tmp1.loc[:, 'ratio'] = tmp1.max(axis=1) / (tmp1.loc[:, 0] + tmp1.loc[:, 1]) 
    
    transit = count_transition[lin]
    direction_1 = transit.loc['0->3'] + transit.loc['3->0'] + transit.loc['1->3'] + transit.loc['3->1']
    direction_2 = transit.loc['2->1'] + transit.loc['1->2'] + transit.loc['0->2'] + transit.loc['2->0']
    tmp2 = pd.concat([direction_1, direction_2], axis=1)
    tmp2.loc[:, 'ratio'] = tmp2.max(axis=1) / (tmp2.loc[:, 0] + tmp2.loc[:, 1])  
    
    axes[i].scatter(tmp2.max(axis=1), tmp2['ratio'], alpha = 0.4 ,s=10, label='Transition only')
    axes[i].scatter(tmp1.max(axis=1), tmp1['ratio'], alpha = 0.6, s=10, label='Weighted only')
    axes[i].set_title(lin, fontsize=10)
    axes[i].legend(fontsize=6)
    
    if i in [1, 2]:
        axes[i].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False)

axes[0].set_ylabel('Transition bias', fontsize=8)
axes[1].set_xlabel('Transition from/to intermediate state', fontsize=8)
plt.tight_layout()

plt.savefig('Fig4.png', dpi=300)

# %%
cog_annot = { line.split('\t')[0]:line.split('\t')[2] for line 
             in open('cog-20.def.tab', encoding='cp1252')}


# %%
def count_state_change(tree):
    result = {}
    for node in tree.traverse(strategy='postorder'):
        node.state = getattr(node, 'trait')
        if node.is_leaf():
            node.connected = {node}
        else:
            node.connected = set()
            if not '|' in node.state:
                for child in node.get_children():
                    if node.state == child.state:
                        node.connected |= child.connected
                    else:
                        if child.connected:
                            direction = f'{node.state}->{child.state}'
                            result.setdefault(node, []).append((direction, child.connected))
    return result


# %%
direction1 = ['2->1', '2->0', '1->2', '0->2']
direction2 = ['3->1', '3->1', '1->3', '0->3']
og1_gain_loss = ['1->3', '3->1', '2->0', '0->2']
og2_gain_loss = ['1->2', '2->1', '0->3', '3->0']

for i, lin in enumerate(lineages):
    with open(f'{lin}/TableS{i+4}.tsv', encoding='utf-8', mode='w') as outfile:
        outfile.write('\t'.join(['COG1', 'COG2', 'COG1_annot', 'COG2_annot', 'from/to only COG1 present',
                                 'from/to only COG2 present', 'COG1 gain/loss', 'COG2 gain/loss', 'transition bias']))
        outfile.write('\n')
        for row in only_weighted[lin].iter_rows():
            pair = row[6]
            tree = et.Tree(f'{lin}/pastml_out/{pair}/named.tree_hq_tree.nwk', format = 1)
            flag = 0
            dir1 = 0
            dir2 = 0
            og1_change = 0
            og2_change = 0
            for kye, value in count_state_change(tree).items():
                for transition in value:
                    direction = transition[0]
                    if direction in direction1:
                        dir1 += 1
                    elif direction in direction2:
                        dir2 += 1
                    if direction in og1_gain_loss:
                        og1_change += 1
                    elif direction in og2_gain_loss:
                        og2_change += 1
            cog1, cog2 = pair.split('_')
            line = '\t'.join([cog1, cog2, cog_annot[cog1], cog_annot[cog2], 
                             str(dir1), str(dir2), str(og1_change), str(og2_change),
                             str(max(dir1, dir2) / (dir1 + dir2))])
            outfile.write(line)
            outfile.write('\n')

# %%
