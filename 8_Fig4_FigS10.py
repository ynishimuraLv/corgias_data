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
## Download COG2020 information
# !https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-20.def.tab

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

df_tp = {}
for lin in lineages:
    df_tp[lin] = {}
    for th in thresholds:
        df_tp[lin][th] = pl.read_csv(f'{lin}/tp_stats{th}.csv')

# %%
df = {}
df_weighted = {}
df_transition = {}

th = 0.7
for lin in lineages:
    df[lin] = df_tp[lin][th]
    sev =  pl.read_csv(f'{lin}/sev_MP.csv')
    sev = sev.with_columns(
        larger = pl.max_horizontal('OG1', 'OG2'),
        smaller = pl.min_horizontal('OG1', 'OG2')
    ).select(pl.concat_str(pl.col('larger'), pl.col('smaller'), separator='_').alias('COG_pair'),
             (((pl.col('num_change1') - pl.col('num_change2')).abs())).alias('diff_change'))
    df[lin] = df[lin].join(sev, on='COG_pair')
    df_weighted[lin] = df[lin].filter(
                       (pl.col('naive')) | (pl.col('rle')) | (pl.col('cwa')) | (pl.col('asa'))
                       )
    df_transition[lin] = df[lin].filter( (pl.col('cotr')) | (pl.col('sev')) ) 

# %%
both = {}
only_weighted = {}
only_transition = {}
for lin in lineages:
    both[lin] = df_weighted[lin].filter(pl.col('COG_pair').is_in(df_transition[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)
    only_weighted[lin] = df_weighted[lin].filter(~pl.col('COG_pair').is_in(both[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)
    only_transition[lin] = df_transition[lin].filter(~pl.col('COG_pair').is_in(both[lin]['COG_pair'])).filter(pl.col('coeff') >= 0)


# %%
def extract_category(both, weighted, transition, category):
    both_category = both.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    weighted_category = weighted.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    transition_category = transition.filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    
    return both_category, weighted_category, transition_category

cog_class = {
    'A':'RNA PROCESSING AND MODIFICATION',
    'B':'CHROMATIN STRUCTURE AND DYNAMICS',
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
    'Y':'Nuclear structure',
    'Z':'Cytoskeleton'
    }

# %%
# Fig. 4AB

fig = plt.figure(figsize=(8.27, 11.69/3))
gs = GridSpec(3, 11, figure=fig)
ax1 = fig.add_subplot(gs[0, :5]) 
ax2 = fig.add_subplot(gs[1, :5])  
ax3 = fig.add_subplot(gs[2, :5])
ax4 = fig.add_subplot(gs[:, 6:])
axes = [ax1, ax2, ax3]

for i, lin in enumerate(lineages):
    if i == 0:
        sns.kdeplot(only_transition[lin].filter(pl.col('coeff') >= 0), x="coeff",  
                    fill=True, alpha=0.5, ax=axes[i], label='Only detected by transtion')
        sns.kdeplot(only_weighted[lin].filter(pl.col('coeff') >= 0), x="coeff", 
                    fill=True, alpha=0.5, ax=axes[i], label='Only detected by weighted')
    else:
        sns.kdeplot(only_transition[lin].filter(pl.col('coeff') >= 0), x="coeff",  
                    fill=True, alpha=0.5, ax=axes[i])
        sns.kdeplot(only_weighted[lin].filter(pl.col('coeff') >= 0), x="coeff", 
                    fill=True, alpha=0.5, ax=axes[i])        

    axes[i].set_xlim([0, 5])
    axes[i].set_ylabel(lin, fontsize=8)
axes[0].legend(ncol=2, loc="upper right", bbox_to_anchor=(1.05, 1.4),
               fontsize = 7)

for i in [0, 1]:
    axes[i].set_xlabel("")
    axes[i].set_xticklabels("")
    axes[i].tick_params(left=False, labelleft=False, labelsize=6)

ax3.set_xlabel("coefficient", fontsize=10)
ax3.tick_params(left=False, labelleft=False, labelsize=8)

category = 'N'
lin = 'mycobacteriales'
both_category = both[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
weighted_category = only_weighted[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
transition_category = only_transition[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))

ax4.scatter(transition_category.select('coeff'),
            transition_category.select('diff_change'),
            label='Transition only', s=15, alpha=0.5)
#ax4.scatter(weighted_category.select('coeff'),
#            weighted_category.select((pl.col('t1_sim') - pl.col('t2_sim')).abs()),
#            label='Weighted only', s=15)
ax4.scatter(both_category.select('coeff'),
            both_category.select('diff_change'),
            label='Both', s=15, c='green', alpha=0.5)


ax4.set_xlabel('Coefficient', fontsize=10)
ax4.set_ylabel('Difference between t1 and t2', fontsize=10)
ax4.set_title('mycobacteriales', fontsize=10)
ax4.legend(fontsize=8, ncol=2)
ax4.set_ylim([0, 100])
ax4.grid()

# plt.savefig('Fig3AB.png', dpi=300)

# %%
methods = ['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev']
df_sorted = {}
for lin in lineages:
    df_sorted[lin] = {}
    for meth in methods:
        df_sorted[lin][meth] = df[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True)

# %%
df_scored = {}
for lin in lineages:
    df_scored[lin] = {}
    for meth in methods:
        df_scored[lin][meth] = df_sorted[lin][meth].filter(
                                  ~pl.col('score').is_null()
                               ).with_columns(
                                    pl.when(pl.col('score') >= 900).then(True).otherwise(False).alias('th09'),
                               ).with_row_index('rank')
        
for lin in lineages:
    for meth in methods:
        df_scored[lin][meth] = df_scored[lin][meth].with_columns(
                                 (pl.col('th09').cum_sum() / (pl.col('rank') + 1)).alias('ppv09'), 
                               )

# %%
df_ppv07 = {}
for lin in lineages:
    df_ppv07[lin] = {}
    for meth in methods:
        th = df_scored[lin][meth].filter(pl.col('ppv09') >= 0.7)[-1]['rank']
        df_ppv07[lin][meth] = df_scored[lin][meth].filter(pl.col('rank') <= th)

tp_sets = {}
for lin in lineages:
    tp_sets[lin] = {}
    for meth in methods:
        tp_sets[lin][meth] = df_ppv07[lin][meth].filter(pl.col('th09')).select('COG_pair')
        
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
for lin in lineages:
    df_tp[lin].loc[:, 'COG_pair'] = df_tp[lin].index
    df_tp[lin] = pl.DataFrame(df_tp[lin])
    df_tp[lin] = df_tp[lin].with_columns(pl.col('COG_pair').str.split('_').list[0].alias('COG1'),
                     pl.col('COG_pair').str.split('_').list[1].alias('COG2'))
    df_tp[lin] = df_tp[lin].join(cog, left_on='COG1', right_on='COG').rename(
         {'category':'category1', 'annot':'annot1', 'pathway':'pathway1'}).join(
         cog, left_on = 'COG2', right_on = 'COG').rename(
         {'category':'category2', 'annot':'annot2', 'pathway':'pathway2'})


# %%
for lin in lineages:
    evolccm = pl.read_csv(f'{lin}/evolCCM_tpr07.csv').transpose()
    evolccm = evolccm.rename({'column_0':'COG1', 'column_1':'COG2', 'column_2':'alpha1',
                          'column_3':'alpha2', 'column_4':'beta1', 'column_5':'beta2', 
                          'column_6':'coeff', 'column_7':'pvalue'})
    evolccm = evolccm.with_columns(
                larger = pl.max_horizontal('COG1', 'COG2'),
                smaller = pl.min_horizontal('COG1', 'COG2')
        ).select(pl.concat_str(pl.col('larger'), pl.col('smaller'), separator='_').alias('COG_pair'),
                 'alpha1', 'alpha2', 'beta1', 'beta2', 'coeff', 'pvalue'
                )
    df_tp[lin] = df_tp[lin].join(evolccm, on='COG_pair')

# %%
for lin in lineages:
    print(df_tp[lin].shape)

# %%
for lin in lineages:
    df_tp[lin].write_csv(f'{lin}/tpr07_stat.csv')


# %%
def extract_category(both, weighted, transition, category):
    both_category = both.filter(pl.col('category1').str.contains(category) & pl.col('category2').str.contains(category))
    weighted_category = weighted.filter(pl.col('category1').str.contains(category) & pl.col('category2').str.contains(category))
    transition_category = transition.filter(pl.col('category1').str.contains(category) & pl.col('category2').str.contains(category))
    
    return both_category, weighted_category, transition_category


# %%
schema = {'naive':pl.Boolean, 'rle':pl.Boolean, 'cwa':pl.Boolean,
          'asa':pl.Boolean, 'cotr':pl.Boolean, 'sev':pl.Boolean,
          'COG_pair':pl.String, 'COG1':pl.String, 'COG2':pl.String,
          'category1':pl.String, 'annot1':pl.String, 'pathway1':pl.String,
          'category2':pl.String, 'annot2':pl.String, 'pathway2':pl.String,
          'alpha1':pl.Float64, 'alpha2':pl.Float64, 'beta1':pl.Float64, 'beta2':pl.Float64, 
          'coeff':pl.Float64, 'pvalue':pl.Float64}

# %%
fig = plt.figure(figsize=(8.27, 11.69/3))
gs = GridSpec(3, 11, figure=fig)
ax1 = fig.add_subplot(gs[0, :5]) 
ax2 = fig.add_subplot(gs[1, :5])  
ax3 = fig.add_subplot(gs[2, :5])
ax4 = fig.add_subplot(gs[:, 6:])
axes = [ax1, ax2, ax3]

for i, lin in enumerate(lineages):
    if i == 0:
        sns.kdeplot(only_transition[lin].filter(pl.col('coeff') >= 0), x="coeff",  
                    fill=True, alpha=0.5, ax=axes[i], label='Only detected by transtion')
        sns.kdeplot(only_weighted[lin].filter(pl.col('coeff') >= 0), x="coeff", 
                    fill=True, alpha=0.5, ax=axes[i], label='Only detected by weighted')
    else:
        sns.kdeplot(only_transition[lin].filter(pl.col('coeff') >= 0), x="coeff",  
                    fill=True, alpha=0.5, ax=axes[i])
        sns.kdeplot(only_weighted[lin].filter(pl.col('coeff') >= 0), x="coeff", 
                    fill=True, alpha=0.5, ax=axes[i])        

    axes[i].set_xlim([0, 5])
    axes[i].set_ylabel(lin, fontsize=8)
axes[0].legend(ncol=2, loc="upper right", bbox_to_anchor=(1.05, 1.4),
               fontsize = 7)

for i in [0, 1]:
    axes[i].set_xlabel("")
    axes[i].set_xticklabels("")
    axes[i].tick_params(left=False, labelleft=False, labelsize=6)

ax3.set_xlabel("coefficient", fontsize=10)
ax3.tick_params(left=False, labelleft=False, labelsize=8)

category = 'N'
lin = 'mycobacteriales'
both_category = both[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
weighted_category = only_weighted[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
transition_category = only_transition[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))

ax4.scatter(transition_category.select('coeff'),
            transition_category.select('diff_change'),
            label='Transition only', s=15, alpha=0.5)
#ax4.scatter(weighted_category.select('coeff'),
#            weighted_category.select((pl.col('t1_sim') - pl.col('t2_sim')).abs()),
#            label='Weighted only', s=15)
ax4.scatter(both_category.select('coeff'),
            both_category.select('diff_change'),
            label='Both', s=15, c='green', alpha=0.5)


ax4.set_xlabel('Coefficient', fontsize=10)
ax4.set_ylabel('Difference between t1 and t2', fontsize=10)
ax4.set_title('mycobacteriales', fontsize=10)
ax4.legend(fontsize=8, ncol=2)
ax4.set_ylim([0, 100])
ax4.grid()

plt.savefig('Fig3AB.png', dpi=300)

# %%
node_both = set(both_category['COG1'].to_list()).union(set(both_category['COG2'].to_list()))
node_transition = set(transition_category['COG1'].to_list()).union(set(transition_category['COG2'].to_list()))

node_transition = node_transition.difference(node_both)
len(node_both), len(node_transition)

# %%
# Fig. 4C

tree = Phylo.read('mycobacteriales/hq_tree.tre', format='newick')
order = [node.name for node in tree.get_terminals()]
table = pd.read_csv('mycobacteriales/COG_table99.csv', index_col=['Unnamed: 0'])
table = table.loc[order]

maxdist = max([tree.distance(tree.root, x) for x in tree.get_nonterminals()])
fig = plt.figure(figsize=(20, 40))

ax1=plt.subplot2grid((1,60), (0, 10), colspan=14)
a=ax1.imshow(table.astype(float).loc[:, node_transition], cmap=plt.cm.Blues,
             vmin=0, vmax=1,
             aspect='auto',
             interpolation='none',
            )
ax1.set_yticks([])
ax1.set_xticks([])
ax1.axis('off')

ax2=plt.subplot2grid((1,60), (0, 30), colspan=20)
a=ax2.imshow(table.astype(float).loc[:, node_both], cmap=plt.cm.Blues,
             vmin=0, vmax=1,
             aspect='auto',
             interpolation='none',
            )
ax2.set_yticks([])
ax2.set_xticks([])
ax2.axis('off')

ax=plt.subplot2grid((1,60), (0, 0), colspan=10, facecolor='white')
ax.axis('off')

fig.subplots_adjust(wspace=0, hspace=0)


Phylo.draw(tree, axes=ax,
           show_confidence=False,
           label_func = lambda x: None,
           xticks = ([], ), yticks = ([], ),
           xlabel = ('',), ylabel = ('',),
           xlim = (-0.01, maxdist+0.01),
           axis = ('off', ),
           )
#plt.savefig('Fig4C.png', dpi=300)

# %%
# Fig. S10

network_transition = nx.from_pandas_edgelist(transition_category, source='COG1', target='COG2')
g = nx.from_pandas_edgelist(both_category.extend(transition_category), source='COG1', target='COG2')
colors = []
for edge in g.edges:
    if edge in network_transition.edges:
        colors.append('steelblue')
    else:
        colors.append('limegreen')

pos = nx.circular_layout(g)

nx.draw(g, node_size=30, edge_color=colors,
       pos=pos)

label_pos = {n: (x*1.1, y*1.1) for n, (x, y) in pos.items()}
nx.draw_networkx_labels(g, label_pos, labels={node: str(node) for node in g.nodes()}, font_size=6)

# plt.savefig('FigS10.png')

# %%
# Fig S1

a4_width = 8.27
aspect = 18/40

fig, ax = plt.subplots(1, 3, figsize = (8.27, 12), sharey='all')

for i, lin in enumerate(lineages):
    offset = [ i for i in range(1, 47, 2)]
    width = 0.5
    categories = [category for category in cog_class.keys() if category not in 'BYZ']
    for j, category in zip(offset, categories):
        both_category, weighted_category, sim_category = \
           extract_category(both[lin], only_weighted[lin], only_transition[lin], category)
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
plt.savefig('FigS1.png', dpi=300)

# %%
# Fig S2
fig, ax = plt.subplots(4, 7, figsize=(35, 20))

lin = 'pseudomonadales'
for i, category in enumerate(cog_class.keys()):
    both_category = both[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    weighted_category = only_weighted[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    sev_category = only_transition[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    j = i // 7
    k = i%7
    flag = 0
    if sev_category.shape[0] >= 1:
        ax[j, k].scatter(sev_category.select('coeff'),
                         sev_category.select('diff_change'),
                         label=sim_category.shape[0], s=10, c='blue')
        flag = 1
    if weighted_category.shape[0] >= 1:
        ax[j, k].scatter(weighted_category.select('coeff'),
                         weighted_category.select('diff_change'),
                         label=weighted_category.shape[0], s=10, c='orange')
        flag = 1
    if both_category.shape[0] >= 1:
        ax[j, k].scatter(both_category.select('coeff'),
                         both_category.select('diff_change'),
                         alpha=0.2, label=both_category.shape[0], s=10, c='green')
        flag = 1
    ax[j, k].set_title(category, fontsize=20)
    ax[j, k].grid()
    if flag == 1:
        ax[j, k].legend(fontsize=10)
        
plt.savefig('FigS2.png', dpi=300)
plt.show()

# %%
# Fig S3

fig, ax = plt.subplots(4, 7, figsize=(35, 20))

lin = 'mycobacteriales'
for i, category in enumerate(cog_class.keys()):
    both_category = both[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    weighted_category = only_weighted[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    sev_category = only_transition[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    j = i // 7
    k = i%7
    flag = 0
    if sev_category.shape[0] >= 1:
        ax[j, k].scatter(sev_category.select('coeff'),
                         sev_category.select('diff_change'),
                         label=sim_category.shape[0], s=10, c='blue')
        flag = 1
    if weighted_category.shape[0] >= 1:
        ax[j, k].scatter(weighted_category.select('coeff'),
                         weighted_category.select('diff_change'),
                         label=weighted_category.shape[0], s=10, c='orange')
        flag = 1
    if both_category.shape[0] >= 1:
        ax[j, k].scatter(both_category.select('coeff'),
                         both_category.select('diff_change'),
                         alpha=0.2, label=both_category.shape[0], s=10, c='green')
        flag = 1
    ax[j, k].set_title(category, fontsize=20)
    ax[j, k].grid()
    if flag == 1:
        ax[j, k].legend(fontsize=10)
        
plt.savefig('FigS3.png', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(4, 7, figsize=(35, 20))

lin = 'archaea'
for i, category in enumerate(cog_class.keys()):
    both_category = both[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    weighted_category = only_weighted[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    sev_category = only_transition[lin].filter(pl.col('category1').str.contains(category) | pl.col('category2').str.contains(category))
    j = i // 7
    k = i%7
    flag = 0
    if sev_category.shape[0] >= 1:
        ax[j, k].scatter(sev_category.select('coeff'),
                         sev_category.select('diff_change'),
                         label=sim_category.shape[0], s=10, c='blue')
        flag = 1
    if weighted_category.shape[0] >= 1:
        ax[j, k].scatter(weighted_category.select('coeff'),
                         weighted_category.select('diff_change'),
                         label=weighted_category.shape[0], s=10, c='orange')
        flag = 1
    if both_category.shape[0] >= 1:
        ax[j, k].scatter(both_category.select('coeff'),
                         both_category.select('diff_change'),
                         alpha=0.2, label=both_category.shape[0], s=10, c='green')
        flag = 1
    ax[j, k].set_title(category, fontsize=20)
    ax[j, k].grid()
    if flag == 1:
        ax[j, k].legend(fontsize=10)
        
plt.savefig('FigS4.png', dpi=300)
plt.show()

# %%
