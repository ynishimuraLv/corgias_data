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

# %%
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
methods = ['naive', 'rle', 'cwa', 'asa',  'cotr', 'sev']
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
df = {}
df_sorted = {}
df_ppv = {}
for lin in lineages:
    df[lin] = pl.read_csv(f'{lin}/scaled_pvalues.csv').rename({'asa_ML':'asa', 'sev_MP':'sev'})
    df_sorted[lin] = {}
    df_ppv[lin] = {}
    for meth in methods:
        df_sorted[lin][meth] = df[lin].filter(pl.col('score') >= 0).sort(by = meth, descending = True).with_row_index('rank')

    for meth in methods:
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
cog = pl.read_csv('cog-20.def.tab', encoding='cp1252', 
                  separator='\t', has_header=False)
cog = cog.rename({'column_1':'COG', 'column_2':'category', 'column_3':'annot', 'column_5':'pathway'})
cog = cog.select('COG', 'category', 'annot', 'pathway')


evolccm = {}
for lin in lineages:
    evolccm[lin] = pl.read_csv(f'{lin}/evolCCM.csv').transpose()[1:]
    evolccm[lin] = evolccm[lin].rename({'column_0':'COG1', 'column_1':'COG2', 'column_2':'alpha1',
                              'column_3':'alpha2', 'column_4':'beta1', 'column_5':'beta2', 
                              'column_6':'coeff', 'column_7':'pvalue'})
    evolccm[lin]= evolccm[lin].with_columns(
                larger = pl.max_horizontal('COG1', 'COG2'),
                smaller = pl.min_horizontal('COG1', 'COG2')
              ).select(pl.concat_str(pl.col('larger'), pl.col('smaller'), separator='_').alias('COG_pair'),
                    'COG1', 'COG2', pl.col('alpha1').cast(pl.Float64), 
                    pl.col('alpha2').cast(pl.Float64),  pl.col('beta1').cast(pl.Float64), 
                    pl.col('beta2').cast(pl.Float64), pl.col('coeff').cast(pl.Float64), 
                    pl.col('pvalue').cast(pl.Float64)
              )
    evolccm[lin] = evolccm[lin].join(cog, left_on='COG1', right_on='COG').rename(
                   {'category':'category1', 'annot':'annot1', 'pathway':'pathway1'}).join(
                   cog, left_on = 'COG2', right_on = 'COG').rename(
                   {'category':'category2', 'annot':'annot2', 'pathway':'pathway2'})

# %%
df_tp = {}
for th in thresholds:
    df_tp[th] = {}
    for lin in lineages:
        index = df_ppv[lin]['naive'][th].filter(pl.col('truth') == 1)['COG_pair']
        df_tp[th][lin] = pd.DataFrame(index=index)
        df_tp[th][lin].loc[:, 'naive'] = True
        df_tp[th][lin].rename_axis('COG_pair', inplace=True)
        for meth in methods[1:]:
            if th in df_ppv[lin][meth]:
                index = df_ppv[lin][meth][th].filter(pl.col('truth') == 1)['COG_pair']
                tmp = pd.DataFrame(index=index)
                tmp.loc[:, meth] = True
                tmp.rename_axis('COG_pair', inplace=True)
                df_tp[th][lin] = df_tp[th][lin].join(tmp, how='outer').fillna(False)
            else:
                print(key, meth, th)

# %%
for lin in lineages:
    for th in thresholds:
        df_tp[th][lin] = df_tp[th][lin].join(evolccm[lin].to_pandas().set_index('COG_pair'), on='COG_pair')
        df_tp[th][lin].to_csv(f'{lin}/tp_stats{th}.csv')

# %%
for lin in lineages:
    for th in thresholds:
        df_tp[th][lin] = pl.read_csv(f'{lin}/tp_stats{th}.csv')

# %%
# Fig. 3
fig, ax = plt.subplots(2, 1, figsize = (8.27, 6))
fig.subplots_adjust(hspace=0.3)
lin = 'archaea'
for j, th in enumerate(thresholds):
    tp = df_tp[th][lin]

    cotr = tp.filter((pl.col('cotr') == True) & (pl.col('coeff') > 0))
    sev = tp.filter((pl.col('sev') == True) & (pl.col('coeff') > 0))
    
    only_cotr = cotr.filter(~pl.col('COG_pair').is_in(set(sev['COG_pair'])))
    only_sev = sev.filter(~pl.col('COG_pair').is_in(set(cotr['COG_pair'])))
    metrics = pl.concat([only_cotr.with_columns(
                                pl.lit('only cotr').alias('method'),
                                pl.lit(th).alias('True Positive Rate')), 
                         only_sev.with_columns(
                                pl.lit('only sev').alias('method'),
                                pl.lit(th).alias('True Positive Rate'))
                        ])

    if j == 0:
        pivot_transition = metrics
    else:
        pivot_transition = pl.concat([pivot_transition, metrics])

        
    cwa = tp.filter((pl.col('cwa') == True) & (pl.col('coeff') > 0))
    asa = tp.filter((pl.col('asa') == True) & (pl.col('coeff') > 0))
    
    only_asa = asa.filter(~pl.col('COG_pair').is_in(set(cwa['COG_pair'])))
    only_cwa = cwa.filter((~pl.col('COG_pair').is_in(set(asa['COG_pair']))))
    metrics = pl.concat([only_asa.with_columns(
                                pl.lit('only asa').alias('method'),
                                pl.lit(th).alias('True Positive Rate')), 
                         only_cwa.with_columns(
                                pl.lit('only cwa').alias('method'),
                                pl.lit(th).alias('True Positive Rate'))
                        ])
    if j == 0:
        pivot_asa_cwa = metrics
    else:
        pivot_asa_cwa = pl.concat([pivot_asa_cwa, metrics])
    
sns.stripplot(data=pivot_transition,  x='True Positive Rate', y='coeff', hue='method', size=5, alpha=0.5,
                  dodge=True, hue_order=["only cotr", "only sev"],
                  order= thresholds, ax=ax[0])
sns.stripplot(data=pivot_asa_cwa,  x='True Positive Rate', y='coeff', hue='method', size=5, alpha=0.5,
                  dodge=True, hue_order=["only cwa", "only asa"],
                  order= thresholds, ax=ax[1])
ax[0].set_xlabel("") 
ax[1].set_xlabel("True positive Rate", fontsize=13)

for i in range(2):
    handles, labels = ax[i].get_legend_handles_labels()
    ax[i].legend_.remove()
    ax[i].legend(handles, labels,
               loc="center left", bbox_to_anchor=(1, 0.5))

fig.savefig('Fig3.png', dpi=300, bbox_inches="tight")

# %%
# Fig. S3
fig, ax = plt.subplots(3, 1, figsize = (8.27, 10))
fig.subplots_adjust(hspace=0.3)
for i, lin in enumerate(lineages):
    for j, th in enumerate(thresholds):
        tp = df_tp[th][lin]
        cotr = tp.filter((pl.col('cotr') == True) & (pl.col('coeff') > 0))
        sev = tp.filter((pl.col('sev') == True) & (pl.col('coeff') > 0))
    
        only_cotr = cotr.filter(~pl.col('COG_pair').is_in(set(sev['COG_pair'])))
        only_sev = sev.filter(~pl.col('COG_pair').is_in(set(cotr['COG_pair'])))
        metrics = pl.concat([only_sev.with_columns(
                                   pl.lit('only sev').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')),
                             only_cotr.with_columns(
                                   pl.lit('only cotr').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')), 
                            ])
        if j == 0:
            pivot_transition = metrics
        else:
            pivot_transition = pl.concat([pivot_transition, metrics])
    
    pivot_transition = pd.DataFrame(pivot_transition).rename(columns = {0:'COG_pair', 13:'coeff', 21:'method', 22:'True Positive Rate'})
    sns.stripplot(data=pivot_transition,  x='True Positive Rate', y='coeff', hue='method', 
                  size=5, alpha=0.5, order=thresholds, dodge=True, 
                  hue_order=['only cotr', 'only sev'], ax=ax[i],
                  )
    ax[i].legend_.remove()
    ax[i].set_title(lin)
    if i != 2:
        ax[i].set_xlabel("")  
    else:
        ax[i].set_xlabel("True positive Rate", fontsize=15)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", 
           fontsize=12, ncol=2)

fig.savefig("FigS3.png", dpi=300)

# %%
# Fig. S4
fig, ax = plt.subplots(3, 1, figsize = (8.27, 10))
fig.subplots_adjust(hspace=0.3)
threshold = [0.9, 0.8, 0.7, 0.6, 0.5]
for i, lin in enumerate(lineages):
    for j, th in enumerate(thresholds):
        tp = df_tp[th][lin]
        asa = tp.filter((pl.col('asa') == True) & (pl.col('coeff') > 0))
        cwa = tp.filter((pl.col('cwa') == True) & (pl.col('coeff') > 0))
    
        only_asa = asa.filter(~pl.col('COG_pair').is_in(set(cwa['COG_pair'])))
        only_cwa = cwa.filter(~pl.col('COG_pair').is_in(set(asa['COG_pair'])))
        metrics = pl.concat([only_asa.with_columns(
                                   pl.lit('only asa').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')),
                             only_cwa.with_columns(
                                   pl.lit('only cwa').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')), 
                            ])
        if j == 0:
            pivot_asa_cwa = metrics
        else:
            pivot_asa_cwa = pl.concat([pivot_asa_cwa, metrics])
    pivot_transition = pd.DataFrame(pivot_asa_cwa).rename(columns = {0:'COG_pair', 13:'coeff', 21:'method', 22:'True Positive Rate'})
    sns.stripplot(data=pivot_transition,  x='True Positive Rate', y='coeff', hue='method', 
                  size=5, alpha=0.5, order=thresholds, dodge=True, 
                  hue_order=['only cwa', 'only asa'], ax=ax[i],
                  )
    ax[i].legend_.remove()
    ax[i].set_title(lin)
    if i != 2:
        ax[i].set_xlabel("")  
    else:
        ax[i].set_xlabel("True positive Rate", fontsize=15)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", 
           fontsize=12, ncol=2)

fig.savefig("FigS4.png", dpi=300)

# %%
# Fig. S5
fig, ax = plt.subplots(3, 1, figsize = (8.27, 10))
fig.subplots_adjust(hspace=0.3)
threshold = [0.9, 0.8, 0.7, 0.6, 0.5]
for i, lin in enumerate(lineages):
    for j, th in enumerate(threshold):
        tp = df_tp[th][lin]
        asa = tp.filter((pl.col('asa') == True) & (pl.col('coeff') > 0))
        rle = tp.filter((pl.col('rle') == True) & (pl.col('coeff') > 0))
    
        only_asa = asa.filter(~pl.col('COG_pair').is_in(set(rle['COG_pair'])))
        only_rle = rle.filter(~pl.col('COG_pair').is_in(set(asa['COG_pair'])))
        metrics = pl.concat([only_asa.with_columns(
                                   pl.lit('only asa').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')),
                             only_cwa.with_columns(
                                   pl.lit('only rle').alias('method'),
                                   pl.lit(th).alias('True Positive Rate')), 
                            ])
        if j == 0:
            pivot_asa_rle = metrics
        else:
            pivot_asa_rle = pl.concat([pivot_asa_rle, metrics])
    pivot_transition = pd.DataFrame(pivot_asa_rle).rename(columns = {0:'COG_pair', 13:'coeff', 21:'method', 22:'True Positive Rate'})
    sns.stripplot(data=pivot_transition,  x='True Positive Rate', y='coeff', hue='method', 
                  size=5, alpha=0.5, order=threshold, dodge=True, 
                  hue_order=['only rle', 'only asa'], ax=ax[i],
                  )
    ax[i].legend_.remove()
    ax[i].set_title(lin)
    if i != 2:
        ax[i].set_xlabel("")  
    else:
        ax[i].set_xlabel("True positive Rate", fontsize=15)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", 
           fontsize=12, ncol=2)

fig.savefig("FigS5.png", dpi=300)

# %%
