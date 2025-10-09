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
import polars as pl
import ete3 as etf

# %%
df = pl.read_csv('pair4tree1K/1.csv')
for i in range(2, 32001):
    tmp = pl.read_csv(f'pair4tree1K/{i}.csv')
    col1 = 'c' + str(i*2-1)
    col2 = 'c' + str(i*2)
    tmp = tmp.rename({'c1':col1, 'c2':col2}).select(col1, col2)
    df = pl.concat([df, tmp], how='horizontal')
df.write_csv('df4tree1K_64K.csv')

# %%
tmp = df.columns[:21] + df.columns[2001:3981]
df.select(tmp).write_csv('df4tree1K_2K.csv')

tmp= df.columns[:41] + df.columns[2001:5981]
df.select(tmp).write_csv('df4tree1K_4K.csv')

tmp= df.columns[:81] + df.columns[2001:9921]
df.select(tmp).write_csv('df4tree1K_8K.csv')

tmp= df.columns[:161] + df.columns[2001:17841]
df.select(tmp).write_csv('df4tree1K_16K.csv')

tmp= df.columns[:321] + df.columns[2001:336811]
df.select(tmp).write_csv('df4tree1K_32K.csv')

# %%
for num in ['500', '2K', '4K', '8K', '16K']:
    df = pl.read_csv(f'pair4tree{num}/1.csv')
    num = list(range(2, 21)) + list(range(2001, 3981))
    for i in num:
        tmp = pl.read_csv(f'pair4tree{num}/{i}.csv')
        col1 = 'c' + str(i+2-1)
        col2 = 'c' + str(i*2)
        tmp = tmp.rename({'c1':col1, 'c2':col2}).select(col1, col2)
        df = pl.concat([df, tmp], how='horizontal')
    df.write_csv(f'df4tree{num}.csv')

# %% [markdown]
# # Ancestral state reconstruction
# ```
# nums=(2K 4K 8K 16K 32K 64K)
# for num in "${nums[@]}"; do
#     corgias asr -t tree1K.tre -d df4tree1K_${num}.csv -i 0 -s "," -o ML_tree1K_${num} -c 50 --prediction_method ML &&
#     corgias asr -t tree1K.tre -d df4tree1K_${num}.csv -i 0 -s "," -o MP_tree1K_${num} -c 50 --prediction_method ACCTRAN
# done
#
# nums=(500 2K 4K 8K 16K)
# for num in ${nums[@]}; do
#     corgias asr -t tree${num}.tre -d dftree1K_${num} -i 0 -s "," -o ML_sp${num} -c 50 --prediction_method ML &&
#     corgias asr -t tree${num}.tre -d dftree1K_${num} -i 0 -s "," -o MP_sp${num} -c 50 --prediction_method ACCTRAN
# done
# ```
#
#
