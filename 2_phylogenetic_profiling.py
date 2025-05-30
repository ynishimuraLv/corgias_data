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
## Perform phylogenetic profiling

# %%bash
dirs=(pseudomonadales mycobacteriales archaea)
for dir in "${dirs[@]}"; do
    (
        cd $dir &&
        corgias asr -t hq_tree.nwk -d COG_table99.csv -i 0 -s "," -o pastml_ML -c 50 --prediction_method ML &&
        corgias asr -t hq_tree.nwk -d COG_table99.csv -i 0 -s "," -o pastml_MP -c 50 --prediction_method ACCTRAN &&
        corgias asr -t hq_tree.nwk -d COG_table99.csv -i 0 -s "," -o pastml_DOWNPASS -c 50 --prediction_method ACCTRAN &&
        corgias asr -t hq_tree.nwk -d COG_table99.csv -i 0 -s "," -o pastml_DELTRAN -c 50 --prediction_method DELTRAN &&
    )
done


### Phylogenetic profiling

dirs=(pseudomonadales mycobacteriales archaea)
for dir in "${dirs[@]}"; do
    (
        cd $dir &&
        corgias profiling -m naive -og COG_table99.csv -o naive.csv -c 50 --gpu &&
		corgias profiling -m rle -og COG_table99.csv -t hq_tree.nwk -o rle.csv -c 50 &&
		corgias profiling -m cwa -og COG_table99.csv -t hq_tree.nwk -o cwa.csv -c 50 &&
        for meth in (ML MP DOWNPASS DELTRAN); do
            corgias profiling -m asa -a pastml_${meth} -t hq_tree.nwk -o asa_${meth}.csv -c 50
        done
		corgias profiling -m cotr -og COG_table99.csv -t hq_tree.nwk -o cotr.csv -c 50 &&
        for meth in  (MP DOWNPASS DELTRAN); do
            corgias profiling -m sev -a pastml_${meth} -t hq_tree.nwk -o sev_$meth.csv -c 50 --gpu
        done
    )
done


### Statistical test

dirs=(pseudomonadales mycobacteriales archaea)
for dir in "${dirs[@]}"; do
    (
        cd $dir &&
        corgias stat -i naive.csv -m naive -o naive_stat.csv -c 50
		corgias stat -i rle.csv -m rle -o rle_stat.csv -c 50
		corgias stat -i cwa.csv -m cwa -o cwa_stat.csv -c 50
        for meth in (ML MP DOWNPASS DELTRAN); do
    		corgias stat -m asa_${meth}.csv -m ASA -o asa_${meth}_stat.csv -c 50
        done
		corgias stat -i cotr.csv -m cotr -o cotr_stat.csv -c 50
        for meth in (MP DOWNPASS DELTRAN); do
		    corgias stat -i sev_${meth}.csv -m sev -o sev_${meth}_stat.csv -c 50
        done
    )
done


# %%
import polars as pl
from sklearn.preprocessing import MinMaxScaler


# %%
def select_pair_pvalue(df, method):
    df = df.with_columns(larger = pl.max_horizontal('OG1', 'OG2'),
            smaller = pl.min_horizontal('OG1', 'OG2')
            ).select(
                pl.concat_str(pl.col('larger'), pl.col('smaller'), separator='_').alias('COG_pair'),
                pl.col('pvalue').alias(method)
            )

    small_value =  4e-324
    df = df.with_columns(pl.when(pl.col(method) == 0).then(small_value).otherwise(pl.col(method)).alias('test'))
    df = df.drop(method).rename({'test':method})
    
    return df


# %%
string = pl.read_csv('COG.links.wo_cooccurence.txt')

lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']
methods = ['naive', 'rle', 'cwa', 'asa_ACCTRAN', 'asa_DOWNPASS', 'asa_DELTRAN', 'asa_ML',
           'cotr', 'sev_MP', 'sev_DOWNPASS', 'sev_DELTRAN']
replace = { f'column_{i}':meth for i, meth in enumerate(methods)}
for lineage in lineages:
    df = pl.read_csv(f'{lineage}/naive_stat.csv')
    df = select_pair_pvalue(df, 'naive')

    for method in methods[1:]:
        pvalue = pl.read_csv(f'{lineage}/{method}_stat.csv')
        pvalue = select_pair_pvalue(pvalue, method)
        df = df.join(pvalue, on='COG_pair')

    df = df.join(string, on='COG_pair')
    df = df.with_columns(pl.when(pl.col('score') >= 900).then(1).otherwise(0).alias('truth'))

    pvalue = df.select(methods)
    log_pvalue = -np.log10(pvalue)
    scaler = MinMaxScaler(copy = True)
    scaler.fit(log_pvalue)
    scaled = pl.DataFrame(scaler.transform(log_pvalue))
    scaled = scaled.rename(repalce)
    scaled = scaled.with_columns(df[['COG_pair', 'score', 'truth']])

    scaled.write_csv(f'{lineage}/scaled_pvalues.csv')
