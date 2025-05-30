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
import os,shutil,pathlib
import ete3 as et
import numpy as np
import polars as pl

# %%
## Downloads data from GTDB RS214(~60GB)

# !mkdir GTDB; cd GTDB

# !wget https://data.gtdb.ecogenomic.org/releases/release214/214.0/bac120_r214.tree
# !wget https://data.gtdb.ecogenomic.org/releases/release214/214.0/ar53_r214.tree
# !wget https://data.gtdb.ecogenomic.org/releases/release214/214.0/bac120_metadata_r214.tar.gz
# !wget https://data.gtdb.ecogenomic.org/releases/release214/214.0/genomic_files_reps/gtdb_proteins_aa_reps_r214.tar.gz

# !tar -zxvf gtdb_proteins_aa_reps_r214.tar.gz
# !cd gtdb_protein_aa_reps_r214/bacteria; gzip -d *.gz
# !../archaea; gzip -d *.gz; cd ../../..


# %%
## Prepare checkM2 input
lineages = ['pseudomonadales', 'mycobacteriales', 'archaea']

GTDB_path = pathlib.Path('GTDB214/').resolve()
prot_path = GTDB_path / 'protein_faa_reps'
bac_meta = pl.read_csv(GTDB_path / 'bac120_metadata_r214.tsv.gz', separator='\t', ignore_errors=True)

for lineage in lineages:
    os.makedirs(lineage + '/' + 'checkM2_input', exist_ok=True)
    os.chdir(lineage + '/' + 'checkM2_input')
    if lineage != 'archaea':
        genomes = bac_meta.filter((pl.col('gtdb_representative') == 't') & 
                                  (pl.col('gtdb_taxonomy').str.contains(f'o__{lineage.capitalize()}')))
        for row in genomes.iter_rows():
            genome_path = prot_path / 'bacteria' / (row[0] + '_protein.faa.gz')
            shutil.copy(genome_path, row[0] + '.faa.gz')
    else:
        for genome in os.listdir(prot_path / 'archaea'):
            genome_path = prot_path / 'archaea' / genome 
            shutil.copy(genome_path, genome)
    os.chdir('../../')

# %%
## CheckM2
# !bash run_checkM2.sh

# %%
## Extract high quality proteomes
for lineage in lineages:
    checkm2 = pl.read_csv(f'{lineage}/checkM2_output/quality_report.tsv', separator='\t')
    hq_genomes = checkm2.filter((pl.col('Completeness') >=90) & (pl.col('Contamination') < 5))
    os.makedirs(f'{lineage}/hq_prot', exist_ok=True)
    os.chdir(f'{lineage}/hq_prot')
    for row in hq_genomes.iter_rows():
        file = row[0] + '.faa'
        os.symlink(f'../checkM2_input/{file}', file)
    os.chdir('..')


# %% language="bash"
#
# dirs=(pseudomonadales mycobacteriales archaea)
#
# for dir in "${dirs[@]}"; do
#     cd $dir
#     mkdir COG_annot
#     for file in hq_prot/*; do
#         if [ -f "$file" ]; then
#         filename=$(basename "$file")   
#         genome="${filename%.*}" 
#         COGclassifier -i $file -o COG_annot/$genome -t 40
#         fi
#     done
#     cd ..
# done
# ```

# %%
def df_T(df):
    df = df.rename({'column_0':'COG'})
    df_pd = df.to_pandas().T
    replace = { i:cog for i, cog in enumerate(df_pd.loc['COG'])}
    df_pd = df_pd.rename(columns = replace).iloc[1:]

    return df_pd


# %%
# prepare profiles

for lineage in lienages:
    cogs = set()
    base = f'{lin}/COG_annot/'
    for folder in os.listdir(base):
        tsv = f'{base}/{folder}/classifier_result.tsv'
        if os.path.exists(tsv):
            cogs.update({ line.split()[1] for line in open(tsv)})
    
    cogs = sorted(list(cogs))
    df = pl.DataFrame(cogs)
    for folder in os.listdir(f'{base}'):
        tsv = f'{base}/{folder}/classifier_result.tsv'
        if os.path.exists(tsv):
            cog = { line.split()[1] for line in open(tsv)}
            tmp = [ 1 if c in cog else 0 for c in cogs]
            new_colums = pl.Series(folder, tmp)
            df = df.with_columns(new_colums)
    
    df = df[:-1]
    df_T(df).to_csv(f'{lin/}COG_table.csv') # all COGs profile
    df = df.with_columns(df[:, 1:].sum_horizontal().alias('total'))
    num_genomes = df.shape[1] - 1
    df = df.filter((pl.col('total') <= num_genomes * 0.99),
              pl.col('total') >= num_genomes * 0.01)
    df_T(df)[-1].write_csv(f'{lin}/COG_table99.csv') # profile of COGs present in 1~99% genomes

# %%
# prepare phylogenetic trees

tree = et.Tree('GTDB/bac120_r214.tree', format=1, quoted_node_names=True)
for lineage in lienages:
    if lineage == 'archaea':
        tree = et.Tree('GTDB/ar53_r214.tree', format=1, quoted_node_names=True)
    
    genomes = { file.replace('.faa', '') for file in os.listdir(f'{lineage}/checkM2_input')}
    tree.prune(otu, preserve_branch_length=True)
    tree.write(format=1, outfile=f'{lineage}/hq_tree.tre')

# %%
## Download STRING socre table

# !wget https://stringdb-downloads.org/download/COG.links.detailed.v12.0.txt.gz
# !gzip -d COG.links.detailed.v12.0.txt.gz

# %%
# Calculate score without coocurrence
# original script can be found from https://stringdb-downloads.org/download/combine_subscores.v2.py

prior = 0.041

def compute_prior_away(score, prior):

    if score < prior: score = prior
    score_no_prior = (score - prior) / (1 - prior)

    return score_no_prior

header = True
input_file = 'COG.links.detailed.v12.0.txt'
output_file = open('COG.links.wo_cooccurence.txt', mode='w')
for line in open(input_file):

    if header:
        header = False
        continue
    
    l = line.split()
    
    ## load the line
        
    (protein1, protein2,
     neighborhood, #neighborhood_transferred,
     fusion, cooccurrence,
#     homology,
     coexpression, #coexpression_transferred,
     experiments, #experiments_transferred,
     database, #database_transferred,
     textmining, #textmining_transferred,
     initial_combined) = l


    ## divide by 1000

    neighborhood = float(neighborhood) / 1000
#    neighborhood_transferred = float(neighborhood_transferred) / 1000
    fusion = float(fusion) / 1000
    cooccurrence =  float(cooccurrence) / 1000
#    homology = float(homology) / 1000
    coexpression = float(coexpression) / 1000
#    coexpression_transferred = float(coexpression_transferred) / 1000
    experiments = float(experiments) / 1000
#    experiments_transferred = float(experiments_transferred) / 1000
    database = float(database) / 1000
#    database_transferred = float(database_transferred) / 1000
    textmining = float(textmining) / 1000
#    textmining_transferred = float(textmining_transferred) / 1000
    initial_combined = int(initial_combined)


    ## compute prior away

    neighborhood_prior_corrected                 = compute_prior_away (neighborhood, prior)             
#    neighborhood_transferred_prior_corrected     = compute_prior_away (neighborhood_transferred, prior) 
    fusion_prior_corrected                       = compute_prior_away (fusion, prior)             
#    cooccurrence_prior_corrected                 = compute_prior_away (cooccurrence, prior)           
    coexpression_prior_corrected                 = compute_prior_away (coexpression, prior)            
#    coexpression_transferred_prior_corrected     = compute_prior_away (coexpression_transferred, prior) 
    experiments_prior_corrected                  = compute_prior_away (experiments, prior)   
#    experiments_transferred_prior_corrected      = compute_prior_away (experiments_transferred, prior) 
    database_prior_corrected                     = compute_prior_away (database, prior)      
#    database_transferred_prior_corrected         = compute_prior_away (database_transferred, prior)
    textmining_prior_corrected                   = compute_prior_away (textmining, prior)            
#    textmining_transferred_prior_corrected       = compute_prior_away (textmining_transferred, prior) 

    ## then, combine the direct and transferred scores for each category:

#    neighborhood_both_prior_corrected = 1.0 - (1.0 - neighborhood_prior_corrected) * (1.0 - neighborhood_transferred_prior_corrected)
#    coexpression_both_prior_corrected = 1.0 - (1.0 - coexpression_prior_corrected) * (1.0 - coexpression_transferred_prior_corrected)
#    experiments_both_prior_corrected = 1.0 - (1.0 - experiments_prior_corrected) * (1.0 - experiments_transferred_prior_corrected)
#    database_both_prior_corrected = 1.0 - (1.0 - database_prior_corrected) * (1.0 - database_transferred_prior_corrected)
#    textmining_both_prior_corrected = 1.0 - (1.0 - textmining_prior_corrected) * (1.0 - textmining_transferred_prior_corrected)

    ## next, do the 1 - multiplication:

    combined_score_one_minus = (
        (1.0 - neighborhood_prior_corrected) *
        (1.0 - fusion_prior_corrected) *
#        (1.0 - cooccurrence_prior_corrected) *
        (1.0 - coexpression_prior_corrected) *
        (1.0 - experiments_prior_corrected) *
        (1.0 - database_prior_corrected) *
        (1.0 - textmining_prior_corrected) 
        
#        (1.0 - neighborhood_both_prior_corrected) *
#        (1.0 - fusion_prior_corrected) *
#        (1.0 - cooccurrence_prior_corrected) *
#        (1.0 - coexpression_both_prior_corrected) *
#        (1.0 - experiments_both_prior_corrected) *
#        (1.0 - database_both_prior_corrected) *
#        (1.0 - textmining_both_prior_corrected) 
    ) 

    ## and lastly, do the 1 - conversion again, and put back the prior *exactly once*

    combined_score = (1.0 - combined_score_one_minus)            ## 1- conversion
    combined_score *= (1.0 - prior)                              ## scale down
    combined_score += prior                                      ## and add prior.

    ## round

    combined_score = int(combined_score * 1000)
    output_file.write(f'{protein1} {protein2} {combined_score}\n')

output_file.close()

# %%
string = pl.read_csv('COG.links.wo_cooccurence.txt', separator=' ',
                     has_header=False)
string = string.rename({'column_1':'COG1', 'column_2':'COG2', 'column_3':'score'})
string = string.filter(pl.col('COG1').str.starts_with('COG') & pl.col('COG2').str.starts_with('COG'))
string = string.with_columns(
    string.select('COG1', 'COG2').max_horizontal().alias('larger'),
    string.select('COG1', 'COG2').min_horizontal().alias('smaller')
)
string = string.with_columns((pl.concat_str([pl.col('smaller'), pl.col('larger')], 
                                            separator='_')).alias('COG_pair')).select('COG_pair', 'score')
string = string.unique('COG_pair')
string.write_csv('COG.links.wo_cooccurence.txt')
