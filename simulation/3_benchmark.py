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

# %% [markdown]
# ## Bechmark with a fix number of genes (4,000) and varying numbers of genome
# ```
# for sp in (500 1K 2K 4K 16K); do
#     table=df4tree${sp}.csv
#     tree=tree${sp}.tre
#     time -f "naive(cpu), ${sp} genomes: %E" corgias profiling -m naive -og $table -o naive_sp${sp}_cpu.csv
#     time -f "naive(gpu), ${sp} genomes: %E" corgias profiling -m naive -og $table -o naive_sp${sp}_gpu.csv --gpu
#     time -f "rle, ${sp} genomes: %E" corgias profiling -m rle -og $table -t $tree -o rle_sp${sp}.csv -c 54
#     time -f "cwa, ${sp} genomes: %E" corgias profiling -m cwa -og $table -t $tree -o cwa_sp${sp}.csv -c 54
#     time -f "asa, ${sp} genomes" %E" corgias profiling -m asa -a ML_sp${sp} -t $tree -o asa_sp${sp}.csv -c 54
#     time -f  cotr(cpu), ${sp} genomes: %E" corgias profiling -m cotr -og $table -t $tree -o cotr_sp${sp}_cpu.csv -c 54
#     time -f  cotr(gpu), ${sp} genomes: %E" corgias profiling -m cotr -og $table -t $tree -o cotr_sp${sp}_gpu.csv -c 54 --gpu
#     time -f  sev(cpu), ${sp} genomes: %E" corgias profiling -m sev -a MP_sp${sp} -t $tree -o sev_sp${sp}_cpu.csv -c 54
#     time -f  sev(gpu), ${sp} genomes: %E" corgias profiling -m sev -a MP_sp${sp} -t $tree -o sev_sp${sp}_gpu.csv -c 54 --gpu
# done
# ```

# %% [markdown]
# ## Bechmark with a fix number of species (1,000) and varying numbers of genes
# ### Note that some settings (e.g., cwa or asa with more than 8K genes) can takes very long time (> 1day)
# ```
# for genes in (2K 8K 16K 32K 64K); do
#     table=df4tree1K_${genes}.csv
#     tree=tree1K.tre
#     time -f "naive(cpu), ${genes} genes: %E" corgias profiling -m naive -og $table -o naive_${genes}genes_cpu.csv
#     time -f "naive(gpu), ${genes} genes: %E" corgias profiling -m naive -og $table -o naive_${genes}genes_gpu.csv --gpu
#     time -f "rle, ${genes} genes: %E" corgias profiling -m rle -og $table -t $tree -o rle_${genes}genes.csv -c 54
#     time -f "cwa, ${genes} genes: %E" corgias profiling -m cwa -og $table -t $tree -o cwa_${genes}genes.csv -c 54
#     time -f "asa, ${genes} genes" %E" corgias profiling -m asa -a ML_tree1K_${genes} -t $tree -o asa_${genes}genes.csv -c 54
#     time -f "cotr(cpu), ${genes} genes: %E" corgias profiling -m cotr -og $table -t $tree -o cotr_${genes}genes_cpu.csv -c 54
#     time -f "cotr(gpu), ${genes} genes: %E" corgias profiling -m cotr -og $table -t $tree -o cotr_${genes}genes_gpu.csv -c 54 --gpu
#     time -f "sev(cpu), ${genes} genes: %E" corgias profiling -m sev -a MP_tree1K_${genes} -t $tree -o sev_${genes}genes_cpu.csv -c 54
#     time -f "sev(gpu), ${genes} genes: %E" corgias profiling -m sev -a MP_tree1K_${genes} -t $tree -o sev_${genes}genes_gpu.csv -c 54 --gpu
# done
# ```

# %%
# !Rscript run_GLDistance.r
