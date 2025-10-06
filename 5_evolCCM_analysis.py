# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
## EvolCCM

library(ape)
library(evolCCM)
library(doParallel)

# %%
lineages <- c("pseudomonadales", "mycobacteriales", "archaea")

for (lineage in lineages) {

    tree <- read.tree(paste0(lineage, '/hq_tree.tre'))
    cog_table <- read.csv(paste0(lineage, '/COG_table99.csv'))
    index <- cog_table[, 'X']
    rownames(cog_table) <- index
    cog_table <- cog_table[, 2:length(colnames(cog_table))]
    
    lines <- readLines(paste0(lineage, '/tp_pairs.txt'))
    
    cores <- 12
    cl <- makeCluster(cores)
    registerDoParallel(cl)
    
    result <- foreach(line = lines, .packages = 'evolCCM') %dopar% {
        pair <- strsplit(line, split='_')[[1]]
        cog1 <- pair[1]
        cog2 <- pair[2]
        pair <- cog_table[, c(cog1, cog2)]
        
        estimate <- EstimateCCM(pair, tree)
        estSE <- ProcessAE(estimate)$hessian
        signif <- estimate$nlm.par[5] / estSE[5]
        pvalue <- 1 - pnorm(abs(signif))
        
        c(cog1, cog2, estimate$nlm.par, pvalue)
    }
    write.csv(result, paste0(lineage, '/evolCCM.csv'))
}

