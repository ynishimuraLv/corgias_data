blibrary(SynExtend)
library(parallel)

for (org in c('pseudomonadales', 'mycobacteriales', 'archaea')) {

    dend <- ReadDendrogram(paste0(org, '/hq_tree4evoweaver.tre'))
    mat <- read.csv(paste0(org, '/COG_table4evoweaver.csv'),
                    row.names = 1, check.names = FALSE)
    gene_groups <- apply(mat, 1, function(row) colnames(mat)[row == 1])
    ew <- EvoWeaver(gene_groups, MySpeciesTree=dend)
    
    
    genes <- rownames(mat)
    all_pairs_mat <- t(combn(genes, 2))
    pair_df <- data.frame(
      gene1 = all_pairs_mat[,1],
      gene2 = all_pairs_mat[,2],
      stringsAsFactors = FALSE
    )
    
    ncores <- 60
    n_pairs <- nrow(pair_df)
    chunks <- split(seq_len(n_pairs), cut(seq_len(n_pairs), ncores))
    methods_to_use <- c("PAJaccard",
                        "PAOverlap",
                        "GLMI",
                        "GLDistance")
    
    results <- mclapply(chunks, function(idx) {
      # idx は行番号のベクトル
      Subset_chunk <- pair_df[idx, , drop = FALSE]  # 1行だけでも2列維持
      predict(
        ew, Method = methods_to_use,
        Subset = Subset_chunk,
        Processors = 1L
      )
    }, mc.cores = ncores)
    
    final_result <- do.call(rbind, results)
    write.csv(final_result, paste0(org, '/evoweaver.csv'))

}