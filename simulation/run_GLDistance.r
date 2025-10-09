library('parallel')
library('SynExtend')

for (num in c('500', '1K', '2K', '4K', '8K', '16K')) {
    dend <- ReadDendrogram(paste0('tree', num, '.tre'))
    if (num == '1K') {
        mat <- read.csv('df4tree1K_4K.csv', row.names=1, check.names=FALSE)
    } else {
        mat <- read.csv(paste0('df4tree', num, '.csv'), row.names=1, check.names=FALSE)
    }
    mat <- t(mat)
    
    gene_groups <- apply(mat, 1, function(row) colnames(mat)[row == 1])
    ew <- EvoWeaver(gene_groups, MySpeciesTree = dend)

    genes <- rownames(mat)
    all_pairs_mat <- t(combn(genes, 2))
    pair_df <- data.frame(
        gene1 = all_pairs_mat[, 1],
        gene2 = all_pairs_mat[, 2],
        stringAsFactors = FALSE
    )
    pair_df <- pair_df[, c('gene1', 'gene2')]
    
    n_cores = 56
    n_pairs <- nrow(pair_df)
    chunks <- split(seq_len(n_pairs),
                    cut(seq_len(n_pairs), n_cores))
#    for (meth in c('PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance')) {
    for (meth in c('GLDistance')) {
        t <- system.time(
            results <- mclapply(chunks, function(idx) {
                Subset_chunk <- pair_df[idx, ]
                predict(
                    ew, Method = meth,
                    Subset = Subset_chunk,
                    Processors = 1L
                    )
                }, mc.cores = n_cores
            )
        )
        print(paste0(meth,' ', num, 'genomes and 4K genes'))
        print(t)

    }
}

for (num in c('2K', '8K', '16K')) {
    dend <- ReadDendrogram('tree1K.tre')
    mat <- read.csv(paste0('df4tree1K_', num, '.csv'), row.names=1, check.names=FALSE)
    mat <- t(mat)
    
    gene_groups <- apply(mat, 1, function(row) colnames(mat)[row == 1])
    ew <- EvoWeaver(gene_groups, MySpeciesTree = dend)

    genes <- rownames(mat)
    all_pairs_mat <- t(combn(genes, 2))
    pair_df <- data.frame(
        gene1 = all_pairs_mat[, 1],
        gene2 = all_pairs_mat[, 2],
        stringAsFactors = FALSE
    )
    pair_df <- pair_df[, c('gene1', 'gene2')]
    
    n_cores = 56
    n_pairs <- nrow(pair_df)
    chunks <- split(seq_len(n_pairs),
                    cut(seq_len(n_pairs), n_cores))
#    for (meth in c('PAJaccard', 'PAOverlap', 'GLMI', 'GLDistance')) {
    for (meth in c('GLDistance')) {
        t <- system.time(
            results <- mclapply(chunks, function(idx) {
                Subset_chunk <- pair_df[idx, ]
                predict(
                    ew, Method = meth,
                    Subset = Subset_chunk,
                    Processors = 1L
                    )
                }, mc.cores = n_cores
            )
        )
        print(paste0(meth,' 1K species and ', num, ' genes'))
        print(t)

    }
}