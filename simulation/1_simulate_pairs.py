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
library(doParallel)
library(phytools)
library(evolCCM)

# %%
tree = pbtree(n = 500)
write(write.tree(tree), file='tree500.tre')

tree = pbtree(n = 1000)
write(write.tree(tree), file='tree1K.tre')

tree = pbtree(n = 2000)
write(write.tree(tree), file='tree2K.tre')

tree = pbtree(n = 4000)
write(write.tree(tree), file='tree4K.tre')

tree = pbtree(n = 8000)
write(write.tree(tree), file='tree8K.tre')

tree = pbtree(n = 16000)
write(write.tree(tree), file='tree16K.tre')

# %%
n <- 2
setting <- matrix(NA, nrow=32000, ncol=5)
for (i in 1:320) {
    alpha <- runif(n, -0.5, 1)
    B_diag <- runif(n, -0.5, 0.3)
    coeff <- runif(1, 0.2, 0.75)
    setting[i,] = c(alpha, B_diag, coeff)
}

for (i in 321:32000) {
    alpha <- runif(n, -0.5, 1)
    B_diag <- runif(n, -0.5, 0.3)
    coeff <- runif(1, 0.2, 0.75)
    setting[i,] = c(alpha, B_diag, coeff)
}
colnames(setting) <- c('a1', 'a2', 'b11', 'b22', 'b12')
write.csv(setting, paste0('sim_params.csv'))

# %%
csv = read.csv('sim_params.csv')
n <- 2

for num in c('500', '2K', '4K', '8K', '16K') {
    tree <- read.tree(paste0('tree', num, '.tre'))
    dir.create(paste0('pairs4tree', num))
    
    cores <- 10
    cl <- makeCluster(cores)
    registerDoParallel(cl)
    foreach(i = 1:40, .packages = "evolCCM")  %dopar% {
        params <- csv[i, ]
        alpha <- unlist(params[2:3])
        B <- matrix(c(params$b11, params$b12, params$b12, params$b22), n, n)
        simDF <- SimulateProfiles(tree, alpha, B)
        write.csv(simDF, paste0('pairs4tree', num, '/', i, '.csv'))
    }
    foreach(i = 2001:5960, .packages = "evolCCM")  %dopar% {
        params <- csv[i, ]
        alpha <- unlist(params[2:3])
        B <- matrix(c(params$b11, params$b12, params$b12, params$b22), n, n)
        simDF <- SimulateProfiles(tree, alpha, B)
        write.csv(simDF, paste0('pairs4tree', num, '/', i-200+40, '.csv'))
    }
    stopCluster(cl)
}

# %%
tree <- read.tree('tree1K.tre')

n <- 2
dir.create('pairs4tree1K')
cores <- 10
cl <- makeCluster(cores)
registerDoParallel(cl)
foreach(i = 1:32000, .packages = "evolCCM")  %dopar% {
    params <- csv[i, ]
    alpha <- unlist(params[2:3])
    B <- matrix(c(params$b11, params$b12, params$b12, params$b22), n, n)
    simDF <- SimulateProfiles(tree, alpha, B)
    write.csv(simDF, paste0('pairs4tree1K', '/', i, '.csv'))
}

# %%
