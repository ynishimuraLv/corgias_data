!# #!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate checkM2

for dir in pseudomonadales mycobacteriales archaea; 
do
(
    cd "$dir" && gzip -d checkM2_input/*.gz &&
    checkm2 predict --input checkM2_input --output-directory checkM2_output --genes --threads 40 -x faa
)
done