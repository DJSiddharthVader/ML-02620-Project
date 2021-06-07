#!/bin/bash

input_dir='./data/downloaded'

# get lung samples for each dataset
cut -d',' -f1 $input_dir/SraRunTable.txt >| ./data/gtex_lung_ids.txt
grep "Lung Adenocarcinoma" $input_dir/TCGA.tsv | cut -f19 | cut -f1 -d'.' >| ./data/tcga_lung_ids.txt

# make list of only coding genes, from ensembl biomart
cut -f1 $input_dir/gene_metadata.txt | sort -u >| ./data/coding_genes.txt
# keep only expression data for coding genes from TCGA/GTEx
echo 'Filtering non-coding genes from GTEx data...'
head -1 $input_dir/gtex_counts.tsv >| ./data/gtex_coding_counts.tsv
grep -f ./data/coding_genes.txt $input_dir/gtex_counts.tsv >> ./data/gtex_coding_counts.tsv
echo 'Filtering non-coding genes from TCGA data...'
head -1 $input_dir/tcga_counts.tsv >| ./data/tcga_coding_counts.tsv
grep -f ./data/coding_genes.txt $input_dir/tcga_counts.tsv >> ./data/tcga_coding_counts.tsv
# be sure to have pandas installed for your current python path
echo 'Building dataframe with TCGA+GTEx samples and labels...'
python ./src/prep_data.py
