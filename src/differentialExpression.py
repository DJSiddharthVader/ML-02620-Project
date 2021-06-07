import pandas as pd
import anndata as ad
import diffxpy.api as de

# Preform differential expression to find genes with clear expression differences in cancer vs non-cancer samples
# See here for column definitions
# https://nbviewer.jupyter.org/github/theislab/diffxpy_tutorials/blob/master/diffxpy_tutorials/test/introduction_differential_testing.ipynb

# load expression count data
data = pd.read_csv('./data/data.csv',sep='\t')
# prep data for diffxpy analysis
labels = {'label':data['labels'].values} # 0/1 = not cancer/cancer
genes = [x for x in data.columns if x != 'labels']
counts = data[genes].values
# like a SummarizedExperiment object in R, used by diffxpy
de_data = ad.AnnData(X=counts,obs=labels,var=genes)
# Preform differential expression between GTEx/TCGA samples
# using a Wald test between cancer/non-cancer label for each gene (~20,000)
de_results = de.test.wald(data=de_data,
                          formula_loc="~ 1 + label",
                          factor_loc_totest="label")
dedf = de_results.summary()
dedf['gene'] = genes
dedf.to_csv('./data/de_results.tsv',sep='\t',index=False)

