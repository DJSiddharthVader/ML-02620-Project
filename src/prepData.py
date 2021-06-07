import pandas as pd
from sklearn.preprocessing import quantile_transform

def quantileNormalize(df_input):
    # from https://github.com/ShawnLYU/Quantile_Normalize/blob/master/quantile_norm.py
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

# Load TCGA data and get lung samples
print('Filtering TCGA data...')
tcga = pd.read_csv('./data/tcga_coding_counts.tsv',sep='\t')
tcga_lung_ids = [x.strip('\n') for x in open('./data/tcga_lung_ids.txt').readlines()]
tcga_lung_ids = list(set(tcga_lung_ids).intersection(set(tcga.columns)))
tcga = tcga[['gene_id'] + tcga_lung_ids] # keep lung samples only

# Load GTEx data and get lung samples
print('Filtering GTEx data...')
gtex = pd.read_csv('./data/gtex_coding_counts.tsv',sep='\t')
gtex_lung_ids = [x.strip('\n') for x in open('./data/gtex_lung_ids.txt').readlines()]
gtex_lung_ids = list(set(gtex_lung_ids).intersection(set(gtex.columns)))
gtex= gtex[['gene_id'] + gtex_lung_ids] # keep lung samples only

# Combine data frames and add labels
print('Combining dataframes...')
data = pd.merge(gtex,tcga,on='gene_id').T # now genes are columns, samples are rows
data.columns = data.iloc[0,:]
data = data.iloc[1:,:]
data = quantile_transform(data,n_quantiles=10,random_state=0)
data['labels'] = [0]*(gtex.shape[1]-1) + [1]*(tcga.shape[1]-1) # label 0 for not cancer,1 for cancer
data.to_csv('./data/data.tsv',sep='\t',index=False)
