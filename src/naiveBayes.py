import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from functools import reduce
import operator

def build_model(df):
    model = {}
    # for each label (0/1)
    for label in set(df['labels']):
        label_probabilities = []
        subdf = df[df['labels'] == label][[x for x in df.columns if x != 'labels']]
        means, stds = subdf.mean(),subdf.std()
        prior= subdf.shape[0]/df.shape[0] #proportion of samples with label
        for i in tqdm(range(len(means))):
            # gaussian prob estimate of sample belonging to label given label specific parameters
            terms = [norm.pdf(sample[i],means[i],stds[i]) for _,sample in df.iterrows()]
            label_prob = prior*reduce(operator.mul,terms,1)
            label_probabilities.append(label_prob)
        model[label] = label_probabilities
    return model

def predict(df,model):
    predictions = []
    for i,sample in tqdm(enumerate(df.iterrows())):
        best_label, best_prob = None, -1
        for label in model.items():
            if model[label][i] > best:
                best_prob = prob
                best_label = label
        predictions.append(best_label)
    return predictions


if __name__ == '__main__':
    df = pd.read_csv('./data/data.tsv',sep='\t')
    model = build_model(df)
    yn = predict(df,model)
    print('Parameters')
    print(len(np.where(yn == df['labels'].values))/len(yn))
