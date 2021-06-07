import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import upsetplot as ups
import seaborn as sns

# Map feature columns names to display names
col_name_map = {'log2fc':'Log2 FC',
                'SVM':'SVM',
                'NB':'Naive Bayes',
                'RF':'Random Forest'}

def load_data():
    idf = pd.read_csv('./data/importances.tsv',sep='\t')
    dedf = pd.read_csv('./data/de_results.tsv',sep='\t')
    df = pd.merge(idf,dedf,on='gene')
    df.index = df.gene
    df = df[[x for x in df.columns if x != 'gene']]
    return df

def gene_table(df,alpha=0.05,top=5):
    # top x most important genes  for each model
    x = {model:df.sort_values(model,ascending=False).index[:top].values for model in ['SVM','NB','RF']}
    x['log2fc'] = df[df['qval'] < alpha].sort_values('log2fc',ascending=False).index[:top]
    return pd.DataFrame(x)


def plot_feature_distributions(df,bins=100,alpha=0.05):
    # Plot DE fold changes and feature importance distributions
    df = df[list(col_name_map.keys())+['qval']]
    # Min-Max scale importances/FC to each be in range [0,1]
    df = (df-df.min())/(df.max()-df.min())
    # Plot distribution of feature importances
    fig, axs = plt.subplots(4,1,sharex=True,constrained_layout=True)
    axs[0].set_title('Feature Importance Distributions')
    axs[-1].set_xlabel('Scaled Feature Importance')
    # for each model plot the distribution
    for i,k in enumerate(col_name_map.keys()):
        if k == 'log2fc':
            df = df[df['qval'] < alpha] # only genes with significant adjusted p-value
        axs[i].hist(df[k],bins=bins,density=True)
        axs[i].set_ylabel(col_name_map[k])
    plt.savefig('./documents/figures/importance_distributions.png')
    plt.close()
    return None

def plot_upset_top_features(df,alpha=0.05,top=500):
    # get the top x most informative features (genes) from each model/DE
    # and make an upsetplot (better venn diagram)
    top_genes = {}
    for model,name in col_name_map.items():
        if model == 'log2fc':
            # for log2fc we get the most DE genes that are significant
            genes = df[df['qval'] < alpha].sort_values(model,ascending=False).index[:top]
        else:
            genes = df.sort_values(model,ascending=False).index[:top]
        top_genes[name] = genes
    top_genes = ups.from_contents(top_genes) # format for upset plot
    upset = ups.UpSet(top_genes, show_counts='%d', sort_by='cardinality')
    f, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
    f.suptitle('Top {} Genes From Each Model'.format(top))
    upset.plot_intersections(ax1)
    ax1.set_ylabel('Common Genes')
    upset.plot_matrix(ax2)
    f.tight_layout()
    f.savefig('./documents/figures/upset_genes_{}.png'.format(top))
    plt.close()
    return None

def plot_upset_GO_terms(df,alpha=0.05,top=100):
    gene_data = pd.read_csv('./data/downloaded/gene_metadata.txt',sep='\t').dropna(axis=0,subset=['GO term accession','Gene stable ID version'])
    top_GO = {}
    for model,name in col_name_map.items():
        if model == 'log2fc':
            # for log2fc we get the most DE genes that are significant
            genes = df[df['qval'] < alpha].sort_values(model,ascending=False).index[:top]
        else:
            genes = df.sort_values(model,ascending=False).index[:top]
        top_GO[name] = list(set(gene_data[gene_data['Gene stable ID version'].isin(genes)]['GO term accession'].values))
    top_GO = ups.from_contents(top_GO) # format for upset plot

    upset = ups.UpSet(top_GO, show_counts='%d', sort_by='cardinality')
    f, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
    f.suptitle('Top {} GO Terms From Each Model'.format(top))
    upset.plot_intersections(ax1)
    ax1.set_ylabel('Common Genes')
    upset.plot_matrix(ax2)
    f.tight_layout()
    f.savefig('./documents/figures/upset_GO_{}.png'.format(top))
    plt.close()
    return None


def accuracy(predictions,labels):
    # proportion of predicted labels matching actual labels
    return sum(np.where(predictions == labels,1,0))/len(labels)

def cross_training(df,data,top=50,alpha=0.05):
    """ Here we train each model using the top x most important features only
    we want to see if model accuracy significantly varies even when using the
    most informative features found by each model"""
    # Prep data
    labels = data['labels']
    data = data[[x for x in data.columns if x != 'labels']]
    accuracy_matrix = {}
    # for each mode, get the most important features and train each model on it
    for model in ['log2fc','SVM','NB','RF']:
        # get the x most important features from the model
        if model == 'log2fc':
            # for log2fc we get the most DE genes that are significant
            features = df[df['qval'] < alpha].sort_values(model,ascending=False).index[:top]
        else:
            features = df.sort_values(model,ascending=False).index[:top]
        # first split train/test data using only the given feature set
        # same random seed so split is the same between runs
        X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                            labels,
                                                            test_size=0.33,
                                                            random_state=42)
        #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        # train svm
        if model != 'log2fc':
            svm = SVC(kernel='linear').fit(X_train,y_train)
            acc_svm = accuracy(svm.predict(X_test),y_test)
        else:
            acc_svm = 0 # fails to converge for log2fc
        # train naive bayes
        nb = MultinomialNB().fit(X_train,y_train)
        acc_nb = accuracy(nb.predict(X_test),y_test)
        # train random forest
        rf = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train,y_train)
        acc_rf = accuracy(rf.predict(X_test),y_test)
        accuracy_matrix[col_name_map[model]] = [acc_svm,acc_nb,acc_rf] # store results
        print("finished {}".format(model))
    # transform to Dataframe and return
    cross_data = pd.DataFrame.from_dict(accuracy_matrix)
    cross_data.index = ['SVM','Naive Bayes','Random Forest']
    return cross_data

def plot_cross_training(cross):
    sns.set_theme()
    sns.heatmap(cross,annot=True,fmt="0.2f",cbar=False)
    plt.xlabel('Feature Set')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('./documents/figures/heatmap.png')
    plt.close()
    return None


if __name__ == '__main__':
    df = load_data()
    df['log2fc'] = np.abs(df['log2fc'])
    print(gene_table(df))
    plot_feature_distributions(df)
    print('distributions')
    plot_upset_top_features(df)
    print('upset gene')
    plot_upset_GO_terms(df)
    print('upset GO')
    data = pd.read_csv('./data/data.tsv',sep='\t')
    cross = cross_training(df,data)
    plot_cross_training(cross)
    print('cross training')

