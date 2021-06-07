import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def accuracy(predictions,labels):
    # proportion of predicted labels matching actual labels
    return sum(np.where(predictions == labels,1,0))/len(labels)

# Load GTEx+TCGA data for training binary classifier models
print('Loading data...')
data = pd.read_csv('./data/data.tsv',sep='\t')
labels = data['labels']
data = data[[x for x in data.columns if x != 'labels']]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
genes = data.columns

# Train each model and print test accuracy
print('Training models...')
## Train SVM and get feature importances
svm = SVC(kernel='linear').fit(X_train,y_train)
svm_fi = {gene:imp for gene,imp in zip(genes,svm.coef_[0])}
print('SVM Test Accuracy: {}'.format(accuracy(svm.predict(X_test),y_test)))
## Train Naive Bayes and get feature importances
nb = MultinomialNB().fit(X_train,y_train)
nb_fi = {gene:imp for gene,imp in zip(genes,nb.feature_log_prob_[1])} # features important to cacner
print('Naive Bayes Test Accuracy: {}'.format(accuracy(nb.predict(X_test),y_test)))
## Train Random Forest and get feature importances
rf = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train,y_train)
rf_fi = {gene:imp for gene,imp in zip(genes,rf.feature_importances_)}
print('Random Forest Test Accuracy: {}'.format(accuracy(rf.predict(X_test),y_test)))

# Put all feature (gene) importances into a dataframe
print('Saving feature importances...')
impdf = pd.DataFrame([nb_fi,rf_fi,svm_fi]).T # rotate so columns are for each model
impdf.columns = ['NB','RF','SVM']
impdf['gene'] = genes
impdf.to_csv('./data/importances.tsv',sep='\t',index=False)

# Model Accuracies
## SVM Test Accuracy: 0.9937888198757764
## Naive Bayes Test Accuracy: 0.9503105590062112
## Random Forest Test Accuracy: 0.9968944099378882
