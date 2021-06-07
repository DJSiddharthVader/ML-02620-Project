# 02-620: Machine Learning for Scientists

## Final Project

### Instructions

First to meet the depedencies use conda to install the provided environment.
```
conda env create --file environment.yml
conda activate ml02620
```

Run the folioing commands to replicate my data and figures.
The commented commands generate the data but this is provided for you in `./data/` already.
```
#bash ./src/prepData.sh* # may take a few minutes and heavy RAM usage
#python prepData.py # prepare data for learning
#python differentialExpression.py # preform differential expression on GTEx/TCGA data
#python featureImportances.py # train models and generate feature importances
python featureAnalysis.py # generate figures in report
```

Note that these scripts do not use my implementations of SVM or Naive Bayes for logistical reasons. If you want to test those run the following (requires numpy)

```
python ./src/svm.py
python ./src/naive_bayes.py
```
