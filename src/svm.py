import sys
import numpy as np
import pandas as pd

# Implementation of a linear kernel SVM to be fit using gradient optimziation

# Arguments
## x is data matrix, rows are observations columns are features
## y is class label
## w is current estimation of decision boundary

def dist(X,y,w):
    # distance of each sample (row in x) to the decision boundary w
    return [np.max([0,1-y[i]*(w @ X[i])]) for i in range(X.shape[0])]

def objective(x,y,w):
    # objective function for scoring SVM (w), to be minimized
    return (1/2)*np.dot(w,w)+np.sum(dist(x,y,w))

def update_boundary(x,y,w,C):
    dw = np.zeros(len(w)) # initialize updates for each w_i
    # gradient descent to update decision boundary vector w
    for i,d in enumerate(dist(x,y,w)):
        if d == 0: # sample is not an SV
           dw = w
        else:
            dw = w - C*y[i]*x[i]
    return dw

def predict(x,w):
    # predict labels on data given a decision boundary
    #x = np.column_stack((np.ones(x.shape[0]),x))
    # get sign of each element (sample) of x*w, -1 and 1 are the respective class labels
    return np.sign(np.matmul(x,w))

def train_svm(x,y,lr=1e-1,C=30,tol=1e-2,iter=200):
    # train an svm on observed data
    # Initialize
    w = np.random.random(x.shape[1])
    objectives = [objective(x,y,w)+1,objective(x,y,w)]
    # until convergence predict and update boundary
    i = 1
    for i in range(iters):
        w -= lr*update_boundary(x,y,w,C)
        objectives.append(objective(x,y,w))
        if i % 15 == 0:
            print('iter: {}, obj:{}'.format(i,objectives[-1]))
        i += 1
        if np.abs(objectives[-1]-objectives[-2]) > tol:
            break
    return w,predict(x,w),objectives[1:]


if __name__ == '__main__':
    data = pd.read_csv('./data/data.tsv',sep='\t')
    y = data['labels'].values
    y[y == 0] = -1
    X = data[[x for x in data.columns if x != 'labels']].values
    X = np.hstack([X,np.ones(X.shape[0]).reshape(-1,1)])
    w,yn,objs = sv.train_svm(X,y)
    print('decision boundary')
    print(len(np.where(yn == y))/len(yn))
    print(len(set(w)))
