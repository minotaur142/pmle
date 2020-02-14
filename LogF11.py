import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import _add_constant, _predict, _predict_proba

class logF11():
    def __init__(self,add_int=True):
        self.add_int=add_int

    def data_augmentation(self,X,y=None,X_only=False):
        num_rows = X.shape[1]
        if self.add_int==True:
            X = _add_constant(X)

        #Augment X
        aug_X = pd.DataFrame(0,columns=X.columns,index=(range(num_rows)))
        for ind, rows in enumerate(range(0,aug_X.shape[0],2),start=1):
            aug_X.iloc[rows:rows+2,ind]=1
        X = pd.concat([X,aug_X])
        self.aug_X = X

        if X_only==True:

            return X

        else:
            #Augment y
            aug_y = pd.Series(0,index=(range(num_rows)))
            aug_y[range(1,num_rows,2)]=1
            y = pd.concat([y,aug_y])
            self.aug_y = y

            #Get sample weights
            sample_weights = np.ones(X.shape[0])
            sample_weights[-num_rows:] = 0.5
            self.sample_weights = sample_weights

            return X, y, sample_weights



    def fit(self,X,y):
        self.X = X
        self.y = y
        X, y, sample_weights = self.data_augmentation(X,y)
        sklogit = LogisticRegression(solver='newton-cg',penalty='none',fit_intercept=False)
        sklogit.fit(X,y,sample_weight=sample_weights)
        self.model = sklogit
        weights = pd.Series(sklogit.coef_.flatten(),index=X.columns)           
        self.weights = weights

    def predict(self,X):
        orig_rows = X.shape[0]
        X = self.data_augmentation(X,X_only=True)
        return _predict(X,self.weights)[:orig_rows]

    def predict_proba(self,X):
        orig_rows = X.shape[0]
        X = self.data_augmentation(X,X_only=True)
        return _predict_proba(X,self.weights)[:orig_rows]