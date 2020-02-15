import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import add_constant, predict, predict_proba

class logF11():
    def __init__(self,add_int=True):
        self.add_int=add_int
        '''PARAMETERS
           add_int: add intercept'''
        
    def data_augmentation(self,X,y=None):
        '''Performs log-F(1,1) data augmentation
        
           PARAMETERS
           X: A pandas dataframe of X values
           y: A pandas dataframe or series of y values
           X_only: If True, only returns augmented X values

           RETURNS
           X: augmented X values              
           y: augmented y values
           sample_weights: sample weights'''
        
        # Determine the number of pseudo
        num_rows = X.shape[1]*2
        if self.add_int==True:
            X = add_constant(X)

        # Create a new dataframe for the X variable pseudo-data
        aug_X = pd.DataFrame(0,columns=X.columns,index=(range(num_rows)))
        
        # Set one X variable to 1 in each pair of pseudo-data rows excluding the intercept
        if self.add_int==True:
            start = 1
        else:
            start = 0
        for ind, rows in enumerate(range(0,aug_X.shape[0],2),start=start):
            aug_X.iloc[rows:rows+2,ind]=1

        # Combine real and pseudo-variable dataframes
        X = pd.concat([X,aug_X]).reset_index(drop=True)
        self.aug_X = X
        
        if type(y)==type(None):
            return X

        else:
            # Augment y variable by adding a 0 and 1 for
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
        '''Calculates log-F(1,1) logistic regression coefficients,
           stores model as model and model coefficients as weights.

           PARAMETERS
           X: A pandas dataframe of X values
           y: A pandas dataframe of y values'''
        
        
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
        X = self.data_augmentation(X)
        return predict(X,self.weights)[:orig_rows]

    def predict_proba(self,X):
        orig_rows = X.shape[0]
        X = self.data_augmentation(X)
        return predict_proba(X,self.weights)[:orig_rows]