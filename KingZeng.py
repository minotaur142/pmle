import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.tools import add_constant

class KingZeng():
    def __init__(self):
        pass
    
    def fit(self,X,y,sample_weights=None):
        if type(sample_weights)==type(None):
            sample_weights = np.ones(X.shape[0])
        sklogit = LogisticRegression(solver='newton-cg',penalty='none')
        sklogit.fit(X,y,sample_weights)
        p = sklogit.predict_proba(data)[:,1]
        coefs = np.insert(sklogit.coef_,0,sklogit.intercept_)
        
        X = add_constant(X)
        W = np.diag(p*(1-p)*sample_weights)
        inv_XWX = np.linalg.inv(np.linalg.multi_dot([X.transpose(),W,X]))
        
        Q = np.diag(np.linalg.multi_dot([X,inv_XWX,X.transpose()]))
        chi = 0.5*Q*(0.5*((1+sample_weights)*p)-sample_weights)
        bias = np.linalg.multi_dot([inv_XWX,X.transpose(),W,chi])
        B_tilde = coefs-bias
        p_tilde = sigmoid_pred(X,B_tilde)
        self.coefs = B_tilde
        
    def predict_proba(self,X):
        X = add_constant(X)
        p_tilde = sigmoid_pred(X,self.coefs)
        V_hat = np.matmul((p*(1-p)*X.transpose()),X)
        V = ((X.shape[0]/(X.shape[0]+X.shape[1]))**2)*V_hat
        C = (0.5-p_tilde)*p_tilde*(1-p_tilde)*np.linalg.multi_dot([X,V,X.transpose()])
        p_KZ = p_tilde + C
        return p_KZ
    
    def predict(self,X):
        return self.predict_proba(X).round()
        