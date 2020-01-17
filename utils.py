#helper functions
import numpy as np
import pandas as pd

def _add_constant(X):
    if isinstance(X, np.ndarray):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
    elif isinstance(X, pd.DataFrame):
        intercept = pd.Series(1,name='const',index=X.index)
        X = pd.concat([intercept,X],axis=1)
    else:
        raise TypeError("Data must be pandas dataframe or numpy array")
    return X

def _sigmoid_pred(X, weights):
    z = np.dot(X,weights)
    sig =  1/(1 + np.exp(-1*z))
    sig = np.clip(sig,.000001,.999999)
    return sig

def _hat_diag(X,weights):
    Xt = X.transpose()

    #Get diagonal of error
    y_pred = _sigmoid_pred(X,weights)
    W = np.diag(y_pred*(1-y_pred))

    #Calculate Fisher Information Matrix
    I = np.linalg.multi_dot([Xt,W,X]) 

    #Get Diagonal of Hat Matrix
    hat = np.linalg.multi_dot([W**0.5,X,np.linalg.inv(I),Xt,W**0.5])
    hat_diag = np.diag(hat)
    return hat_diag

def _information_matrix(X,weights):
    Xt = X.transpose()

    #Get diagonal of error
    y_pred = _sigmoid_pred(X,weights)
    W = np.diag(y_pred*(1-y_pred))

    #Calculate Fisher Information Matrix
    I = np.linalg.multi_dot([Xt,W,X])
    return I

def _predict_proba(X):
    if X.shape[1]==X.shape[1]-1:
        X = _add_constant(X)
    preds = _sigmoid_pred(X,weights)
    return preds

def _predict(X):
    if X.shape[1]==X.shape[1]-1:
        X = _add_constant(X)
    preds = _sigmoid_pred(X,weights).round()
    return preds
                             