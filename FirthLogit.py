import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
import statsmodels.api as sm
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from utils import _add_constant,_hat_diag,_sigmoid_pred,_sigmoid_pred, _information_matrix, _predict, _predict_proba, _FLIC, _FLAC_aug, _FLAC_pred_aug

class Firth_Logit():
        def __init__(self,num_iters=10000, lr=0.01,add_int=True, metric = None, readout_rate = None, lmbda=0.5,FLAC=False, FLIC=False):
            
            '''PARAMETERS
               num_iters: number of iterations in gradient descent
               lr: learning rate
               add_int: add intercept
               metric: use 'log_loss' or 'recall_score' in progress readouts and lr reduction schedule
               readout_rate: number of iterations between readouts
               
               MODIFICATIONS FOR RARE EVENTS
               lmbda: tuneable parameter for target mean prediction value               
               FLAC: perform Firth Logistic regression with added covariate
               FLIC: perform Firth Logistic regression with Intercept Correction'''

            self.lr = lr
            self.num_iters = num_iters
            self.add_int = add_int
            self.lmbda=lmbda
            self.FLAC = FLAC
            self.FLIC=FLIC
            self.metric = metric
            self.readout_rate = readout_rate
        
        def firth_gd(self,X,y,weights):
            y_pred = _sigmoid_pred(X=X,weights=weights)
            H =_hat_diag(X,weights)
            I = _information_matrix(X,weights)
            U = np.matmul((y -y_pred + self.lmbda*H*(1 - 2*y_pred)),X)
            weights += np.matmul(np.linalg.inv(I),U)*self.lr
            return weights
        
        def fit(self,X,y):
            #add intercept if necessary
            orig_X = X
            if self.add_int==True:
                X =_add_constant(X)
            self.X = X
            self.y = y

            #initialize weights
            weights=np.ones(X.shape[1])
            
            #initialize metric infrastructure if necessary 
            if self.metric != None:
                scores = []
            if self.metric == 'log_loss':
                metric = log_loss
                min_max = min
            elif self.metric == 'recall_score':
                metric = recall_score
                min_max = max
            
            #Perform gradient descent
            for i in range(self.num_iters):
                weights = self.firth_gd(X,y,weights)
            
                if self.metric != None:
                    proba = _sigmoid_pred(X,weights)
                    if self.metric == 'recall_score':
                        proba = proba.round()
                    score = metric(y,proba)

                    #Print metric
                    scores.append(score)
                    if i%self.readout_rate==0:
                        print('Batch {} Recall: {}'.format((i),score))

                    #Reduce learning rate if necessary 
                    if (i > 10) & (min_max(scores) not in scores[-10:]):
                        self.lr = self.lr*0.9
                
            
            if (self.FLAC==True)&(self.FLIC==True):
                X,y,aug_sample_weights=_FLAC_aug(X,y,weights)
                self.X = X
                self.y = y
                sklogit = LogisticRegression(solver='newton-cg',penalty='none',fit_intercept=False)
                sklogit.fit(X,y,sample_weight=aug_sample_weights)
                weights = sklogit.coef_[1:]
                eta = np.dot(orig_X,weights)
                target = y-eta
                b0_model = sm.OLS(target,np.ones(y.shape[0])).fit()
                b0 = b0_model.params[0]
                weights = np.insert(weights,0,b0)
            
            elif self.FLAC==True:
                X,y,aug_sample_weights=_FLAC_aug(X,y,weights)
                self.X = X
                self.y = y
                sklogit = LogisticRegression(solver='newton-cg',penalty='none',fit_intercept=False)
                sklogit.fit(X,y,sample_weight=aug_sample_weights)
                weights = sklogit.coef_
                
            
            elif self.FLIC==True:
                weights = weights[1:]
                eta = np.dot(orig_X,weights)
                target = y-eta
                b0_model = sm.OLS(target,np.ones(y.shape[0])).fit()
                b0 = b0_model.params[0]
                weights = np.insert(weights,0,b0)
            
            weights = pd.Series(weights.flatten(),index=self.X.columns)
            self.weights = weights
            
            I = _information_matrix(X,weights)
            hat_matrix_diag = _hat_diag(X,weights)
            Hessian = -I
            y_pred = _sigmoid_pred(X,weights)
            
            self.I = I
            self.hat_matrix_diag = hat_matrix_diag
            self.Hessian = Hessian
            
            
            self.log_likelihood = (y*np.log(y_pred)+(1-y)*np.log(1-y_pred)).sum()+0.5*np.log(np.linalg.det(I))
            
            
        def marginal_effects(self,values=None):
            '''PARAMETERS
               values: user-specified X values
               
               RETURNS
               marginal effects at mean X variable values
               mean of marginal effects for all rows
               marginal effects at user-specified values'''
                
            def at_specific_values(self,values):
                n_features = self.weights.shape[0]
                if values.shape[0]==n_features-1:
                    values = _add_constant(values)
                
                p = _sigmoid_pred(values,self.weights)
                effs = np.ones(n_features)
                for i in range(n_features):
                    weights_copy = self.weights.copy()
                    weights_copy[i]+=1
                    new_p =_sigmoid_pred(values,weights_copy)
                    effs[i] = new_p-p
                return effs
            
            #at mean column values
            column_means = self.X.mean()
            at_means = at_specific_values(weights=column_means)

            #find marginal effects for each row and take mean
            averaged_marg_effs = np.ones((self.X.shape[0],self.X.shape[1]))
            for i in range(self.X.shape[0]):
                row = self.X.iloc[i]
                p = _sigmoid_pred(row,self.weights)
                for j in range(self.weights.shape[0]):
                    weights_copy = self.weights.copy()
                    weights_copy[j]+=1
                    new_p =_sigmoid_pred(row,weights_copy)
                    eff = new_p-p
                    averaged_marg_effs[i,j] = eff
                ame = pd.DataFrame(averaged_marg_effs.mean(axis=0),index=self.X.columns, columns=['mean'])
                ame['at_means'] = at_means
            #user requested
            if (type(values)==numpy.ndarray) | (type(values)==pandas.core.series.Series):
                user_requested = at_specific_values(values)
                ame['requested_values'] = user_requsted
            return ame