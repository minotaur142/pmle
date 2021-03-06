# PMLE
## Python implementation of five classification algorithms using penalized maximum likelihood estimation, and King  Zeng's Approximate Bayesian Logit
Used to obtain superior binary classifications with small, separated and imbalanced datasets

### FirthLogit contains:
* Firth's Logit - small datasets, separated datasets, imbalanced datasets 
* Tuneable Firth - imbalanced datasets 
* Firth's Logit with Intercept Correction (FLIC) - imbalanced datasets 
* Firth's Logit with Added Covariate (FLAC) - imbalanced datasets 

### LogF11 contains:
* Log-F(1,1) Logistic Regression - small datasets 

**NB: Data must be passed as pandas dataframes or series**

### KingZeng contains:
* Approximate Bayesian Logit - imbalanced datasets
