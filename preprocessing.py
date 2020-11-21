import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
#Custom transformer we wrote to engineer features ( bathrooms per bedroom and/or how old the house is in 2019  ) 
#passed as boolen arguements to its constructor
class TargetTransformer(BaseEstimator, TransformerMixin):
#Class Constructor
    def __init__(self):
        self = self

#Return self, nothing else to do here
    def fit( self, X ):
        return self 

#Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X):
        #Check if needed 
        X = np.array([1 if y=='yes' else 0 for y in X])
        #returns a numpy array
        return X.reshape(-1,1)

class LogPlus1Transformer(BaseEstimator, TransformerMixin):
#Class Constructor
    def __init__(self):
        self=self
    
    def fit(self, X):
        return(self)
    
    def transform(self, X):
        X = np.log(X+1)
        return(X.values)

class CampaignTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self = self
    
    def fit(self, X):
        return(self)

    def transform(self, X):
        X = np.array(pd.cut(X, np.quantile(X, [0,0.25,0.5,0.75,1]), duplicates='drop', include_lowest=True).astype(str))
        return(X.reshape(-1,1))