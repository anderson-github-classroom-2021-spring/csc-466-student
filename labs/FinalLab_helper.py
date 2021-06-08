import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class NBClassifier(BaseEstimator, ClassifierMixin):
    @staticmethod
    def compute_priors(y):
        priors = {}
        return priors

    @staticmethod
    def specific_class_conditional(x,xv,y,yv):
        prob = None
        return prob

    @staticmethod
    def class_conditional(X,y):
        probs = {}
        return probs

    @staticmethod
    def posteriors(probs,priors,x):
        post_probs = {}
        return post_probs
    
    
    def __init__(self):
        ## Your solution here
        pass

    def fit(self, X, y):
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Your solution here
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        return np.array(predictions)
    
class PCA( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self):
        # Your solution here
        pass
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        # Your solution here
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        Xt = None
        # Your solution here
        return Xt