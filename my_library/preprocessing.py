import numpy as np

class StandardScaler:
    
    def fit(self,X):
        self.mean=np.mean(X,axis=0)
        
        self.standard_deviation=np.std(X,axis=0)+1e-8
        
    def transform(self,X):
        return (X-self.mean)/self.standard_deviation
        
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)