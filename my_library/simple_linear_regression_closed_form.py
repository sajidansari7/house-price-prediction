import numpy as np
class simple_linear_regression_closed_form:
    def __init_(self):
        self.slope=None
        self.bias=None
    
    def fit(self,X_train,y_train):
        numerator=np.sum((X_train-np.mean(X_train))*((y_train-np.mean(y_train)))
        denominator=np.sum((X_train-np.mean(X_train))**2)
        
        slope=numerator/denominator
        bias=np.mean(y_train)-self.slope*np.mean(X_train)
    
        return {slope,bias};
    
    def predict(self,X_test):
        return self.slope*X_test+self.bias