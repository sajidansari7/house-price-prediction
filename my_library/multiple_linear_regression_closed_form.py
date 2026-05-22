import numpy as np
class multiple_linear_regression_closed_form:
    def __init__(self):
        self.weights=None
        self.intercept=None
    
    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        
        #calculate coefficients
        coefficients=np.linalg.pinv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        
        self.intercept=coefficients[0]
        self.weights=coefficients[1:]
        
       
        
    def predict(self,X_test):
        return np.dot(X_test,self.weights)+self.intercept
        