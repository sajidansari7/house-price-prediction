import numpy as np
class multiple_linear_regression(closed_form):
    def __init_(self):
        self.weights=None
        self.intercept=None
    
    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        
        #calculate coefficients
        coefficients=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        
        intercept=coefficients[0]
        weights=coefficient[1:]
        
        return {intercept,weights}
        
    def predict(X_test):
        return np.dot(X_test,self.weights)+self.intercept
        