import numpy as np
class LassoRegressionGradientDescent:
    
    def __init__(self,epochs,learning_rate,alpha):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.loss=[]
    def fit(self,X_train,y_train):
        n_samples, n_features = X_train.shape
        
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        for i in range(self.epochs):
            y_pred = np.dot(X_train, self.coef_) + self.intercept_
            error = y_train - y_pred
            
            # standard MSE gradients
            bias_derivative = -2 * np.mean(error)
            coef_derivative = -2 * np.dot(error, X_train) / n_samples
            
            # add Ridge penalty to weights only (not bias)
            coef_derivative +=  self.alpha * np.sign(self.coef_)
            
            self.intercept_ = self.intercept_ - (self.learning_rate * bias_derivative)
            self.coef_ = self.coef_ - (self.learning_rate * coef_derivative)
            
            loss = np.mean(error ** 2)
            self.loss.append(loss)
        
        
        
    
    def predict(self,X_test):
        
        return np.dot(X_test,self.coef_) + self.intercept_

