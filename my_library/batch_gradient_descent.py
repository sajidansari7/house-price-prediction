import numpy as np

class BatchGradientDescent:
    
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None
        self.loss=[]
        
    def fit(self,X_train,y_train):
        n_samples,n_features=X_train.shape
        
        self.weights=np.zeros(n_features)
        self.bias=0.0
        
        for epoch in range(self.epochs):
            
            y_pred=np.dot(X_train,self.weights)+self.bias
            
            error=y_train-y_pred
            
            bias_derivative=-2*np.mean(y_train-y_pred)
            self.bias=self.bias-(self.learning_rate*bias_derivative)
            
            weights_derivative=-2*np.dot((y_train-y_pred),X_train)/n_samples
            self.weights=self.weights-(self.learning_rate*weights_derivative)
            
            loss=np.mean(error**2)
            self.loss.append(loss)
            
            
    def predict(self,X_test):
        return np.dot(X_test,self.weights)+self.bias