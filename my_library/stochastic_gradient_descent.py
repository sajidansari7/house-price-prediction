import numpy as np

class StochasticGradientDescent:
    
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None
        self.loss=[]
        
    def fit(self,X_train,y_train):
        n_samples,n_features=X_train.shape
        
        self.weights=np.ones(n_features)
        self.bias=0.0
        
        for epoch in range(self.epochs):
            for index in range(n_samples):
                sample_index=np.random.randint(0,n_samples)
                
                y_pred=np.dot(X_train[sample_index],self.weights)+self.bias
                error=y_train-y_pred
                
                bias_derivative=-2*(y_train[sample_index]-y_pred)
                self.bias=self.bias-(self.learning_rate*bias_derivative)
                
                weights_derivative=-2*np.dot((y_train[sample_index]-y_pred,X_train[sample_index])
                self.weights=self.weights-(self.learning_rate*weights_derivative)
                
                loss=np.mean(error**2)
                self.loss.append(loss)
                
    def predict(self,X_test):
        return np.dot(X_test,self.weights)+self.bias
    
                
            
            