

def normalize(X_train):
    mean=np.mean(X_train,axis=0)
    standard_deviation=np.std(X_train,axis=0)
    
    return (X_train-mean)/standard_deviation