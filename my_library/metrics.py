
def mean_squared_error(y_test,y_pred):
    return np.mean((y_test-y_pred)**2)
    
def root_mean_squared_error(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
    
def mean_absolute_error(y_test,y_train):
    return np.mean(np.abs(y_test-y_pred))
    
def r2_score(y_test,y_pred):
    error_wrt_regline=np.sum((y_test-y_pred)**2)
    error_wrt_mean=np.sum((y_test-np.mean(y_test))**2)
    
    return 1-(error_wrt_regline/error_wrt_mean)
    
    
    