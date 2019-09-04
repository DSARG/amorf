from numpy import mean, sqrt 
import torch as t

def average_relative_error(y_test,y_pred):
    return sum(sum(abs(y_test-y_pred)/y_test)/len(y_test))/len(y_test[0,:])

def average__relative_root_mean_squared_error(y_test,y_pred): 
    return sum(sqrt(sum((y_test-y_pred)**2) / sum((y_test-mean(y_test,axis=0))**2))) / len(y_pred[0,:])

def tensor_average__relative_root_mean_squared_error(y_pred,y_test): 
    return t.sum(t.sqrt(t.sum(t.pow((y_test-y_pred),2)) / t.sum(t.pow((y_test-t.mean(y_test,dim=0)),2)))) / len(y_pred[0,:])