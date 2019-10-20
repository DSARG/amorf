from sklearn.datasets import make_regression 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import normalize
import arff, numpy as np
from numpy import mean

'''average relative error''' 
def average_relative_error(y_test,y_pred,dim):
    result = 0
    for i in range(0,dim): 
        sum_of_relative_error = 0
        for j in range(0, len(y_test)):
            sum_of_relative_error += abs(y_test[j,i]-y_pred[j,i])/abs(y_test[j,i]) #not sure wether to use abs()here
        avg_sre = sum_of_relative_error/len(y_test) 
        result += avg_sre 
    return result/dim

''' average relative root mean squared error''' 
def average__relative_root_mean_squared_error(y_test,y_pred,dim): 
    result = 0
    for i in range(0,dim):
        sum_squared_error = 0
        for j in range(0, len(y_test)): 
            sum_squared_error += ((y_test[j,i]-y_pred[j,i])**2)
        sum_squared_deviation = 0

        for k in range(0, len(y_test)):
            sum_squared_deviation += (y_test[k,i]- mean(y_test[:,i]))**2
        
        result += (sum_squared_error/sum_squared_deviation)**(1/2) 
    return result/dim


'''Prepare Data'''
''' Example Data
X, y = make_regression(n_samples=10000, n_features=50, n_targets=3, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) 
'''
'''Water Quality'''
dataset = arff.load(open('wq.arff'))
data = np.array(dataset['data'])
X = normalize(data[:,16:30],norm='l1') 
y= data[:,0:16]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) 

''' Single Target Method '''
y_pred_st = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X_train, y_train).predict(X_test)

error_ar = average_relative_error(y_test,y_pred_st,3)
error_aRRMSE = average__relative_root_mean_squared_error(y_test,y_pred_st,3)
print("Error ST AR:") 
print(error_ar) 
print('Error ST aRRMSE:') 
print(error_aRRMSE) 

print(y_pred_st) 
print(y_test)

'''Decission Tree Method''' 
regressor_1 = DecisionTreeRegressor(max_depth=10) 
regressor_1.fit(X_train,y_train) 
y_pred_regtree1 = regressor_1.predict(X_test) 

reg1_error = average_relative_error(y_test,y_pred_regtree1,3)
reg1_error_aRRMSE = average__relative_root_mean_squared_error(y_test,y_pred_regtree1,3)
print("Error RT AR:") 
print(reg1_error)  
print("Error Reg1 aRRMSE:") 
print(reg1_error_aRRMSE)
