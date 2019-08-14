import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

X,y = ds.load_EDM()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

print("Single Target Methods") 
custom = RidgeCV()
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('linear').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod(custom_regressor = custom).fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('xgb').fit(X_train, y_train).predict(X_test)))

print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('kneighbors').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('adaboost').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('mlp').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('svr').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod().fit(X_train, y_train).predict(X_test)))  
''' 


print('MO-RT') 
print(er.average__relative_root_mean_squared_error(y_test,mor.MultiOutputRegressionTree().fit(X_train,y_train).predict(X_test)))
print("MO-MLP")
print(er.average__relative_root_mean_squared_error(y_test,mor.MultiLayerPerceptron().fit(X_train, y_train).predict(X_test))) 

print("MO-NN") 
print(er.average__relative_root_mean_squared_error(y_test,mor.NeuronalNetRegressor(patience=2,selector='nn').fit(X_train, y_train).predict(X_test))) 
print("MO-CNN") 
print(er.average__relative_root_mean_squared_error(y_test,mor.NeuronalNetRegressor(patience=5,selector='cnn').fit(X_train, y_train).predict(X_test)))
'''