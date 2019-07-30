import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split

X,y = ds.load_RF1()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)  
print("Single Target Methods")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('linear').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('kneighbors').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('adaboost').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('mlp').fit(X_train, y_train).predict(X_test)))
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod().fit(X_train, y_train).predict(X_test)))  

print("MO-MLP")
print(er.average__relative_root_mean_squared_error(y_test,mor.MultiLayerPerceptron().fit(X_train, y_train).predict(X_test))) 