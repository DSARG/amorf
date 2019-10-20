import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

X,y = ds.load_RF1()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

print("\n###Single Target Methods") 
print("St- XGBoost")
print(er.average__relative_root_mean_squared_error(y_test, mor.SingleTargetMethod('xgb').fit(X_train, y_train).predict(X_test)))
print("St- Custom(Ridge Regression)")
custom = RidgeCV() 
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod(custom_regressor = custom).fit(X_train, y_train).predict(X_test)))
print("St- Linear")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('linear').fit(X_train, y_train).predict(X_test)))
print("St- KNN")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('kneighbors').fit(X_train, y_train).predict(X_test)))
print("St- MLP")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('mlp').fit(X_train, y_train).predict(X_test)))
print("St- SVR")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('svr').fit(X_train, y_train).predict(X_test)))
print("St- AdaBoost")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('adaboost').fit(X_train, y_train).predict(X_test)))
print("St- GradientBoosting")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod().fit(X_train, y_train).predict(X_test)))  

print('###DecisionTree') 
print(er.average__relative_root_mean_squared_error(y_test,mor.MultiOutputRegressionTree().fit(X_train,y_train).predict(X_test)))
print("###MLP")
print(er.average__relative_root_mean_squared_error(y_test,mor.MultiLayerPerceptron().fit(X_train, y_train).predict(X_test))) 

print('Linear NN') 
model = mor.NeuronalNetRegressor.load(load_path = 'Linear_rf1_best.ckpt')
reg = mor.NeuronalNetRegressor(model=model)
print(er.average__relative_root_mean_squared_error(y_test,reg.predict(X_test)))

print("CNN") 
model = mor.NeuronalNetRegressor.load(load_path = 'CNN_rf1_best.ckpt')
reg = mor.NeuronalNetRegressor(model = model) 
print(er.average__relative_root_mean_squared_error(y_test,reg.predict(X_test)))
