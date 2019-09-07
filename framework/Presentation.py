import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

X,y = ds.load_WQ()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 
#print('MLSSVR')
#print(er.average__relative_root_mean_squared_error(y_test,mor.MLSSVR(0,0,'linear').fit(X_train, y_train,0.5,4).predict(X_test,X_train,y_test)))
print("St- XGBoost")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('xgb').fit(X_train, y_train).predict(X_test)))
print("###Single Target Methods") 
custom = RidgeCV() 
print("St- Custom(Ridge Regression)")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod(custom_regressor = custom).fit(X_train, y_train).predict(X_test)))
print("St- Linear")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('linear').fit(X_train, y_train).predict(X_test)))
print("St- KNN")
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('kneighbors').fit(X_train, y_train).predict(X_test)))
#print("St- MLP")
#print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod('mlp').fit(X_train, y_train).predict(X_test)))
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

'''
print("MO-CNN") 
model = mor.ConvNet(input_dim=len(X_train[0]),output_dim=len(y_train[0])) 
reg = mor.NeuronalNetRegressor(patience=8,model = model,learning_rate=0.001)
print(er.average__relative_root_mean_squared_error(y_test,reg.fit(X_train, y_train).predict(X_test)))
reg.save("CNN_WQ") 
'''
print('Linear NN') 
model = mor.NeuronalNet(input_dim=len(X_train[0]),output_dim=len(y_train[0])) 
reg = mor.NeuronalNetRegressor(patience=8,model = model)
print(er.average__relative_root_mean_squared_error(y_test,reg.fit(X_train, y_train).predict(X_test)))
reg.save("Linear_wq") 