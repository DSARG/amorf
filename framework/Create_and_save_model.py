import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split

X,y = ds.load_RF1()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

print("MO-CNN") 
model = mor.ConvNet(input_dim=len(X_train[0]),output_dim=len(y_train[0])) 
reg = mor.NeuronalNetRegressor(patience=8,model = model,batch_size=2000)
print(er.average__relative_root_mean_squared_error(y_test,reg.fit(X_train, y_train).predict(X_test)))
reg.save("test")