from framework.datasets import RiverFlow1 
import framework.neuralNetRegression as nn
from  framework.metrics import average_relative_root_mean_squared_error
from sklearn.model_selection import train_test_split

X, y = RiverFlow1().get_numpy() 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
input_dim=len(X_train[0])
output_dim=len(y_train[0])
model = nn.Linear_NN_Model(input_dim,output_dim,'mean') 
reg = nn.NeuralNetRegressor(model,patience=1,learning_rate=0.1,print_after_epochs=1) 
reg.fit(X_train,y_train) 
res = reg.predict(X_test) 

print(average_relative_root_mean_squared_error(res,y_test))