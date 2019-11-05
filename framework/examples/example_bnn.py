from framework.datasets import RiverFlow1 
import framework.neuralNetRegression as nn 
from  framework.probabalisticRegression import BayesianNeuralNetworkRegression
from  framework.metrics import average_relative_root_mean_squared_error
from sklearn.model_selection import train_test_split

X, y = RiverFlow1().get_numpy() 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
input_dim=len(X_train[0])
output_dim=len(y_train[0]) 
model = BayesianNeuralNetworkRegression(patience=3) 
model.fit(X_train,y_train) 
stds,mean = model.predict(X_test,y_test) 

print(average_relative_root_mean_squared_error(stds,y_test))