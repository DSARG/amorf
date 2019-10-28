from framework.datasets import RiverFlow1 
import framework.neuralNetRegression as nn
from sklearn.model_selection import train_test_split

X, y = RiverFlow1().get_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model = nn.Linear_NN_Model(input_dim=len(X_train[0]),output_dim=len(y_train[0]),selector='doubleInput') 
reg = nn.NeuralNetRegressor(model,patience=8,learning_rate=0.001,print_after_epochs=100,batch_size=1000)
