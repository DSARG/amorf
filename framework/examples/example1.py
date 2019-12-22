from framework.datasets import RiverFlow1, EDM, WaterQuality
import framework.neuralNetRegression as nn
from framework.problemTransformation import AutoEncoderRegression
from framework.metrics import average_relative_root_mean_squared_error
from sklearn.model_selection import train_test_split
import torch

X, y = RiverFlow1().get_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
input_dim = len(X_train[0])
output_dim = len(y_train[0])
#model = nn.Linear_NN_Model(input_dim, output_dim, 'max',p_dropout_1=0.1, p_dropout_2=0.1)
reg = AutoEncoderRegression('xgb', patience=5, training_limit=None)
reg.fit(X_train, y_train)
res = reg.predict(X_test)

print(average_relative_root_mean_squared_error(
    torch.from_numpy(res), torch.from_numpy(y_test)))
print(average_relative_root_mean_squared_error(res, y_test))
