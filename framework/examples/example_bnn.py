from framework.datasets import RiverFlow1 
import framework.neuralNetRegression as nn 
from  framework.probabalisticRegression import BayesianNeuralNetworkRegression
from  framework.metrics import average_relative_root_mean_squared_error
from sklearn.model_selection import train_test_split 
import numpy as np
X, y = RiverFlow1().get_numpy() 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
input_dim=len(X_train[0])
output_dim=len(y_train[0]) 
model = BayesianNeuralNetworkRegression(patience=3,use_gpu=True, batch_size=2000) 

model.fit(X_train,y_train) 
stds,means = model.predict(X_test,y_test) 

print(average_relative_root_mean_squared_error(means,y_test)) 

import matplotlib.pyplot as plt 
from matplotlib import style

style.use('seaborn')

plt.hist(np.mean(stds,1), facecolor='green', alpha=0.6)

plt.title('Mean Std. Per Sample')
plt.xlabel('Mean Standard Deviation')
plt.ylabel('Counts')
plt.show() 

plt.bar(range(8),np.mean(stds,0), facecolor='green', align='center',alpha=0.6)
 
plt.title('Std. by Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Mean Standard Deviation')
plt.show() 