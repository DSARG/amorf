
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arff, numpy as np
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean



dataset = arff.load(open('wq.arff'))
data = np.array(dataset['data'])
X = data[:,16:30] #row 16-30 
y= data[:,0:16] #row 0-14
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) 

n_samples = len(X_train) 
n_features = len(X_train[0]) 
n_targets = len(y_train[0])

X_train_t = torch.from_numpy(X_train).float() 
X_test_t = torch.from_numpy(X_test).float()
y_train_t = torch.from_numpy(y_train.reshape((n_samples, len(y_train[0])))).float()
y_test_t = torch.from_numpy(y_test.reshape((len(y_test),len(y_test[0])))).float()

class LinReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.beta = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.beta(X)

''' average relative root mean squared error''' 
def average__relative_root_mean_squared_error(y_test,y_pred,dim): 
    result = 0
    for i in range(0,dim):
        sum_squared_error = 0
        for j in range(0, len(y_test)): 
            sum_squared_error += ((y_test[j,i]-y_pred[j,i])**2)
        sum_squared_deviation = 0
        for k in range(0, len(y_test)):
            sum_squared_deviation += (y_test[k,i]- mean(y_test[:,i]))**2
        
        result += (sum_squared_error/sum_squared_deviation)**(1/2) 
    return result/dim

model = LinReg(n_features,n_targets).to(DEVICE)  # <-- here
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


X_train_t, y_train_t = X_train_t.to(DEVICE), y_train_t.to(DEVICE)  # <-- here 


# Train step
model.train()  # <-- here
optimizer.zero_grad()

y_pred_t = model(X_test_t)
loss = loss_fn(y_pred_t, y_test_t)
loss.backward()
optimizer.step()

print('aRRMSE') 
print(average__relative_root_mean_squared_error(y_test,y_pred_t.detach().numpy(),n_targets))
'''
# Eval
model.eval()  # <-- here
with torch.no_grad():
    y_ = model(X)  
'''
