
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

class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_features):
        super().__init__()
        self.fc1 = nn.BatchNorm1d(14)
        self.fc2 = nn.Linear(14,64)
        self.fc3 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,16)

    def forward(self, x):
        out = self.fc1(x) 
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc5(out)
        return out

''' average relative root mean squared error''' 
def average__relative_root_mean_squared_error(y_test,y_pred,dim): 
    result = 0
    for i in range(0,dim):
        sum_squared_error = 0
        sum_squared_deviation = 0
        y_test_mean = mean(y_test[:,i])
        for j in range(0, len(y_test)): 
            sum_squared_error += (y_test[j,i]-y_pred[j,i])**2
            sum_squared_deviation += (y_test[j,i]- y_test_mean)**2
        
        result += (sum_squared_error/sum_squared_deviation)**(1/2) 
    return result/dim

model = ConvNet(n_features,n_targets,n_features).to(DEVICE)  # <-- here
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


X_train_t, y_train_t = X_train_t.to(DEVICE), y_train_t.to(DEVICE)  # <-- here 


# Train step
model.train()  # <-- here
for i in range(0,1000):
    optimizer.zero_grad()

    y_pred_t = model(X_train_t)
    loss = loss_fn(y_pred_t, y_train_t)
    loss.backward()
    optimizer.step() 
    if i % 100 is 0: 
        print(average__relative_root_mean_squared_error(y_train_t.detach().numpy(),y_pred_t.detach().numpy(),n_targets))

# Eval
model.eval()  # <-- here
with torch.no_grad():
    y_pred_t = model(X_test_t)  

print('aRRMSE') 
print(average__relative_root_mean_squared_error(y_test,y_pred_t.detach().numpy(),n_targets))

