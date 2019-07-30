import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arff, numpy as np 
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression  
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor 

import numpy as np
from numpy import mean 

import datasets as ds 
import earlyStopping as early
import error as er
import multiOutputRegressors as mor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' Example Data
X, y = make_regression(n_samples=10000, n_features=50, n_targets=3, random_state=1)
''' 
X,y = ds.load_RF1()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) 

n_samples = len(X_train) 
n_features = len(X_train[0]) 
n_targets = len(y_train[0])

X_train_t = torch.from_numpy(normalize(X_train)).float().reshape(6843,1,64) 
X_test_t = torch.from_numpy(normalize(X_test)).float().reshape(2282,1,64)
y_train_t = torch.from_numpy(y_train).float()

class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvNet, self).__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(64*15, 1000)
        self.fc2 = nn.Linear(1000, output_dim)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet(64,8)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


X_train_t, y_train_t = X_train_t.to(DEVICE), y_train_t.to(DEVICE)  # <-- here 


# Train step
model.train()  # <-- here
stopper = early.earlyStopping(20) 
stop = False
epochs = 0
while(stop == False):
    optimizer.zero_grad()

    y_pred_t = model(X_train_t)
    loss = loss_fn(y_pred_t, y_train_t)
    loss.backward()
    optimizer.step() 
    error = er.average__relative_root_mean_squared_error(y_train_t.detach().numpy(),y_pred_t.detach().numpy())
    if epochs%100 == 0: 
        print(error)
    epochs += 1 
    stop = stopper.stop(error)

print(epochs)
# Eval
model.eval()  # <-- here
with torch.no_grad():
    y_pred_t = model(X_test_t)  

print('aRRMSE') 
print(er.average__relative_root_mean_squared_error(y_test,y_pred_t.detach().numpy())) 