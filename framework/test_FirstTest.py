
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arff, numpy as np 
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression  

from numpy import mean 

import datasets as ds 
import error as er
import multiOutputRegressors as mor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' Example Data
X, y = make_regression(n_samples=10000, n_features=50, n_targets=3, random_state=1)
''' 
X,y = ds.load_EDM()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) 

n_samples = len(X_train) 
n_features = len(X_train[0]) 
n_targets = len(y_train[0])

X_train_t = torch.from_numpy(normalize(X_train)).float() 
X_test_t = torch.from_numpy(normalize(X_test)).float()
y_train_t = torch.from_numpy(y_train).float()

class BasicNeuronalNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_features):
        super().__init__()
        middleLayerNeurons = max([input_dim,output_dim])
        self.batchNorm1 = nn.BatchNorm1d(middleLayerNeurons)
        self.batchNorm2 = nn.BatchNorm1d(middleLayerNeurons)

        self.fc1 = nn.Linear(input_dim,middleLayerNeurons)
        self.fc2 = nn.Linear(middleLayerNeurons,middleLayerNeurons)
        self.fc3 = nn.Linear(middleLayerNeurons,output_dim)

    def forward(self, x):
        out = self.fc1(x)  
        out = self.batchNorm1(out)  
        out = F.relu(out) 
        out = F.dropout(out,0.2) 
        out = self.fc2(out) 
        out = self.batchNorm2(out)  
        out = F.relu(out) 
        out = F.dropout(out,0.2)
        out = self.fc3(out)
      
        return out


model = BasicNeuronalNet(n_features,n_targets,n_features).to(DEVICE)  # <-- here
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


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
        print(er.average__relative_root_mean_squared_error(y_train_t.detach().numpy(),y_pred_t.detach().numpy()))

# Eval
model.eval()  # <-- here
with torch.no_grad():
    y_pred_t = model(X_test_t)  

print('aRRMSE') 
print(er.average__relative_root_mean_squared_error(y_test,y_pred_t.detach().numpy())) 
print(er.average__relative_root_mean_squared_error(y_test,mor.SingleTargetMethod().fit(X_train, y_train).predict(X_test)))

