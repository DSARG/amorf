from pprint import pprint

import arff
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import mean
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''NeuronalNet'''
class NeuronalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.normalization = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        out = self.normalization(x) 
        out = self.fc2(x)
        return out

''' average relative root mean squared error''' 
def average__relative_root_mean_squared_error(y_test,y_pred,dim): 
    result = 0
    for i in range(0,dim):
        sum_squared_error = 0
        sum_squared_deviation = 0
        for j in range(0, len(y_test)): 
            sum_squared_error += ((y_test[j,i]-y_pred[j,i])**2)
            sum_squared_deviation += (y_test[j,i]- mean(y_test[:,i]))**2
        
        result += (sum_squared_error/sum_squared_deviation)**(1/2) 
    return result/dim  

def getMinValueAndIndex(array_in): 
    min_value = 9999
    min_index = -1
    for i in range(0,len(array_in)): 
        if (array_in[i] < min_value): 
            min_value = array_in[i]
            min_index = i
    return min_value,min_index

    

n_features = [10,30,60,100,200] 
n_instances = [100,1000,10000,50000] 
n_targets = [2,4,8,16,32] 

csv_data = []
csv_data.append(["instances","features","targets","Single Target","Regression Tree","RT Depth","NN", "NN Steps"])
print("instances\\features\\targets \t\t Single Target\t Regression Tree\t NN")
for instances in n_instances: 
    for features in n_features: 
        for targets in n_targets:
            '''Prepare Test Data'''
            X, y = make_regression(n_samples=instances, n_features=features, n_targets=targets, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
            
            ''' Single Target Method '''
            y_pred_st = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X_train, y_train).predict(X_test)
            error_y_pred_st = average__relative_root_mean_squared_error(y_test,y_pred_st,targets)

            '''MultiOutputRegressionTress''' 
            results_mort = []
            for i in range(1,11):
                regressor_1 = DecisionTreeRegressor(max_depth=i) 
                regressor_1.fit(X_train,y_train) 
                y_pred_regtree1 = regressor_1.predict(X_test)
                results_mort.append(average__relative_root_mean_squared_error(y_test,y_pred_regtree1,targets))  
            
            error_y_pred_regtree1, depth  = getMinValueAndIndex(results_mort)

            '''Neuronal Net'''
            model = NeuronalNet(features,targets,).to(DEVICE)  # <-- here
            loss_fn = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            
            n_samples = len(X_train) 
            X_train_t = torch.from_numpy(X_train).float() 
            X_test_t = torch.from_numpy(X_test).float()
            y_train_t = torch.from_numpy(y_train.reshape((n_samples, len(y_train[0])))).float()
            X_train_t, y_train_t = X_train_t.to(DEVICE), y_train_t.to(DEVICE)  # <-- here 

            result_nn=[] 
            steps = [100,500,1000,5000]
            for step_number in steps:    
                # Train step
                model.train()  # <-- here
                for i in range(0,step_number):
                    optimizer.zero_grad()

                    y_pred_t = model(X_train_t)
                    loss = loss_fn(y_pred_t, y_train_t)
                    loss.backward()
                    optimizer.step() 

                # Eval
                model.eval()  # <-- here
                with torch.no_grad():
                    result_nn.append(average__relative_root_mean_squared_error(y_test,model(X_test_t).detach().numpy(),targets)) 

            error_y_pred_NN_t, step_index = getMinValueAndIndex(result_nn) 
            steps_nn = steps[step_index]

           
            '''Print Results''' 
            print("{}\\{}\\{}\t \t{}\t{}\t{}\t{}\t{}".format(instances,features,targets,error_y_pred_st,error_y_pred_regtree1,depth,error_y_pred_NN_t,steps_nn))  
            csv_data.append([instances,features,targets,error_y_pred_st,error_y_pred_regtree1,depth,error_y_pred_NN_t,steps_nn])
with open('comparinsonResults.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
    f.close()