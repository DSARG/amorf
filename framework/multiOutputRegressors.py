#For the ST Methods
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR 
import xgboost as xgb
#For the MLP
from sklearn.neural_network import MLPRegressor 
#For MORT 
from sklearn.tree import DecisionTreeRegressor
#For the NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import error as er
import earlyStopping as early  
from abstract_NeuronalNet import AbstractNeuronalNet
from numpy import mean 
from sklearn.model_selection import train_test_split

# TODO: Add MOSVR as Whole new Method 
# TODO: Add MO-RegTree   
# TODO: Add Full Parameters 

class SingleTargetMethod:
    '''Default is GradientBoostingRegressor'''
    def __init__(self, selector='gradientboost', custom_regressor = None): 
        super().__init__()  
        ESTIMATORS = { 
            'linear' : LinearRegression(), 
            'kneighbors' : KNeighborsRegressor(), 
            'adaboost' : AdaBoostRegressor(), 
            'gradientboost' : GradientBoostingRegressor(), 
            'mlp' : MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1),
            'svr' : SVR(), 
            'xgb' : xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.2,max_depth = 6, alpha = 10, n_estimators = 10)
        } 
        if (custom_regressor != None and self.implements_SciKitLearn_API(custom_regressor)): 
            self.MORegressor = MultiOutputRegressor(custom_regressor) 
            return
        elif( selector.lower() in ESTIMATORS):
            self.MORegressor = MultiOutputRegressor(ESTIMATORS[selector.lower()] ) 
            if custom_regressor != None: 
                raise Warning('\'{}\' is not valid regressor using \'{}\' instead'.format(custom_regressor,selector))
        else: 
            raise ValueError('\'{}\' is not a valid selector for SingleTargetMethod'.format(selector)) 
    
    def implements_SciKitLearn_API(self,object): 
        fit = getattr(object,'fit',None)
        predict = getattr(object,'predict',None) 
        if(fit != None and predict != None):
            return True 
        return False

    def fit(self,X_train, y_train): 
        self.MORegressor.fit(X_train,y_train)  
        return self.MORegressor

    def predict( self, X_test): 
        result = self.MORegressor.predict(X_test) 
        return result

# Wrapper around sklearns DecisionTreeRegressor https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
class MultiLayerPerceptron: 
    def __init__(self,hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08): 
        super().__init__() 
        self.mlp = MLPRegressor(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon)
    
    def fit(self, X_train, y_train): 
        self.mlp.fit(X_train,y_train)
        return self.mlp 
    
    def predict(self, X_test): 
        result = self.mlp.predict(X_test)
        return result  

# Wrapper around sklearns DecisionTreeRegressor https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
class MultiOutputRegressionTree: 
    def __init__(self, criterion="mse",splitter="best",max_depth=None, min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.,min_impurity_split=None,presort=False): 
        super().__init__() 
        self.mort = DecisionTreeRegressor(criterion,splitter,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,random_state,max_leaf_nodes,min_impurity_decrease,min_impurity_split,presort)

    def fit(self, X_train, y_train): 
        self.mort.fit(X_train,y_train)
        return self.mort 

    def predict(self, X_test): 
        return self.mort.predict(X_test)

class NeuronalNetRegressor: 
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,model,patience=5,learning_rate=0.01, print_after_epochs=10): 
        if not isinstance(model, nn.Module):
            raise ValueError('\'{}\' is not a valid instance of pytorch.nn'.format(model))
        
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.patience = patience 
        self.learning_rate = learning_rate  
        self.print_after_epochs = print_after_epochs

    def fit(self,X_train,y_train):
        n_samples,n_features ,n_targets = len(X_train), len(X_train[0]), len(y_train[0])

        #Create Validation Set from train Set
        X_train,X_validate,y_train,y_validate = train_test_split(X_train,y_train, test_size=0.1)      
        X_train_t,y_train_t = self.model.convert_train_set_to_tensor(X_train,y_train)
        X_validate_t,y_validate_t = self.model.convert_train_set_to_tensor(X_validate,y_validate)

        #Calculate on GPU if possible
        X_train_t, y_train_t = X_train_t.to(self.DEVICE), y_train_t.to(self.DEVICE)
        
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.model.train() 
        stopper = early.earlyStopping(self.patience) 
        stop = False
        epochs = 0
        
        while(stop == False):
            #train
            self.optimizer.zero_grad()
            y_pred_train = self.model(X_train_t)
            loss = self.loss_fn(y_pred_train, y_train_t)
            loss.backward() 
            self.optimizer.step() 
            #caculate validation loss an perform early stopping  
            y_pred_val = self.model(X_validate_t)
            validation_loss = self.loss_fn(y_pred_val,y_validate_t)
            stop = stopper.stop(validation_loss)
            
            if epochs % self.print_after_epochs == 0:
                validation_error = er.average__relative_root_mean_squared_error(y_validate,y_pred_val.detach().numpy()) 
                train_error = er.average__relative_root_mean_squared_error(y_train,y_pred_train.detach().numpy()) 
                print('Validation Error: {} \nTrain Error: {}'.format(validation_error,train_error))
            epochs += 1 
            
        final_train_error = er.average__relative_root_mean_squared_error(y_train,y_pred_train.detach().numpy())  
        final_validation_error = er.average__relative_root_mean_squared_error(y_validate,y_pred_val.detach().numpy()) 
        print("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(epochs,final_train_error,final_validation_error))  
        
        return self

    def predict(self, X_test):
        n_samples = len(X_test) 
        n_features = len(X_test[0])
        X_test_t = self.model.convert_test_set_to_tensor(X_test)     
                
        self.model.eval() 
        with torch.no_grad(): 
            y_pred_t = self.model(X_test_t) 
        
        return y_pred_t.detach().numpy()

@AbstractNeuronalNet.register
class NeuronalNet(nn.Module):
    def __init__(self, input_dim, output_dim,selector='max'): 
        MIDDLE_LAYER_NEURON_CALCULATION ={ 
            'mean': mean([input_dim,output_dim]), 
            'max' :  max([input_dim,output_dim]), 
            'doubleInput': input_dim * 2
        }
        super().__init__() 
        if selector not in MIDDLE_LAYER_NEURON_CALCULATION: 
            raise ValueError('Selector \'{}\' is not valid')

        middleLayerNeurons = int(MIDDLE_LAYER_NEURON_CALCULATION[selector])
        self.batchNorm = nn.BatchNorm1d(middleLayerNeurons)

        self.fc1 = nn.Linear(input_dim,middleLayerNeurons)
        self.fc2 = nn.Linear(middleLayerNeurons,middleLayerNeurons)
        self.fc3 = nn.Linear(middleLayerNeurons,output_dim)

    def forward(self, x):
        out = self.fc1(x)  
        out = self.batchNorm(out)  
        out = F.relu(out) 
        out = F.dropout(out,0.2) 
        out = self.fc2(out) 
        out = self.batchNorm(out)  
        out = F.relu(out) 
        out = F.dropout(out,0.2)
        out = self.fc3(out)

        return out 

    def convert_train_set_to_tensor(self,X_train,y_train):
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        return X_train_t,y_train_t 

    def convert_test_set_to_tensor(self,X_test): 
        X_test_t = torch.from_numpy(X_test).float()
        return X_test_t

@AbstractNeuronalNet.register
class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(64*15, 100)
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def convert_train_set_to_tensor(self,X_train,y_train): 
        X_train_t = torch.from_numpy(X_train).float().reshape(len(X_train),1,len(X_train[0]))
        y_train_t = torch.from_numpy(y_train).float() 
        return X_train_t,y_train_t

    def convert_test_set_to_tensor(self,X_test): 
        X_test_t = torch.from_numpy(X_test).float().reshape(len(X_test),1,len(X_test[0]))
        return X_test_t
