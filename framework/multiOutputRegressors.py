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
import numpy as np
from sklearn.model_selection import train_test_split 
import traceback 

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

    def __init__(self,model,patience=5,learning_rate=0.01, print_after_epochs=10, batch_size=None): 
        if not isinstance(model, nn.Module):
            raise ValueError('\'{}\' is not a valid instance of pytorch.nn'.format(model))
        
        self.model = model
        self.loss_fn = er.tensor_average__relative_root_mean_squared_error #nn.MSELoss()
        self.patience = patience 
        self.learning_rate = learning_rate  
        self.print_after_epochs = print_after_epochs 
        self.batch_size = batch_size

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

        X_train_t_batched,y_train_t_batched = self.split_train_to_batches(X_train_t,y_train_t,self.batch_size)
        
        while(stop == False):
            #train
            for batch_X, batch_y in zip(X_train_t_batched,y_train_t_batched):
                self.optimizer.zero_grad()
                y_pred_train = self.model(batch_X)
                loss = self.loss_fn(y_pred_train, batch_y)
                loss.backward() 
                self.optimizer.step() 
            #caculate validation loss an perform early stopping  
            y_pred_val = self.model(X_validate_t)
            validation_loss = self.loss_fn(y_pred_val,y_validate_t)
            stop = stopper.stop(validation_loss)
            
            if epochs % self.print_after_epochs == 0:
                y_pred_train = self.model(X_train_t)
                validation_error = er.tensor_average__relative_root_mean_squared_error(y_pred_val,y_validate_t) 
                train_error = er.tensor_average__relative_root_mean_squared_error(y_pred_train,y_train_t) 
                print('Validation Error: {} \nTrain Error: {}'.format(validation_error,train_error))
            epochs += 1 
            
        y_pred_train = self.model(X_train_t) 
        final_train_error = er.tensor_average__relative_root_mean_squared_error(y_pred_train,y_train_t)  
        final_validation_error = er.tensor_average__relative_root_mean_squared_error(y_pred_val,y_validate_t) 
        print("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(epochs,final_train_error,final_validation_error))  
        
        return self

    def split_train_to_batches(self,X_train_t,y_train_t,batch_size): 
        if batch_size == None: 
            return torch.split(X_train_t, len(X_train_t)), torch.split(y_train_t, len(X_train_t))  
        else: 
            return torch.split(X_train_t, batch_size),torch.split(y_train_t, batch_size)

    def predict(self, X_test):
        n_samples = len(X_test) 
        n_features = len(X_test[0])
        X_test_t = self.model.convert_test_set_to_tensor(X_test)     
                
        self.model.eval() 
        with torch.no_grad(): 
            y_pred_t = self.model(X_test_t) 
        
        return y_pred_t.detach().numpy() 
    
    def save(self, store_path): 
        try:
            torch.save(self.model, store_path + '.ckpt') 
        except Exception: 
            print(traceback.format_exc())
    
    def load(self, load_path): 
        try: 
            self.model = torch.load(load_path)   
            return self 
        except Exception: 
            print(traceback.format_exc())

@AbstractNeuronalNet.register
class NeuronalNet(nn.Module):
    def __init__(self, input_dim, output_dim,selector='max',p_dropout_1=0.5,p_dropout_2=0.5): 
        MIDDLE_LAYER_NEURON_CALCULATION ={ 
            'mean': mean([input_dim,output_dim]), 
            'max' :  max([input_dim,output_dim]), 
            'doubleInput': input_dim * 2
        }
        super().__init__() 
        if selector not in MIDDLE_LAYER_NEURON_CALCULATION: 
            raise ValueError('Selector \'{}\' is not valid') 
        self.dropout_1 = p_dropout_1 
        self.dropout_2 = p_dropout_2

        middleLayerNeurons = int(MIDDLE_LAYER_NEURON_CALCULATION[selector])
        self.batchNorm = nn.BatchNorm1d(middleLayerNeurons)

        self.fc1 = nn.Linear(input_dim,middleLayerNeurons)
        self.fc2 = nn.Linear(middleLayerNeurons,middleLayerNeurons)
        self.fc3 = nn.Linear(middleLayerNeurons,output_dim)

    def forward(self, x):
        out = self.fc1(x)  
        out = self.batchNorm(out)  
        out = F.relu(out) 
        out = F.dropout(out,self.dropout_1) 
        out = self.fc2(out) 
        out = self.batchNorm(out)  
        out = F.relu(out) 
        out = F.dropout(out,self.dropout_2)
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
    def __init__(self, input_dim, output_dim,p_dropout=0.5):
        super().__init__()  
        self.p_dropout = p_dropout
        width_out = self.get_output_size(input_dim)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2))
        self.drop_out = nn.Dropout(self.p_dropout)
        self.fc1 = nn.Linear(64*width_out, 100)
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

    def get_output_size(self,input_dim): 
        k_c_1 = 2
        k_p_1 = 3
        k_c_2 = 2
        k_p_2 = 3
        s_c_1 = 2
        s_c_2 = 2
        s_p_1 = 2
        s_p_2 = 2  
        wk1 = self.get_size(input_dim,k_c_1,0,s_c_1) 
        wp1 = self.get_size(wk1,k_p_1,0,s_p_1) 
        wk2 = self.get_size(wp1,k_c_2,0,s_c_2) 
        wp2 = self.get_size(wk2,k_p_2,0,s_p_2) 
        return int(wp2)

    def get_size(self,w,k,p=0,s=0):
        return ((w-k+2*p)/s) +1




class MLSSVR: 
    def __init__(self, kernel_param1, kernel_param2, kernel_selector = "linear"): 
        self.kernel_selector = kernel_selector 
        self.kernel_param1 = kernel_param1 
        self.kernel_param2 = kernel_param2

    def kernel_function(self, kernel_selector, X, Z, param1, param2): 
        
        if(len(X[0,:]) != len(Z[0,:])): 
            raise ValueError('2nd Dimension of X and Z must match') 
        
        if kernel_selector.lower() == 'linear':
            K = X @ Z.T
        elif kernel_selector.lower() == 'poly':
            K = (X @ Z.T + param1)**param2

        #elif kernel_selector.lower() == 'rbf': 
        #    K = rbf_kernel(X,Z,gamma=param1)
        #
        #elif kernel_selector.lower() is 'erbf':
        elif kernel_selector.lower() == 'sigmoid': 
            K = np.tanh(param1 * X @ Z.T / len(X[0]) + param2)
        else:  
            raise ValueError('Unknown kernel method: {}'.format(kernel_selector))
        
        return K  

    def fit(self, X_train, y_train,gamma,lambd): 
        self.gamma = gamma 
        self.lambd = lambd 
        if(len(X_train) != len(y_train)): 
            raise ValueError('Sample size of inputs does not match') 

        l = y_train.shape[0]
        m = y_train.shape[1] 
        K = self.kernel_function(self.kernel_selector,X_train,X_train,self.kernel_param1,self.kernel_param2) 
        H = np.tile(K,(m,m)) + np.eye(m*l) / gamma  

        P = np.zeros((m*l,m)) 
        for t in range(m): #Verify complete section
            idx1 = l * (t)
            idx2 = l * (t+1)
            H[idx1: idx2, idx1: idx2] = H[idx1: idx2, idx1: idx2] + K*(m/lambd)
            P[idx1:idx2, t] = np.ones(l) 
        
        eta = np.linalg.lstsq(H,P) 
        nu = np.linalg.lstsq(H,y_train.flatten()) #CHECK 
        S = P.T @ eta[0]
        b =  np.linalg.inv(S) @ eta[0].T @ y_train.flatten()
        alpha = nu[0] - eta[0] @ b 

        alpha = alpha.reshape(l,m)
        self.alpha = alpha 
        self.b = b
        return self 

    def predict(self,X_test,X_train,y_test): 
        if(len(y_test[0,:]) != np.size(self.b)): 
            raise ValueError('Nnumber of columns in y_test and b must be equal.') 
        m,l = len(y_test[0,:]), len(X_train)

        if(len(self.alpha) != l or len(self.alpha[0]) != m): 
            raise ValueError('The size of slpha should be {} * {}'.format(l,m)) 
        
        N_test = len(X_test) 
        b = self.b.flatten() 

        K = self.kernel_function(self.kernel_selector,X_test,X_train,self.kernel_param1,self.kernel_param2); 
        t0 = np.sum(K @ self.alpha, axis=1)
        t1 =  np.tile(t0,(m,1)).T 
        t2 =  K @ self.alpha * (m/self.lambd)
        t3 =  np.tile(b.T, (N_test,1)) 
        y_pred = t1 + t2 + t3
        return y_pred