from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression  

from sklearn.neural_network import MLPRegressor

#TODO: Add Support vector regression 


class SingleTargetMethod:
    '''Default is GradientBoostingRegressor'''
    def __init__(self, selector='gradientboost'): 
        super().__init__() 
        if(selector == 'linear' ): 
            regressor = LinearRegression()
        elif(selector == 'kneighbors'):  
            regressor = KNeighborsRegressor()
        elif selector == 'adaboost': 
            regressor = AdaBoostRegressor()
        elif selector == 'gradientboost': 
            regressor = GradientBoostingRegressor()
        elif selector == 'mlp': 
            regressor = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)   
        
        self.MORegessor =  MultiOutputRegressor(regressor)
    
    def fit(self,X_train, y_train): 
        self.MORegessor.fit(X_train,y_train)  
        return self.MORegessor

    def predict( self, X_test): 
        result = self.MORegessor.predict(X_test) 
        return result

class MultiLayerPerceptron: 
    def __init__(self): 
        super().__init__() 
        self.mlp = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(8,), random_state=1)
    
    def fit(self, X_train, y_train): 
        self.mlp.fit(X_train,y_train)
        return self.mlp 
    
    def predict(self, X_test): 
        result = self.mlp.predict(X_test)
        return result
