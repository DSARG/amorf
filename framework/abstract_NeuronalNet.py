from abc import ABC, abstractmethod 

class AbstractNeuronalNet(ABC): 

    @abstractmethod
    def convert_train_set_to_tensor(self,X_train,y_train): 
        pass  
    
    @abstractmethod
    def convert_test_set_to_tensor(self,X_test): 
        pass  