# patience = number of error values higher than the last one until interruption
class earlyStopping(): 
    def __init__(self,patience): 
        self.lastError = 0 
        self.patience = patience 
        self.succeedingHigherValues = 0; 
    #returns True if training should stop and False if it should continue
    def stop(self,newError): 
        if(newError >= self.lastError): 
            self.succeedingHigherValues+= 1 
        else: 
            self.succeedingHigherValues = 0 

        self.lastError = newError
        if(self.patience == self.succeedingHigherValues):
            return True 
        else: 
            return False 

