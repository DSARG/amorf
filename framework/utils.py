class EarlyStopping():
    """
    Early Stopping Mechanism

    Early stopping ends training when training or validation error continously rise over a given interval of steps (patience) 

    Args: 
        patience (int) : Stop after how many continous incrementations 

    Attributes: 
        lastError (float) : Error-value of previous call of the 'stop'-function
        patience (int) : Stop after how many continous incrementations
        suceedingsHigherValues (int) :Number of continous incrementations of error
    """

    def __init__(self, patience):
        self.lastError = 0
        self.patience = patience
        self.succeedingHigherValues = 0
    # returns True if training should stop and False if it should continue

    def stop(self, newError):
        """
        Decides whether training should be stopped

        Args:
            newError (float): Training or validation Error

        Returns:
            bool :  Return true if number of values higher than the previous one equals patience
        """
        if(newError > self.lastError):
            self.succeedingHigherValues += 1
        else:
            self.succeedingHigherValues = 0

        self.lastError = newError
        if(self.patience <= self.succeedingHigherValues):
            return True
        else:
            return False
