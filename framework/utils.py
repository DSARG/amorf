import torch


class EarlyStopping:
    """
    Early Stopping Mechanism

    Returns True if training should stop and False if it should continue. 
    Saves the best model. 
    Only works for decreasing error values.

    Args:
        patience (int) : Stop after p continous incrementations.

    Attributes:
        lastError (float) : Error-value of previous call of the 'stop'-function
        patience (int) : Stop after how many continous incrementations
        suceedingsHigherValues (int) :Number of continous incrementations of error
    """

    def __init__(self, patience):
        # FIXME: setfirst error to None ->
        # TODO: Keep the best model and return after stop
        # TODO: keep the best model with the lowest validation error
        self.lastError = 0
        self.patience = patience
        self.succeedingHigherValues = 0
        self.best_model = None

    def stop(self, newError, model):
        """
        Decides whether training should be stopped 

        Decides whether training should be stopped. Every time the erorr decreases the model is saved.

        Args:
            newError (float): Training or validation Error.

        Returns:
            bool :  Return true if number of values higher than the previous one equals patience.
        """
        if self.best_model is None:
            self.best_model = model
        if(newError > self.lastError):
            self.succeedingHigherValues += 1
        else:
            self.succeedingHigherValues = 0
            self.__save_model(model)

        self.lastError = newError
        if(self.patience <= self.succeedingHigherValues):
            if self.patience > 1:
                self.best_model = torch.load("checkpoint.ckpt")
            return True
        else:
            return False

    def __save_model(self, model):
        torch.save(model, "checkpoint.ckpt")


def printMessage(Message, verbosity):
    """Prints messages if verbosity is set 

    Does not do much currently but can be expanded later.
    """
    if(verbosity == 1):
        print(Message)
