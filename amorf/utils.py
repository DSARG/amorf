import torch


class EarlyStopping:
    """
    Early Stopping Mechanism

    Returns True if training should stop and False if it should continue. 
    Saves the best model. 
    Only works for decreasing loss values.

    Args:
        patience (int) : Stop after p continous incrementations.

    Attributes:
        lastLoss (float) : Loss-value of previous call of the 'stop'-function
        patience (int) : Stop after how many continous incrementations
        suceedingsHigherValues (int) :Number of continous incrementations of loss
    """

    def __init__(self, patience):
        self.lastLoss = 0
        self.patience = patience
        self.succeedingHigherValues = 0
        self.best_model = None

    def stop(self, newLoss, model):
        """
        Decides whether training should be stopped 

        Decides whether training should be stopped. Every time the erorr decreases the model is saved.

        Args:
            newLoss (float): Training or validation Loss.

        Returns:
            bool :  Return true if number of values higher than the previous one equals patience.
        """
        if self.best_model is None:
            self.best_model = model
        if(newLoss > self.lastLoss):
            self.succeedingHigherValues += 1
        else:
            self.succeedingHigherValues = 0
            self.__save_model(model)

        self.lastLoss = newLoss
        if(self.patience <= self.succeedingHigherValues):
            if self.patience > 1:
                self.best_model = torch.load("checkpoint.pth.tar")
            return True
        else:
            return False

    def __save_model(self, model):
        torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')


def printMessage(Message, verbosity):
    """Prints messages if verbosity is set 

    Does not do much currently but can be expanded later.
    """
    if(verbosity == 1):
        print(Message)
