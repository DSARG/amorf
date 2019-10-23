from numpy import mean, sqrt
import torch as t


def average_relative_error(y_test, y_pred):
    """[summary] 

    Args:
        y_test ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    return sum(sum(abs(y_test - y_pred) / y_test) / len(y_test)) / len(y_test[0, :])


def average_relative_root_mean_squared_error(y_test, y_pred):
    """[summary]
    
    Args:
        y_test ([type]): [description]
        y_pred ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    return sum(sqrt(sum((y_test - y_pred)**2) / sum((y_test - mean(y_test, axis=0))**2))) / len(y_pred[0, :])


def tensor_average_relative_root_mean_squared_error(y_pred, y_test):
    """[summary]
    
    Args:
        y_pred ([type]): [description]
        y_test ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    return t.sum(t.sqrt(t.sum((y_test - y_pred)**2, dim=0) / t.sum(((y_test - t.mean(y_pred, dim=0))**2), dim=0))) / len(y_pred[0, :])
