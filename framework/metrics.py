from numpy import mean, sqrt
import torch as t


def average_relative_error(y_test, y_pred):
    """Calculate Average Relative Error 

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error 
    """
    return sum(sum(abs(y_test - y_pred) / y_test) / len(y_test)) / len(y_test[0, :])


def average_relative_root_mean_squared_error(y_test, y_pred):
    """Calculate Average Relative Root Mean Squared Error 

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error 
    """
    return sum(sqrt(sum((y_test - y_pred)**2) / sum((y_test - mean(y_test, axis=0))**2))) / len(y_pred[0, :])


def tensor_average_relative_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Relative Root Mean Squared Error 

    Args:
        y_test (torch.FloatTensor): array of dimension N x d with actual values
        y_pred (torch.FloatTensor): array of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error 
    """
    return t.sum(t.sqrt(t.sum((y_test - y_pred)**2, dim=0) / t.sum(((y_test - t.mean(y_pred, dim=0))**2), dim=0))) / len(y_pred[0, :]).item()
