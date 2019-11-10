from numpy import mean, sqrt
import numpy as np
import torch as t 

# TODO: Hide "tensor_" methods since they are only for internal use


def average_relative_error(y_test, y_pred):
    """Calculate Average Relative Error

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error
    """
    result = sum(sum(abs(y_test - y_pred) / y_test) /
                 len(y_test)) / len(y_test[0, :])
    return result 

def tensor_average_relative_error(y_test, y_pred):
    """Calculate Average Relative Error

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error
    """
    result = t.sum(t.sum(t.abs(y_test - y_pred) / y_test, dim=0) /
                 len(y_test)) / len(y_test[0, :])
    return result


def average_relative_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Relative Root Mean Squared Error (aRRMSE)

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error
    """
    result = sum(sqrt(sum((y_test - y_pred)**2) /
                      sum((y_test - mean(y_test, axis=0))**2))) / len(y_pred[0, :])
    return result


def tensor_average_relative_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Relative Root Mean Squared Error (aRRMSE)

    Args:
        y_test (torch.FloatTensor): array of dimension N x d with actual values
        y_pred (torch.FloatTensor): array of dimension N x d with predicted values

    Returns:
        torch.FloatTensor : Average Relative Root Mean Squared Error
    """
    result = t.sum(t.sqrt(t.sum((y_test - y_pred)**2, dim=0) /
                          t.sum(((y_test - t.mean(y_test, dim=0))**2), dim=0))) / len(y_pred[0, :])
    return result


def tensor_mean_squared_error(y_pred, y_test):
    """Calculate Mean Squared Error (MSE)

    Args:
        y_test (torch.FloatTensor): array of dimension N x d with actual values
        y_pred (torch.FloatTensor): array of dimension N x d with predicted values

    Returns:
        torch.FloatTensor: Mean Squared Error
    """
    result = t.sum(t.sum((y_test - y_pred)**2) /
                   len(y_test))
    return result


def mean_squared_error(y_pred, y_test):
    """Calculate Mean Squared Error (MSE)

     Args:
         y_test (np.array): array of dimension N x d with actual values
         y_pred (np.ar ray): array of dimension N x d with predicted values

     Returns:
         float : Mean Squared Error
     """
    result = sum((sum((y_test - y_pred)**2) /
                  len(y_test)))
    return result


def tensor_average_root_mean_square_error(y_pred, y_test):
    """Calculate Average Root Mean Squared Error (aRMSE)

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.ar ray): array of dimension N x d with predicted values

    Returns:
        torch.FloatTensor: : Average Root Mean Squared Error
    """
    a = t.sum((y_test - y_pred)**2,dim=0)/ len(y_test)
    b = t.sqrt(a)
    result = t.sum(t.sqrt(t.sum((y_test - y_pred)**2,dim=0) /
                          len(y_test) ))/len(y_test[0])
    return result


def average_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Root Mean Squared Error (aRMSE)

     Args:
         y_test (np.array): array of dimension N x d with actual values
         y_pred (np.ar ray): array of dimension N x d with predicted values

    Returns:
        float : Average Root Mean Squared Error
     """
    result = sum(sqrt((sum((y_test - y_pred)**2)
                       / len(y_test)) ))/len(y_test[0, :])
    return result

