import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from amorf.metrics import average_relative_root_mean_squared_error


class MLSSVR:
    """Multi-Output Least Squares Support Vector Regression

    MLSSVR implementation based on this matlab implementation:https://github.com/pzczxs/MLSSVR

    Original Paper:
    Shuo Xu, Xin An, Xiaodong Qiao, Lijun Zhu, and Lin Li, 2013.
    Multi-Output Least-Squares Support Vector Regression Machines.
    Pattern Recognition Letters, Vol. 34, No. 9, pp. 1078-1084. 

    Raises: 
        ValueError : If Kernel-selector is invalid 
        ValueError : If dimensions do not match

    Args:
        kernel_param1 (float): first Parameter for kernel function
        kernel_param1 (float):  seconde Parameter for kernel function
        kernel_selector (string,optional): One of 'linear','poly','rbf','erbf','sigmoid'. Default:'linear'

    """

    def __init__(self, kernel_param1, kernel_param2, kernel_selector="linear"):
        self.kernel_selector = kernel_selector
        self.kernel_param1 = kernel_param1
        self.kernel_param2 = kernel_param2
        selectors = ['linear', 'poly', 'rbf', 'erbf', 'sigmoid']
        if self.kernel_selector not in selectors:
            raise ValueError(
                'Unknown kernel method: {}'.format(kernel_selector))

    def __kernel_function(self, kernel_selector, X, Z, param1, param2):

        if(len(X[0, :]) != len(Z[0, :])):
            raise ValueError('2nd Dimension of X and Z must match')

        if kernel_selector.lower() == 'linear':
            K = X @ Z.T
        elif kernel_selector.lower() == 'poly':
            K = (X @ Z.T + param1)**param2

        elif kernel_selector.lower() == 'rbf':

            K = rbf_kernel(X, Z, gamma=param1)
        # TODO: Add ERBF
        # elif kernel_selector.lower() is 'erbf':
        elif kernel_selector.lower() == 'sigmoid':
            K = np.tanh(param1 * X @ Z.T / len(X[0]) + param2)

        else:
            raise NotImplementedError

        return K

    def fit(self, X_train, y_train, gamma, lambd):
        """Fits the model to the training data set

        Args:
            X_train (nd.array): Set of descriptive Variables
            y_train (nd.array): Set of target Variables
            gamma (float): gamma parameter
            lambd (float): lambda parameter

        Raises:
            ValueError: if length of X_train and y_train don't match

        Returns:
            MLSSVR: fitted MLSSVR
        """
        self.gamma = gamma
        self.lambd = lambd
        if(len(X_train) != len(y_train)):
            raise ValueError('Sample size of inputs does not match')

        l = y_train.shape[0]
        m = y_train.shape[1]

        self.n_targets = m
        self.n_samples = l

        K = self.__kernel_function(
            self.kernel_selector, X_train, X_train, self.kernel_param1, self.kernel_param2)
        H = np.tile(K, (m, m)) + np.eye(m * l) / gamma

        P = np.zeros((m * l, m))
        for t in range(m):  # Verify complete section
            idx1 = l * (t)
            idx2 = l * (t + 1)
            H[idx1: idx2, idx1: idx2] = H[idx1: idx2,
                                          idx1: idx2] + K * (m / lambd)
            P[idx1:idx2, t] = np.ones(l)

        eta = np.linalg.lstsq(H, P, rcond=None)
        nu = np.linalg.lstsq(H, y_train.flatten(), rcond=None)  # CHECK
        S = P.T @ eta[0]
        b = np.linalg.inv(S) @ eta[0].T @ y_train.flatten()
        alpha = nu[0] - eta[0] @ b

        alpha = alpha.reshape(l, m)
        self.alpha = alpha
        self.b = b
        return self

    # FIXME: Remove X_train
    def predict(self, X_test, X_train):
        """Predicts the target variables for the given test set

        Args:
            X_test (np.ndarray): Test set withdescriptive variables
            X_train (np.ndarray): Train set with descriptive variables

        Returns:
            np.ndarray: Predicted target variables
        """
        m, l = self.n_targets, self.n_samples

        N_test = len(X_test)
        b = self.b.flatten()

        # Why training set here?
        K = self.__kernel_function(
            self.kernel_selector, X_test, X_train, self.kernel_param1, self.kernel_param2)
        t0 = np.sum(K @ self.alpha, axis=1)
        t1 = np.tile(t0, (m, 1)).T
        t2 = K @ self.alpha * (m / self.lambd)
        t3 = np.tile(b.T, (N_test, 1))
        y_pred = t1 + t2 + t3
        return y_pred

    def score(self, X_test, X_train, y_test):
        """Returns Average Relative Root Mean Squared Error for given test data and targets

        Args:
            X_test (np.ndarray): Test samples
            y_test (np.ndarray): True targets
        """
        return average_relative_root_mean_squared_error(self.predict(X_test, X_train), y_test)
