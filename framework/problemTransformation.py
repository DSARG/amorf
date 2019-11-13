from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# AutoEncoderRegression
from framework.utils import EarlyStopping as early 
from framework.utils import printMessage
import torch
from torch import nn
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split


class SingleTargetMethod:
    """ Performs regression for each target variable separately.

        This method is a wrapper around scikit learns MultiOutputRegressor
        class. It has some estimators readily provided and allows for
        custom estimators to be used.

    Args:
        selector (string): Can be one of the following linear', 'kneighbors',
                            'adaboost', 'gradientboost', 'mlp', 'svr', 'xgb'
        custom_regressor (object): Custom Estimator that must implement 'fit()'
                            and 'predict()' function.

    Raises:
        Warning: If Custom Regressor is not valid, default estimator will be
                used instead
        ValueError: If selector is not a valid value
    """

    def __init__(self, selector='gradientboost', custom_regressor=None):
        super().__init__()

        ESTIMATORS = {
            'linear': LinearRegression(),
            'kneighbors': KNeighborsRegressor(),
            'adaboost': AdaBoostRegressor(),
            'gradientboost': GradientBoostingRegressor(),
            'mlp': MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, ), max_iter=1000, random_state=1),
            'svr': SVR(gamma='auto'),
            'xgb': xgb.XGBRegressor(verbosity=0, objective='reg:squarederror', colsample_bytree=1, learning_rate=0.2, max_depth=6, alpha=10, n_estimators=10)
        }
        if custom_regressor is not None and _implements_SciKitLearn_API(custom_regressor):
            try:
                self.MORegressor = MultiOutputRegressor(custom_regressor)
            finally:
                pass
            return
        elif isinstance(selector, str) and selector.lower() in ESTIMATORS:
            self.MORegressor = MultiOutputRegressor(
                ESTIMATORS[selector.lower()])
            if custom_regressor is not None:
                raise Warning('\'{}\' is not valid regressor using \'{}\' instead'.format(
                    custom_regressor, selector))
        else:
            raise ValueError(
                '\'{}\' is not a valid selector for SingleTargetMethod'.format(selector))

    def fit(self, X_train, y_train):
        """Fits the estimator to the training data

        Args:
            X_train (np.ndarray): Training set descriptive variables
            y_train (np.ndarray): Training set target variables

        Returns:
            [sklearn.MultiOutputRegressor]: Trained estimator
        """
        self.MORegressor.fit(X_train, y_train)
        return self.MORegressor

    def predict(self, X_test):
        """Predicts the target variables for a given set of descriptive variables

        Args:
            X_test (np.ndarray): Array with descriptive variables

        Returns:
            np.ndarray: Array with predicted target variables
        """
        result = self.MORegressor.predict(X_test)
        return result


class AutoEncoderRegression:
    """Regressor that uses an Autoencoder to reduce dimensionality of target variables 

    Raises:
        Warning: If Custom Regressor is not valid, default estimator will be
                used instead
        ValueError: If selector is not a valid value

    Args:
        regressor (string): Can be one of the following linear', 'kneighbors',
                            'adaboost', 'gradientboost', 'mlp', 'svr', 'xgb'
        custom_regressor (object): Custom Estimator that must implement 'fit()'
                            and 'predict()' function.
        batch_size (int): Default None - otherwise training set is split into batches of given size
        learning_rate (float): learning rate for optimizer
        use_gpu (bool): Flag that allows usage of cuda cores for calculations
        patience (int): Stop training after p continous incrementations
        training_limit (int): Default None - After specified number of epochs training will be terminated, regardless of early stopping
        verbosity (int): 0 to only print errors, 1 (default) to print status information
        print_after_epochs (int): Specifies after how many epochs training and validation error will be printed to command line
    """
    # TODO: Add Data Loaders
    # FIXME: Naming Inconsistencies (y_train, y_data) -> find scheme to apply everywhere

    def __init__(self, regressor='gradientboost', custom_regressor=None, batch_size=None, learning_rate=1e-3, use_gpu=False, patience=5,  training_limit=None, verbosity=1, print_after_epochs=500):
        self.learning_rate = learning_rate
        self.path = ".autoncoder_bestmodel_validation"
        self.print_after_epochs = print_after_epochs
        self.patience = patience
        self.batch_size = batch_size 
        self.training_limit = training_limit
        self.verbosity = verbosity
        self.Device = 'cpu'
        if use_gpu is True and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.Device = "cuda:0"

        ESTIMATORS = {
            'linear': LinearRegression(),
            'kneighbors': KNeighborsRegressor(),
            'adaboost': AdaBoostRegressor(),
            'gradientboost': GradientBoostingRegressor(),
            'mlp': MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), max_iter=1000, random_state=1),
            'svr': SVR(gamma='auto'),
            'xgb': xgb.XGBRegressor(verbosity=0, objective='reg:squarederror', colsample_bytree=1, learning_rate=0.2, max_depth=6, alpha=10, n_estimators=10)
        }
        if custom_regressor is not None and _implements_SciKitLearn_API(custom_regressor):
            try:
                self.regressor = custom_regressor
            finally:
                pass
            return
        elif isinstance(regressor, str) and regressor.lower() in ESTIMATORS:
            self.regressor = ESTIMATORS[regressor.lower()]
            if custom_regressor is not None:
                raise Warning('\'{}\' is not valid regressor using \'{}\' instead'.format(
                    custom_regressor, regressor))
        else:
            raise ValueError(
                '\'{}\' is not a valid selector for AutoEncoderRegression'.format(regressor))

    def fit(self, X_train, y_train):
        """Fits the model to the training data set 

        Trains an AutoEncoder to encode multidimensional target variables into scalar. 
        The resulting data set is used to train the given regressor to predict these scalars.

        Args:
            X_train (nd.array): Set of descriptive Variables
            y_train (nd.array): Set of target Variables

        Returns:
            AutoEncoderRegressor: fitted AutoEncoderRegressor
        """
        n_targets = len(y_train[0])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1)
        y_data = torch.tensor(y_train, dtype=torch.float).to(self.Device)
        y_data_val = torch.tensor(y_val, dtype=torch.float).to(self.Device)

        model = autoencoder(n_targets).to(self.Device)
        best_model, best_score = None, np.inf
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        val_losses = []

        stopper = early(self.patience)
        stop = False
        epoch = 0

        y_data_batched = self.__split_training_set_to_batches(
            y_data, self.batch_size)

        while(stop is False):
            model.train()
            for batch_y in y_data_batched:
                # ===================forward=====================
                output = model(batch_y)
                loss = criterion(output, batch_y)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================validate========================
            model.eval()
            val_pred = model(y_data_val)
            v_loss = criterion(val_pred, y_data_val)
            val_losses.append(v_loss.cpu().detach().numpy())
            if v_loss < best_score:
                best_score = v_loss
                torch.save(model.state_dict(), self.path)
            stop = stopper.stop(v_loss)
            # ===================log========================
            if epoch % self.print_after_epochs == 0:
                printMessage('epoch [{}], train_loss:{} \n \t\t validation_loss:{}'.format(
                    epoch + 1, loss, v_loss),self.verbosity)
            epoch += 1 
            if self.training_limit is not None and self.training_limit >= epoch:
                stop = True

        self.best_model = autoencoder(n_targets)
        self.best_model.load_state_dict(torch.load(self.path))
        self.best_model.to(self.Device)
        y_enc_train = self.best_model.encoder(y_data)

        if self.Device is 'cpu':
            self.regressor.fit(X_train, y_enc_train.detach().numpy().ravel())
        else:
            self.regressor.fit(
                X_train, y_enc_train.cpu().detach().numpy().ravel())
        return self

    def predict(self, X_test):
        """Predicts the encoded target variables and decodes them for the given test set

        Args:
            X_test (np.ndarray): Test set with descriptive variables

        Returns:
            np.ndarray: Predicted target variables
        """
        y_pred_test = self.regressor.predict(X_test)
        y_pred_test_t = torch.tensor(
            y_pred_test, dtype=torch.float).unsqueeze(1).to(self.Device)
        y_pred_dec = self.best_model.decoder(y_pred_test_t)

        return y_pred_dec.detach().numpy() if self.Device is 'cpu' else y_pred_dec.cpu().detach().numpy()

    def __split_training_set_to_batches(self, y_train_t, batch_size):
        if batch_size is None:
            return torch.split(y_train_t, len(y_train_t))
        else:
            return torch.split(y_train_t, batch_size)


class autoencoder(nn.Module):
    def __init__(self, n_targets):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_targets, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.Linear(12, 1))
        self.decoder = nn.Sequential(
            nn.Linear(1, 12),
            # nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_targets))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def _implements_SciKitLearn_API(object):
    fit = getattr(object, 'fit', None)
    predict = getattr(object, 'predict', None)
    if(fit is not None and predict is not None):
        return True
    return False
