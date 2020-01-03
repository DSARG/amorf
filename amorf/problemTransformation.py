from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# AutoEncoderRegression
from amorf.utils import EarlyStopping, printMessage
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from amorf.metrics import average_relative_root_mean_squared_error


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

#FIXME: Wrong Output (100..0..100)
class AutoEncoderRegression:
    """Regressor that uses an Autoencoder to reduce dimensionality of target variables 

    Raises:
        Warning: If Custom Regressor is not valid, default estimator will be
                used instead
        ValueError: If selector is not a valid value

    Args:
        regressor (string,optional): Can be one of the following linear', 'kneighbors',
                            'adaboost', 'gradientboost', 'mlp', 'svr', 'xgb'. Default: 'gradientboost'
        custom_regressor (object,optional): Custom Estimator that must implement 'fit()'
                            and 'predict()' function. Default: None
        batch_size (int,optional): Otherwise training set is split into batches of given size. Default: None
        shuffle (bool,optional) Set to True to have the data reshuffled at every epoch. Default: False
        learning_rate (float,optional): Learning rate for optimizer. Default: 1e-3
        use_gpu (bool,optional): Flag that allows usage of cuda cores for calculations. Default: False
        patience (int,optional): Stop training after p continous incrementations. Default: None
        training_limit (int,optional): After specified number of epochs training will be terminated, regardless of EarlyStopping stopping. Default: 100
        verbosity (int,optional): 0 to only print errors, 1 (default) to print status information. Default: 1
        print_after_epochs (int,optional): Specifies after how many epochs training and validation error will be printed to command line. Default: 500
    """

    def __init__(self, regressor='gradientboost', custom_regressor=None, batch_size=None, shuffle=False, learning_rate=1e-3, use_gpu=False, patience=None, training_limit=100, verbosity=1, print_after_epochs=500):
        self.learning_rate = learning_rate
        self.path = ".autoncoder_bestmodel_validation"
        self.print_after_epochs = print_after_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training_limit = training_limit
        self.verbosity = verbosity
        self.Device = 'cpu'
        if use_gpu is True and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.Device = "cuda:0"

        if training_limit is None and patience is None:
            raise ValueError('Either training_limit or patience must be set')

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
        y_train_t = torch.tensor(y_train, dtype=torch.float).to(self.Device)
        y_validate_t = torch.tensor(y_val, dtype=torch.float).to(self.Device)

        model = autoencoder(n_targets).to(self.Device)
        best_model, best_score = None, np.inf
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        val_losses = []

        if self.patience is not None:
            stopper = EarlyStopping(self.patience)
        stop = False
        epochs = 0
        self.batch_size = len(
            y_train_t) if self.batch_size is None else self.batch_size
        train_dataloader = DataLoader(TensorDataset(
            y_train_t), batch_size=self.batch_size, shuffle=self.shuffle)

        while(stop is False):
            model.train()
            for batch in train_dataloader:
                batch_y = batch[0]
                # ===================forward=====================
                output = model(batch_y)
                loss = criterion(output, batch_y)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================validate========================
            model.eval()
            y_pred_validate = model(y_validate_t)
            validation_loss = criterion(y_pred_validate, y_validate_t)
            if validation_loss < best_score:
                best_score = validation_loss
                torch.save(model.state_dict(), self.path)
            if self.patience is not None:
                stop = stopper.stop(validation_loss, model)
            if stop is True and self.patience > 1:
                model.load_state_dict(stopper.best_model['state_dict'])
            # ===================log========================
            if epochs % self.print_after_epochs == 0:
                printMessage('Epoch {}\nValidation Error: {}\nTrain Error:{}'.format(
                    epochs, loss, validation_loss), self.verbosity)
            epochs += 1
            if self.training_limit is not None and self.training_limit <= epochs:
                stop = True

        y_pred_train = model(y_train_t)
        final_train_error = criterion(y_pred_train, y_train_t)
        final_validation_error = criterion(y_pred_validate, y_validate_t)
        printMessage("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(
            epochs, final_train_error, final_validation_error), self.verbosity)

        self.best_model = autoencoder(n_targets)
        self.best_model.load_state_dict(torch.load(self.path))
        self.best_model.to(self.Device)
        y_enc_train = self.best_model.encoder(y_train_t)

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

    def score(self, X_test, y_test):
        """Returns Average Relative Root Mean Squared Error for given test data and targets

        Args:
            X_test (np.ndarray): Test samples
            y_test (np.ndarray): True targets
        """
        return average_relative_root_mean_squared_error(self.predict(X_test), y_test)


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
