from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# AutoEncoderRegression
import torch
import torchvision
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
        if (custom_regressor is not None and self.__implements_SciKitLearn_API(custom_regressor)):
            try:
                self.MORegressor = MultiOutputRegressor(custom_regressor)
            finally:
                pass
            return
        elif(selector.lower() in ESTIMATORS):
            self.MORegressor = MultiOutputRegressor(
                ESTIMATORS[selector.lower()])
            if custom_regressor is not None:
                raise Warning('\'{}\' is not valid regressor using \'{}\' instead'.format(
                    custom_regressor, selector))
        else:
            raise ValueError(
                '\'{}\' is not a valid selector for SingleTargetMethod'.format(selector))

    def __implements_SciKitLearn_API(self, object):
        # TODO: Move to framework.utils or make package private
        fit = getattr(object, 'fit', None)
        predict = getattr(object, 'predict', None)
        if(fit is not None and predict is not None):
            return True
        return False

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
    # TODO: Add Batch Mechanism
    # TODO: Add Data Loaders
    # TODO: Add Validation set for model selection
    # FIXME: Naming Inconsistencies (y_train, y_data) -> find scheme to apply everywhere
    PATH = "autoncoder_bestmodel_validation"

    def __init__(self, regressor='gradientboost', custom_regressor=None, learning_rate=1e-3, print_after_epochs=500, use_gpu=False):
        self.learning_rate = learning_rate
        self.print_after_epochs = print_after_epochs
        self.num_epochs = 5000
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
        if (custom_regressor is not None and self.__implements_SciKitLearn_API(custom_regressor)):
            try:
                self.regressor = custom_regressor
            finally:
                pass
            return
        elif(regressor.lower() in ESTIMATORS):
            self.regressor = ESTIMATORS[regressor.lower()]
            if custom_regressor is not None:
                raise Warning('\'{}\' is not valid regressor using \'{}\' instead'.format(
                    custom_regressor, regressor))
        else:
            raise ValueError(
                '\'{}\' is not a valid selector for SingleTargetMethod'.format(regressor))

    def __implements_SciKitLearn_API(self, object):
        # TODO: Move to framework.utils
        fit = getattr(object, 'fit', None)
        predict = getattr(object, 'predict', None)
        if(fit is not None and predict is not None):
            return True
        return False

    def fit(self, X_train, y_train):
        # X_train, y_train = __scaleTrainigSet(X_train, y_train)
        n_targets = len(y_train[0])
        y_train, y_val = train_test_split(y_train, test_size=0.1)
        y_data = torch.tensor(y_train, dtype=torch.float).to(self.Device)
        y_data_val = torch.tensor(y_val, dtype=torch.float).to(self.Device)

        model = autoencoder(n_targets).to(self.Device)
        best_model, best_score = None, np.inf
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        val_losses = []

        for epoch in range(self.num_epochs):
            model.train()
            # ===================forward=====================
            output = model(y_data)
            loss = criterion(output, y_data)
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
                torch.save(model.state_dict(), PATH)

            # ===================log========================
            if epoch % print_after_epochs == 0:
                print('epoch [{}/{}], train_loss:{} '.format(
                    epoch + 1, self.num_epochs, loss))

        self.best_model = autoencoder(n_targets)
        self.best_model.load_state_dict(torch.load(PATH))
        self.best_model.to(self.Device)
        y_enc_train = self.best_model.encoder(y_data)

        if self.Device is 'cpu':
            self.regressor.fit(X_train, y_enc_train.detach().numpy().ravel())
        else:
            self.regressor.fit(
                X_train, y_enc_train.cpu().detach().numpy().ravel())
        return self

    def predict(self, X_test):
        y_pred_test = self.regressor.predict(X_test)
        y_pred_test_t = torch.tensor(
            y_pred_test, dtype=torch.float).unsqueeze(1).to(self.Device)
        y_pred_dec = self.best_model.decoder(y_pred_test_t)

        return y_pred_dec.detach().numpy() if self.Device is 'cpu' else y_pred_dec.cpu().detach().numpy()

    # def __scaleTrainigSet(self,X_train, y_train):
    #     scaler_x_train = StandardScaler()
    #     scaler_x_train.fit(X_train)
    #     X_train = scaler_x_train.transform(X_train)

    #     scaler_y_train = StandardScaler()
    #     scaler_y_train.fit(y_train)
    #     y_train = scaler_y_train.transform(y_train)
    #     return X_train,y_train


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
