from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor


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
            'mlp': MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), max_iter=1000, random_state=1),
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
        fit = getattr(object, 'fit', None)
        predict = getattr(object, 'predict', None)
        if(fit is not None and predict is not None):
            return True
        return False

    def fit(self, X_train, y_train):
        """Fits the estimator to the training data

        Args:
            X_train (np.array): Training set descriptive variables
            y_train (np.array): Training set target variables

        Returns:
            [sklearn.MultiOutputRegressor]: Trained estimator
        """
        self.MORegressor.fit(X_train, y_train)
        return self.MORegressor

    def predict(self, X_test):
        """Predicts the target variables for a given set of descriptive variables

        Args:
            X_test (np.array): Set of descriptive variables

        Returns:
            [np.array]: Set of predicted target variables
        """
        result = self.MORegressor.predict(X_test)
        return result
