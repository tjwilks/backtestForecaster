import numpy as np
import pandas as pd
from decimal import Decimal
import math
from sklearn import preprocessing


class AdaptiveHedge:

    def __init__(self, alpha, multiplier):
        self.alpha = alpha
        self.multiplier = multiplier

    def fit(self, x):
        exp_losses = {}
        sum_losses = {}
        models = x.columns
        error_df = pd.DataFrame(preprocessing.normalize(x), columns=x.columns)
        for model in models:
            exp_decay = [(1 - self.alpha) ** i for i in range(len(error_df[model]))]
            exp_decay_error = error_df[model] * exp_decay
            sum_loss = abs(exp_decay_error).sum()
            exp_loss = math.e**(-self.multiplier * sum_loss)
            sum_losses[model] = sum_loss
            exp_losses[model] = exp_loss
        sum_exp_losses = sum(exp_losses.values())
        self.weights = {model: exp_loss/sum_exp_losses for model, exp_loss in exp_losses.items()}
        max_exp_loss = max(exp_losses.values())
        max_exp_loss_model = [model for model, exp_loss in exp_losses.items() if exp_loss == max_exp_loss][0]
        max_weight = max(self.weights.values())
        max_weight_model = [model for model, weight in self.weights.items() if weight == max_weight][0]
        print(f"min exp loss model: {max_exp_loss_model} vs max weight model: {max_weight_model}")
        if max_weight_model != max_exp_loss_model:
            print(exp_decay)
            for model in exp_losses.keys():
                print(f"weight: {self.weights[model]}, exp_loss: {exp_losses[model]}, sum loss: {sum_losses[model]}, model: {model}")
        assert max_weight_model == max_exp_loss_model, f"{exp_losses}, \n {self.weights}"
        self.is_fit = True

    def predict(self, x):
        x_prepared = np.array(x, dtype=float)
        return (np.array(list(self.weights.values())) * x_prepared).sum(axis=1)

    def get_weights(self):
        if self.is_fit:
            return np.array(list(self.weights.values()))
        else:
            raise ValueError("Method has not been fit to data yet")


class FollowTheLeader:

    def __init__(self):
        pass

    def fit(self, x):
        model_errors = {}
        models = x.columns
        for model in models:
            error = x[model].to_numpy()
            error = abs(error).sum()
            model_errors[model] = error
        min_error = min(model_errors.values())
        weights = {model: (1 if error == min_error else 0) for model, error in model_errors.items()}
        min_models = [model for model, weight in weights.items() if weight == 1]
        print(min_models)
        self.weights = {model: (1 if model == min_models[0] else 0) for model, weight in weights.items()}
        self.is_fit = True

    def predict(self, x):
        x_prepared = np.array(x, dtype=float)
        return (np.array(list(self.weights.values())) * x_prepared).sum(axis=1)

    def get_weights(self):
        if self.is_fit:
            return np.array(list(self.weights.values()))
        else:
            raise ValueError("Method has not been fit to data yet")


# x_train = pd.DataFrame({"m1": [2, 2, 2], "m2": [3, 3, 3],"m3": [2, 3, 3],"m4": [3, 3, 4],"m5": [0, 0, 1]})
# y_train = pd.Series([2.5, 2.5, 2.5])
# x_test = pd.DataFrame({"m1": [1, 1, 1], "m2": [1.5, 1.5, 1.5], "m3": [2, 3, 3], "m4": [3, 3, 4], "m5": [0, 0, 1]})
# y_test = pd.Series([1.25, 1.25, 1.25])
# followTheLeader = FollowTheLeader()
# followTheLeader.fit(x=x_train, y=y_train)
# print(followTheLeader.get_weights())
# followTheLeader_forecast = followTheLeader.predict(x=x_test)
# print(followTheLeader_forecast)