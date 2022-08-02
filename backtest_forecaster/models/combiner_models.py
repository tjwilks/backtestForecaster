import numpy as np
import pandas as pd
from decimal import Decimal
import math
from sklearn import preprocessing
from abc import ABC, abstractmethod


class AbstractCombinerModel(ABC):

    def __init__(self):
        self.is_fit = False
    @abstractmethod
    def fit(self, x):
        """
        Fit the method to error data x

        :param x: primitive model error data
        """

    def predict(self, x):
        """
        Predict by combining forecast data x

        :param x: primitive model forecast data
        """
        x_prepared = np.array(x, dtype=float)
        prediction = (np.array(list(self.weights.values())) *
                      x_prepared).sum(axis=1)
        return prediction

    def get_weights(self):
        """
        Get weights from combiner fitting
        """
        if self.is_fit:
            return np.array(list(self.weights.values()))
        else:
            raise ValueError("Method has not been fit to data yet")


class AdaptiveHedge(AbstractCombinerModel):

    def __init__(self, alpha, multiplier):
        self.alpha = alpha
        self.multiplier = multiplier

    def fit(self, x):
        exp_losses = {}
        sum_losses = {}
        models = x.columns
        error_df = pd.DataFrame(preprocessing.normalize(x), columns=x.columns)
        for model in models:
            exp_decay = [(1 - self.alpha) ** i
                         for i in range(len(error_df[model]))]
            exp_decay_error = error_df[model] * exp_decay
            sum_loss = abs(exp_decay_error).sum()
            exp_loss = math.e**(-self.multiplier * sum_loss)
            sum_losses[model] = sum_loss
            exp_losses[model] = exp_loss
        sum_exp_losses = sum(exp_losses.values())
        self.weights = {model: exp_loss/sum_exp_losses
                        for model, exp_loss in exp_losses.items()}
        max_exp_loss = max(exp_losses.values())
        max_exp_loss_model = [model for model, exp_loss
                              in exp_losses.items()
                              if exp_loss == max_exp_loss][0]
        max_weight = max(self.weights.values())
        max_weight_model = [model for model, weight
                            in self.weights.items()
                            if weight == max_weight][0]
        assert max_weight_model == max_exp_loss_model, \
            f"{exp_losses}, \n {self.weights}"
        self.is_fit = True


class FollowTheLeader(AbstractCombinerModel):

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
        weights = {model: (1 if error == min_error else 0)
                   for model, error in model_errors.items()}
        min_models = [model for model, weight
                      in weights.items()
                      if weight == 1]
        self.weights = {model: (1 if model == min_models[0] else 0)
                        for model, weight
                        in weights.items()}
        self.is_fit = True