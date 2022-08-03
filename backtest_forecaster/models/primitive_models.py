from abc import ABC, abstractmethod
import math
from typing import List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters.model import \
    ExponentialSmoothing as smExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class AbstractPrimitiveModel(ABC):

    def __init__(self):
        self.is_fit = False

    @abstractmethod
    def fit(self, y: pd.Series):
        """
        Fit the method to timeseries y

        :param y: timeseries to forecast
        """
        pass

    @abstractmethod
    def predict(self, h: int) -> pd.Series:
        """
        Forecast the timeseries from horizon 1 to horizon h

        :param h: horizon to forecast up to
        """


class Naive(AbstractPrimitiveModel):
    """
    Naive and seasonal naive implementations
    """

    def __init__(self, seasonal_period: int = None):
        self.seasonal = seasonal_period or 1
        self.stored: Optional[List] = None

    def fit(self, y: pd.Series):
        assert len(y) > self.seasonal
        self.stored = np.array(y[-self.seasonal:])
        self.is_fit = True

    def predict(self, h: int) -> pd.Series:
        assert self.is_fit, "Model must be fitted before prediction"
        n_repeats = math.ceil(h / self.seasonal)
        repeated = np.repeat(self.stored, n_repeats)
        return pd.Series(repeated[:h])


class ExponentialSmoothing(AbstractPrimitiveModel):
    """
    Wrapper for statsmodels Exponential smoothing model
    :param trend_type: trend type.
        None - no trend, 'add' - additive, 'mul' - multiplicative
    :param seasonal_period: number of seasonal periods.
        None for no seasonality.
    :param seasonal_type: seasonal type.
        None - no seasonality, 'add' - additive, 'mul' - multiplicative
    """

    def __init__(
            self,
            trend_type: str = None,
            seasonal_period: int = None,
            seasonal_type: str = None
    ):
        if seasonal_type is not None:
            assert seasonal_period is not None, \
                "If 'seasonal_type' specified then must specify 'seasonal_period'"
        if seasonal_period is not None:
            assert seasonal_type is not None, \
                "If 'seasonal_period' specified then must specify 'seasonal_type'"
        self.seasonal = seasonal_period or 1
        self.trend_type = trend_type
        self.seasonal_type = seasonal_type
        self.model = None

    def fit(self, y: pd.Series):
        assert len(y) > self.seasonal
        if self.seasonal_type:
            self.model = smExponentialSmoothing(
                endog=y,
                seasonal_periods=self.seasonal,
                seasonal=self.seasonal_type,
                trend=self.trend_type,
                initialization_method='estimated'
            ).fit()
        else:
            self.model = smExponentialSmoothing(
                y,
                trend=self.trend_type,
                initialization_method='estimated'
            ).fit()
        self.is_fit = True

    def predict(self, h: int) -> pd.Series:
        assert self.model is not None, "Model must be fitted before prediction"
        return self.model.forecast(h)


class SARIMA(AbstractPrimitiveModel):
    """
    Wrapper for statsmodel SARIMAX model

    :param order: ARIMA order, as specified by statsmodels SARIMAX class.
        Default (1, 0, 0).
    :param seasonal_order: ARIMA seasonal order, as specified by statsmodels
        SARIMAX class. Default no seasonality. Only one of seasonal_order or
        order should be provided.
    :param trend_type: trend type as specified by statsmodels SARIMAX class
        'trend' parameter.
    """

    def __init__(
            self,
            auto_regressive: int = 1,
            integrated: int = 0,
            moving_average: int = 0,
            seasonal_auto_regressive: int = 1,
            seasonal_integrated: int = 0,
            seasonal_moving_average: int = 0,
            seasonal_period: int = 12,
            trend_type: str = None):
        self.order = (
            auto_regressive,
            integrated,
            moving_average
        )
        self.seasonal_order = (
            seasonal_auto_regressive,
            seasonal_integrated,
            seasonal_moving_average,
            seasonal_period
        )
        self.trend_type = trend_type
        self.model = None

    def fit(self, y: pd.Series):
        if self.seasonal_order is not None:
            assert len(y) > self.seasonal_order[1]
        self.model = SARIMAX(
            endog=y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend_type,
            enforce_invertibility=False,
            enforce_stationarity=False
        ).fit()
        self.is_fit = True

    def predict(self, h: int) -> pd.Series:
        assert self.model is not None, "Model must be fitted before prediction"
        return self.model.forecast(h)
