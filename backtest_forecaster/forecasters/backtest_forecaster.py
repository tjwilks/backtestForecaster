from abc import abstractmethod, ABC
import pandas as pd
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from backtest_forecaster.models.combiner_models import AbstractCombinerModel
from typing import Dict, Union, Tuple, List
from collections import namedtuple
import numpy as np
import re
from datetime import date
from dateutil.relativedelta import relativedelta
import copy


class AbstractBacktestForecaster(ABC):
    """
    Abstract class that defines forecaster interface.
    """
    @abstractmethod
    def get_backtest_models_and_forecasts(self) -> Tuple[
        Dict[str, Dict[pd.Timestamp, Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]]]],
            pd.DataFrame]:
        """
        Gets backtest forecasts for all models for each window of every
        time-series
        """

    @abstractmethod
    def _get_fit_models(
            self,
            train_series: pd.DatetimeIndex
    ) -> Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]]:
        """
        Fits all models to training data of a single period
        of a single time series

        param train_series: training data of a single backtest window
        of a single time series for models to be fit to.

        returns fit_models: models fit to training data
        """

    @abstractmethod
    def _get_forecasts_all_models(
            self,
            fit_models: Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]],
            test_series: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Forecasts all models for a test period of a single backtest window
        of a single time series

        param fit_models: all models that have already been fit to the
        training data for the test series specified

        param test_series: test data of a single backtest window
        of a single time series for a forecast to be produced for

        returns: model forecasts and actuals for a single
        test period of a single backtest window of a single time
        series
        """

    @abstractmethod
    def _get_windows(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generates all possible windows for all time series within
        time_series_data
        """

    @abstractmethod
    def _get_model_forecasts_all_windows(
            self,
            time_series: pd.DatetimeIndex
    ) -> Tuple[Dict[pd.Timestamp, Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]]], pd.DataFrame]:
        """
        Produces forecasts for all models for periods specified for a
        single time series
        """
    @abstractmethod
    def _get_train_and_test_series(self, time_series, window):
        """
        Extract train and test series from time series based on indexes
        specified in window
        """


class BaseBacktestForecaster(AbstractBacktestForecaster):

    def __init__(
            self,
            time_series_data: pd.DataFrame,
            models: Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]],
            max_horizon: int,
            max_train_window_len: int,
            min_train_window_len: int,
            max_windows:int,
            window_index: str
    ):
        """
        param time_series_data: multiple time series containing a series
        identifier, date index and actuals column or multiple forecasts from
        multiple time series containing a series identifier, date index,
        predict_from and actuals column

        param models: dictionary containing all  models and equivalent model
        names

        param max_horizon: maximum horizon to forecast models to

        param min_train_window_len: minimum length of training data required
        for window to be used

        param max_train_window_len: maximum length of training data to be used
        for every window

        param max_windows: maximum number of backtest windows to collect
        primitive model forecasts for

        """
        self.time_series_data = time_series_data
        self.models = models
        self.max_horizon = max_horizon
        self.min_train_window_len = min_train_window_len
        self.max_train_window_len = max_train_window_len
        self.max_windows = max_windows
        self.window_index = window_index
        self.windows = self._get_windows()

    def get_backtest_models_and_forecasts(self) -> Tuple[
        Dict[str, Dict[pd.Timestamp, Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]]]],
            pd.DataFrame]:
        """
        Gets backtest forecasts for all models for each window of every
        time-series
        """
        all_fit_models = {}
        backtest_forecasts = []
        unique_series_ids = self.time_series_data["series_id"].unique()
        for series_id in unique_series_ids:
            time_series = self.time_series_data[self.time_series_data["series_id"] == series_id]
            fit_models_all_windows, series_models_forecasts = self._get_model_forecasts_all_windows(
                time_series=time_series
            )
            series_models_forecasts["series_id"] = series_id
            all_fit_models[series_id] = fit_models_all_windows
            backtest_forecasts.append(series_models_forecasts)
        backtest_forecasts = pd.concat(backtest_forecasts)
        backtest_forecasts = backtest_forecasts.sort_values(
            by=["series_id", "date_index"],
            ascending=False
        )
        return all_fit_models, backtest_forecasts

    def _get_windows(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generates all possible windows for all time series within
        time_series_data
        """
        final_index = self.time_series_data[self.window_index].max()
        first_index = self.time_series_data[self.window_index].min()
        test_index_starts = pd.date_range(first_index, final_index, freq="MS", inclusive="right").tolist()[::-1]
        get_horizon = lambda test_index_start: (
            test_index_start + relativedelta(months=+self.max_horizon - 1)
            if test_index_start + relativedelta(months=+self.max_horizon) <= final_index
            else final_index
        )
        get_train_index_start = lambda test_index_start: (
            test_index_start - relativedelta(months=+self.max_train_window_len)
            if (len(pd.date_range(first_index, test_index_start, freq="MS", inclusive="left").tolist())
                > self.max_train_window_len)
            else first_index
        )
        window = namedtuple("Window", "train_index test_index")
        windows = [
            window(
                train_index=pd.date_range(get_train_index_start(test_index_start), test_index_start,
                                          freq="MS", inclusive="left").tolist(),
                test_index=pd.date_range(test_index_start, get_horizon(test_index_start), freq="MS").tolist()
            )
            for test_index_start in test_index_starts
        ]
        windows = [
            window for window in windows if len(window.train_index) >= self.min_train_window_len
        ]
        max_windows = self.max_windows if len(windows) > self.max_windows else len(windows)
        windows = windows[:max_windows]
        return windows

    def _get_model_forecasts_all_windows(
            self,
            time_series: pd.DatetimeIndex
    ) -> Tuple[Dict[pd.Timestamp, Dict[str, Union[AbstractPrimitiveModel, AbstractCombinerModel]]], pd.DataFrame]:
        """
        Produces forecasts for all models for periods specified for a
        single time series
        """
        max_windows = self.max_windows if len(self.windows) > self.max_windows else len(self.windows)
        models_forecasts_backtest_all_windows = []
        fit_models_all_windows = {}
        for window in self.windows[:max_windows]:
            train_series, test_series = self._get_train_and_test_series(time_series, window)
            fit_models = self._get_fit_models(
                train_series
            )
            models_backtest_window = self._get_forecasts_all_models(
                fit_models,
                test_series
            )
            fit_models_all_windows[test_series["date_index"].min()] = fit_models
            models_forecasts_backtest_all_windows.append(models_backtest_window)
        models_forecasts_backtest_all_windows = pd.concat(models_forecasts_backtest_all_windows)
        return fit_models_all_windows, models_forecasts_backtest_all_windows

    def _get_fit_models(
            self,
            train_series: pd.DatetimeIndex
    ):
        raise NotImplementedError("Must override _get_fit_models")

    def _get_forecasts_all_models(
                self,
                fit_models: Dict[str, AbstractPrimitiveModel],
                test_series: pd.DatetimeIndex
        ) -> pd.DataFrame:
        raise NotImplementedError("Must override _get_forecasts_all_models")

    def _get_train_and_test_series(self, time_series, window):
        raise NotImplementedError("Must override _get_train_and_test_series")


class PrimitiveModelBacktestForecaster(BaseBacktestForecaster):
    """
    Creates backtest of forecasts for primitive models
    """
    def __init__(
            self,
            time_series_data: pd.DataFrame,
            models: Dict[str, AbstractPrimitiveModel],
            max_horizon: int = 12,
            max_train_window_len: int = 12,
            min_train_window_len: int = 12,
            max_windows=30
    ):
        """
        param time_series_data: multiple time series containing a series
        identifier, date index and actuals collum

        param models: dictionary containing all primitive models and equivalent
        primitive model names

        param max_horizon: maximum horizon to forecast primitive models to

        param min_train_window_len: minimum length of training data required
        for window to be used

        param max_train_window_len: maximum length of training data to be used
        for every window

        param max_windows: maximum number of backtest windows to collect
        primitive model forecasts for
        """
        super().__init__(
            time_series_data=time_series_data,
            models=models,
            max_horizon=max_horizon,
            max_train_window_len=max_train_window_len,
            min_train_window_len=min_train_window_len,
            max_windows=max_windows,
            window_index="date_index"
        )

    def _get_fit_models(
            self,
            train_series: pd.DatetimeIndex
    ) -> Dict[str, AbstractPrimitiveModel]:
        """
        Fits all primitive models to training data of a single period of
        a single time series

        param train_series: training data of a single backtest window
        of a single time series for models to be fit to.

        returns fit_models: primitive models fit to training data
        """
        y_train = np.array(train_series["actuals"], dtype=np.float64)
        fit_models = dict()
        for primitive_model_name, primitive_model in self.models.copy().items():
            primitive_model.fit(y=y_train)
            fit_models[primitive_model_name] = primitive_model
        return fit_models

    def _get_forecasts_all_models(
            self,
            fit_models: Dict[str, AbstractPrimitiveModel],
            test_series: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Forecasts all primitive models for a test period of a single
        backtest window of a single time series

        param fit_models: all primitive models that have already been
        fit to the training data for the test series specified

        param test_series: test data of a single backtest window
        of a single time series for a forecast to be produced for

        returns: primitive model forecasts and actuals for a single
        test period of a single backtest window of a single time
        series
        """
        y_test = np.array(test_series["actuals"], dtype=np.float64)
        primitive_model_forecasts = dict()
        primitive_model_forecasts["predict_from"] = test_series["date_index"].min()
        primitive_model_forecasts["date_index"] = test_series["date_index"]
        primitive_model_forecasts["actuals"] = y_test
        for primitive_model_name, primitive_model in self.models.copy().items():
            forecast = primitive_model.predict(h=len(test_series))
            primitive_model_forecasts[primitive_model_name] = forecast
        primitive_model_forecasts = pd.DataFrame(
            data=primitive_model_forecasts, 
            index=test_series.index
        )
        return primitive_model_forecasts

    def _get_train_and_test_series(self, time_series, window):
        train_series = time_series.loc[time_series["date_index"].isin(window.train_index), :]
        test_series = time_series.loc[time_series["date_index"].isin(window.test_index), :]
        return train_series, test_series


class CombinerBacktestForecaster(BaseBacktestForecaster):
    """
    Creates backtest of forecasts for combiner models
    """
    def __init__(
            self,
            forecast_data: pd.DataFrame,
            models: Dict[str, AbstractCombinerModel],
            max_horizon: int = 12,
            min_train_window_len: int = 12,
            max_train_window_len: int = 12,
            max_windows: int = 30,
            horizon_length: int = 1
    ):
        """
        :param forecast_data: multiple forecasts from multiple time series
        containing a series identifier, date index, predict_from and actuals
        column

        param models: dictionary containing all combiner models and equivalent
        combiner model names

        param max_horizon: maximum horizon to forecast primitive models to

        param min_train_window_len: minimum length of training data required
        for window to be used

        param max_train_window_len: maximum length of training data to be used
        for every window

        param max_windows: maximum number of backtest windows to collect
        primitive model forecasts for

        param horizon_length: length of horizon of forecasts to use to
        calculate total forecast error for a given predict from
        """
        super().__init__(
            time_series_data=forecast_data,
            models=models,
            max_horizon=max_horizon,
            max_train_window_len=max_train_window_len,
            min_train_window_len=min_train_window_len,
            max_windows=max_windows,
            window_index="predict_from"
        )
        self.horizon_length = horizon_length

    def _get_fit_models(
            self,
            train_series: pd.DatetimeIndex
    ) -> Dict[str, AbstractCombinerModel]:
        """
        Fits all combiner models to training data of a single period
        of a single time series

        param train_series: training data of a single backtest window
        of a single time series for models to be fit to.

        returns fit_models: combiner models fit to training data
        """
        X_train = train_series.drop(
            columns=["predict_from", "date_index", "actuals", "series_id"]).apply(pd.to_numeric)
        y_train = train_series["actuals"]
        fit_models = dict()
        for combiner_model_name, combiner_model in self.models.copy().items():
            combiner_model = copy.copy(combiner_model)
            combiner_model.fit(
                x=X_train,
            )
            fit_models[combiner_model_name] = combiner_model
        return fit_models

    def _get_forecasts_all_models(
            self,
            fit_models: Dict[str, AbstractCombinerModel],
            test_series: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Forecasts all combiner models for a test period of a single
        backtest window of a single time series

        param fit_models: all combiner models that have already been fit to the
        training data for the test series specified

        param test_series: test data of a single backtest window
        of a single time series for a forecast to be produced for

        returns: combiner model forecasts and actuals for a single
        test period of a single backtest window of a single time
        series
        """
        first_predict_from = test_series["predict_from"].tolist()[0]
        test_series = test_series[test_series["predict_from"] == first_predict_from]
        X_test = test_series.drop(
            columns=["predict_from", "date_index", "actuals", "series_id"]).apply(pd.to_numeric)
        combiner_forecasts = dict()
        combiner_forecasts["predict_from"] = test_series["predict_from"].min()
        combiner_forecasts["date_index"] = test_series["date_index"]
        combiner_forecasts["actuals"] = test_series["actuals"]
        for combiner_model_name, combiner_model in fit_models.copy().items():
            forecast = combiner_model.predict(x=X_test)
            combiner_forecasts[combiner_model_name] = forecast
        combiner_forecasts = pd.DataFrame(combiner_forecasts, index=test_series.index)
        return combiner_forecasts

    def _get_train_and_test_series(self, time_series, window):
        train_series = time_series[((time_series["predict_from"].isin(window.train_index)) &
                                   (time_series["date_index"] < window.test_index[0]))]
        train_series = self._filter_forecast_horizon(train_series, self.horizon_length)
        train_series = self._replace_forecasts_with_error(train_series)
        train_series = self._aggregate_forecast_horizon(train_series)
        test_series = time_series[((time_series["predict_from"] == window.test_index[0]) &
                                   (time_series["date_index"].isin(window.test_index)))]
        return train_series, test_series

    @staticmethod
    def _aggregate_forecast_horizon(
            forecast_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregates all time points of forecasts over a given horizon length
        to a single time point

        param forecast_data: data containing forecasts to aggregated

        param horizon_length: length of horizon to use in aggregation
        """
        forecast_data = forecast_data.drop(["date_index", "horizon"], axis=1)
        time_series_data = forecast_data.groupby(
            by=["series_id", "predict_from"],
            as_index=False).sum()
        time_series_data["date_index"] = time_series_data["predict_from"]
        return time_series_data

    @staticmethod
    def _filter_forecast_horizon(forecast_data, horizon_length):
        forecast_data["horizon"] = round(
           (forecast_data['date_index'] - forecast_data['predict_from']) / np.timedelta64(1, 'M')
        ).astype(int) + 1
        forecast_data = forecast_data[forecast_data["horizon"] <= horizon_length]
        return forecast_data

    @staticmethod
    def _replace_forecasts_with_error(train_series):
        models = train_series.drop(columns=["predict_from", "date_index", "actuals", "horizon", "series_id"]).columns
        for model in models:
            train_series[model] = abs(train_series[model] - train_series["actuals"])
        return train_series