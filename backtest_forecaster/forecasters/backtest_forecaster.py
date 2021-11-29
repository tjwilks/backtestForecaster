from abc import abstractmethod, ABC
import pandas as pd
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from typing import Dict
from collections import namedtuple
import numpy as np
import re


class BacktestForecaster:
    """

    """

    def get_windows(self):
        final_index = self.index_lookup["numerical_date_index_lookup"].max()
        test_index_starts = reversed(range(self.min_train_window_len, final_index))
        get_horizon = lambda test_index_start: (
            self.max_horizon if (test_index_start + self.max_horizon < final_index)
            else final_index - test_index_start
        )
        window = namedtuple("Window", "train_index test_index")
        windows = [
            window(
                train_index=list(range(test_index_start)),
                test_index=list(range(test_index_start, test_index_start + get_horizon(test_index_start)))
            )
            for test_index_start in test_index_starts
        ]
        return windows

    @staticmethod
    def get_index_lookup(series_data):
        series_data_min_date = series_data["date_index"].min().date()
        series_data_max_date = series_data["date_index"].max().date()
        series_data_date_range = pd.date_range(
            start=series_data_min_date,
            end=series_data_max_date,
            freq="MS",
            closed=None
        )
        index = range(0, len(series_data_date_range))
        index_lookup = pd.DataFrame({
            "date_index_lookup": series_data_date_range,
            "numerical_date_index_lookup": index
        })
        return index_lookup

    def get_backtest_models_and_forecasts(self):
        all_fit_models = {}
        backtest_forecasts = []
        unique_series_ids = self.series_data["series_id"].unique()
        for series_id in unique_series_ids:
            series = self.series_data[self.series_data["series_id"] == series_id]
            fit_models_all_windows, series_models_forecasts = self.get_model_forecasts_all_windows(
                tseries=series,
                windows=self.windows,
            )
            series_models_forecasts["series_id"] = series_id
            all_fit_models[series_id] = fit_models_all_windows
            backtest_forecasts.append(series_models_forecasts)
        backtest_forecasts = pd.concat(backtest_forecasts)
        backtest_forecasts = self.post_forecast_forematting(backtest_forecasts)
        return all_fit_models, backtest_forecasts

    def get_model_forecasts_all_windows(self, tseries, windows):
        max_windows = self.max_windows if len(windows) > self.max_windows else len(windows)
        models_forecasts_backtest_all_windows = []
        fit_models_all_windows = {}
        for window in windows[:max_windows]:
            train_series = tseries.iloc[window.train_index, :]
            test_series = tseries.iloc[window.test_index, :]
            assert train_series["numerical_date_index"].max() + 1 == test_series["numerical_date_index"].min()
            fit_models = self.get_fit_models(
                train_series
            )
            models_backtest_window = self.get_forecasts_all_models(
                fit_models,
                test_series
            )
            fit_models_all_windows[test_series["numerical_date_index"].min()] = fit_models
            models_forecasts_backtest_all_windows.append(models_backtest_window)
        models_forecasts_backtest_all_windows = pd.concat(models_forecasts_backtest_all_windows)
        return fit_models_all_windows, models_forecasts_backtest_all_windows

    @abstractmethod
    def get_fit_models(self, train_series, test_series):
        pass

    @abstractmethod
    def get_forecasts_all_models(self, train_series, test_series):
        pass

    def post_forecast_forematting(self, backtest_forecasts):
        backtest_forecasts = self.replace_numerical_with_dates(
            data=backtest_forecasts,
            numerical_col="numerical_predict_from"
        )
        backtest_forecasts = self.replace_numerical_with_dates(
            data=backtest_forecasts,
            numerical_col="numerical_date_index"
        )
        backtest_forecasts = backtest_forecasts.sort_values(
            by=["series_id", "predict_from", "date_index"],
            ascending=False
        )
        return backtest_forecasts

    def replace_dates_with_numerical(self, data, columns_to_replace):
        columns_to_replace = [columns_to_replace] if type(columns_to_replace) == str else columns_to_replace
        for column in columns_to_replace:
            data = data.set_index(column).join(
                other=self.index_lookup.set_index("date_index_lookup")
            ).reset_index(drop=True)
            data = data.rename({"numerical_date_index_lookup": f"numerical_{column}"}, axis=1)
        numerical_columns = [f"numerical_{column}" for column in columns_to_replace]
        data = data.sort_values(by=["series_id"] + numerical_columns, ascending=False)
        return data

    def replace_numerical_with_dates(self, data, numerical_col):
        data = data.set_index(numerical_col).join(
            other=self.index_lookup.set_index("numerical_date_index_lookup")
        ).reset_index(drop=True)
        data[re.sub("^numerical_", "", numerical_col)] = data["date_index_lookup"]
        data = data.drop("date_index_lookup", axis=1)
        return data

    def get_hyp_param_df(self):
        model_names = self.models.keys()
        hyp_params = {
            model_name: model_name.split("--")
            for model_name in model_names
        }
        hyp_params_datasets = []
        for model_name, hyp_params in hyp_params.items():
            algo_name = hyp_params[0].split("--")[0]
            hyp_params = [
                f"{algo_name}-{re.sub(r'^p[0-9]{1}_', '', hyp_param)}" for hyp_param in hyp_params
            ][1:]
            hyp_params = {
                hyp_param.split(":")[0] : hyp_param.split(":")[1]
                for hyp_param in hyp_params
            }
            hyp_params = pd.DataFrame(hyp_params, index=[model_name])
            hyp_params_datasets.append(hyp_params)
        hyp_param_df = pd.concat(hyp_params_datasets)
        return hyp_param_df


class PrimitiveModelBacktestForecaster(BacktestForecaster):

    def __init__(
            self,
            series_data: pd.DataFrame,
            models,
            max_horizon: int = 12,
            min_train_window_len: int = 12,
            max_windows=30
    ):
        self.series_data = series_data
        self.index_lookup = self.get_index_lookup(self.series_data)
        self.series_data = self.replace_dates_with_numerical(self.series_data, "date_index")
        self.max_horizon = max_horizon
        self.min_train_window_len = min_train_window_len
        self.max_windows = max_windows
        self.windows = self.get_windows()
        self.models = models

    def get_fit_models(self, train_series):
        y_train = np.array(train_series["actuals"], dtype=np.float64)
        fit_models = dict()
        for primitive_model_name, primitive_model in self.models.copy().items():
            primitive_model.fit(y=y_train)
            fit_models[primitive_model_name] = primitive_model
        return fit_models

    def get_forecasts_all_models(self, fit_models, test_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        y_test = np.array(test_series["actuals"], dtype=np.float64)
        primitive_model_forecasts = dict()
        primitive_model_forecasts["numerical_predict_from"] = test_series["numerical_date_index"].max()
        primitive_model_forecasts["numerical_date_index"] = test_series["numerical_date_index"]
        primitive_model_forecasts["actuals"] = y_test
        for primitive_model_name, primitive_model in self.models.copy().items():
            forecast = primitive_model.predict(h=len(test_series))
            primitive_model_forecasts[primitive_model_name] = forecast
        primitive_model_forecasts = pd.DataFrame(primitive_model_forecasts, index=test_series.index)
        return primitive_model_forecasts


class CombinerBacktestForecaster(BacktestForecaster):

    def __init__(
            self,
            forecast_data: pd.DataFrame,
            models,
            max_horizon: int = 12,
            min_train_window_len: int = 12,
            max_windows: int = 30,
            horizon_length: int = 1
    ):

        # replace date columns with numerical
        self.index_lookup = self.get_index_lookup(forecast_data)
        forecast_data = self.replace_dates_with_numerical(forecast_data, ["predict_from", "date_index"])
        self.series_data = self._handle_horizons(forecast_data, horizon_length)
        self.models = models
        self.max_horizon = max_horizon
        self.min_train_window_len = min_train_window_len
        self.windows = self.get_windows()
        self.max_windows = max_windows

    def get_fit_models(self, train_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        X_train = train_series.drop(
            columns=["numerical_predict_from", "numerical_date_index",
                     "actuals", "series_id"]).apply(pd.to_numeric)
        y_train = train_series["actuals"]
        fit_models = dict()
        for combiner_model_name, combiner_model in self.models.copy().items():
            combiner_model.fit(
                x=X_train,
                y=y_train,
                epochs=100
            )
            fit_models[combiner_model_name] = combiner_model
        return fit_models

    def get_forecasts_all_models(self, fit_models, test_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        # Check test set starts 1 point on from train set
        X_test = test_series.drop(
            columns=["numerical_predict_from", "numerical_date_index",
                     "actuals", "series_id"]).apply(pd.to_numeric)
        y_test = test_series["actuals"]
        combiner_forecasts = dict()
        combiner_forecasts["numerical_predict_from"] = test_series["numerical_date_index"].min()
        combiner_forecasts["numerical_date_index"] = test_series["numerical_date_index"]
        combiner_forecasts["actuals"] = y_test
        for combiner_model_name, combiner_model in fit_models.copy().items():
            forecast = combiner_model.predict(x=X_test)
            combiner_forecasts[combiner_model_name] = forecast
        combiner_forecasts = pd.DataFrame(combiner_forecasts, index=test_series.index)
        return combiner_forecasts

    @staticmethod
    def _handle_horizons(forecast_data, horizon_length):
        forecast_data["horizon"] = (
                forecast_data["numerical_predict_from"] - forecast_data["numerical_date_index"] + 1
        )
        forecast_data = forecast_data[forecast_data["horizon"] == horizon_length]
        forecast_data = forecast_data.drop(["numerical_date_index", "horizon"], axis=1)
        series_data = forecast_data.groupby(
            by=["series_id", "numerical_predict_from"],
            as_index=False).sum()
        min_numerical_predict_from = series_data.groupby(
            by="series_id")["numerical_predict_from"].min("numerical_predict_from")
        min_numerical_predict_from.name = "min_numerical_predict_from"
        series_data = series_data.merge(
            right=min_numerical_predict_from, on="series_id", how="inner"
        )
        series_data["numerical_date_index"] = (
                series_data["numerical_predict_from"] - series_data["min_numerical_predict_from"] + 1
        )
        series_data = series_data.drop("min_numerical_predict_from", axis=1)
        return series_data

    def get_primitive_model_weights_all_combiners(self, all_fit_models):
        weights_datasets = []
        index_n = 0
        for series_id, fit_models_all_windows in all_fit_models.items():
            for predict_from, fit_models in fit_models_all_windows.items():
                for combiner_model_name, combiner_model in fit_models.items():
                    model_weights = combiner_model.get_weights()
                    primitive_model_names = self.series_data.drop(
                        labels=["series_id", "numerical_predict_from",
                                "numerical_date_index", "actuals"],
                        axis=1
                    ).columns
                    weights_data = {
                        primitive_model_name: model_weight
                        for primitive_model_name, model_weight
                        in zip(primitive_model_names, model_weights)
                    }
                    weights_data = pd.DataFrame(weights_data, index=[index_n])
                    weights_data["combiner_model_name"] = combiner_model_name
                    weights_data["predict_from"] = predict_from
                    weights_data["series_id"] = series_id
                    weights_datasets.append(weights_data)
                    index_n += 1
        primitive_model_weights_all_combiners = pd.concat(weights_datasets)
        primitive_model_weights_all_combiners = self.replace_numerical_with_dates(
            data=primitive_model_weights_all_combiners,
            numerical_col="predict_from"
        )
        return primitive_model_weights_all_combiners