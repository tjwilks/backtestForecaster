import pytest
from pytest import fixture
import configparser
from datetime import datetime
import pandas as pd
import numpy as np
import json
from backtest_forecaster.utils import model_loader
from backtest_forecaster.forecasters.backtest_forecaster import AbstractBacktestForecaster,\
    PrimitiveModelBacktestForecaster, CombinerBacktestForecaster
pd.options.display.width = 0


class TestAbstractBacktestForecaster:
    @pytest.fixture
    def load_config(self):
        config = configparser.ConfigParser()
        config.read("tests/test_resources/test_config.ini")
        return config

    @pytest.fixture
    def load_primitive_model_backtest_forecasts(self, load_config):
        config = load_config
        local_data_paths = config["local_data_paths"]
        custom_date_parser = lambda date: datetime.strptime(date, "%Y-%m-%d")
        primitive_model_backtest_forecasts = pd.read_csv(
            filepath_or_buffer=local_data_paths["primitive_model_backtest_forecasts_path"],
            index_col=0,
            parse_dates=["predict_from", "date_index"],
            date_parser=custom_date_parser
        )
        return primitive_model_backtest_forecasts

    @pytest.fixture
    def load_combiners(self, load_config):
        config = load_config
        loader_config_path = config["loader_config_path"]
        with open(loader_config_path["combiner_loader_config_path"], 'r') as j:
            combiner_loader_config = json.loads(j.read())
        combiners = model_loader.load_models(combiner_loader_config)
        return combiners

    @pytest.fixture
    def load_combiner_backtest_forecaster(self, load_primitive_model_backtest_forecasts, load_combiners):
        primitive_model_backtest_forecasts = load_primitive_model_backtest_forecasts
        combiners = load_combiners
        combiner_backtest_forecaster = CombinerBacktestForecaster(
            forecast_data=primitive_model_backtest_forecasts,
            models=combiners,
            max_train_window_len=12,
            min_train_window_len=3,
            max_windows=6,
            horizon_length=4,
            max_horizon=4
        )
        return combiner_backtest_forecaster

    def test_get_windows(self, load_combiner_backtest_forecaster):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        windows = combiner_backtest_forecaster._get_windows()
        test_train_index_lengths = [len(window.train_index) for window in windows]
        required_train_index_lengths = list(range(12, 2, -1))
        test_required_train_index_lengths = zip(test_train_index_lengths, required_train_index_lengths)
        assert all([test == required for test, required in test_required_train_index_lengths])

        test_test_index_lengths = [len(window.test_index) for window in windows]
        required_test_index_lengths = [1, 2, 3] + ([4] * 6)
        test_required_test_index_lengths = zip(test_test_index_lengths, required_test_index_lengths)
        assert all([test == required for test, required in test_required_test_index_lengths])

        assert all([max(window.train_index) < min(window.test_index) for window in windows])

    def test_get_backtest_models_and_forecasts(self, load_combiner_backtest_forecaster):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        all_fit_models, combiner_forecasts = combiner_backtest_forecaster.get_backtest_models_and_forecasts()
        required_columns = [
            "predict_from",
            "date_index",
            "actuals",
            "AdaptiveHedge--p1_alpha:0.1--p3_multiplier:1",
            "FollowTheLeader",
            "series_id"
        ]
        test_columns = list(combiner_forecasts.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required for test, required in test_required_columns])
        n_forecasts = combiner_forecasts["predict_from"].nunique()
        assert n_forecasts == 6
        combiner_forecasts["horizon"] = round(
           (combiner_forecasts['date_index'] - combiner_forecasts['predict_from']) / np.timedelta64(1, 'M')
        ).astype(int) + 1
        max_horizon = max(combiner_forecasts["horizon"])
        assert max_horizon == 4
        assert max(combiner_forecasts["predict_from"]) == max(combiner_forecasts["date_index"])

    def test_get_model_forecasts_all_windows(self, load_combiner_backtest_forecaster):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        combiner_backtest_forecaster.time_series_data = combiner_backtest_forecaster.time_series_data[
            combiner_backtest_forecaster.time_series_data["series_id"] == "N1681"
        ]
        test_models, test_forecasts = combiner_backtest_forecaster.get_backtest_models_and_forecasts()
        test_model_dates = [datetime.date(date) for date in test_models["N1681"].keys()]
        test_forecast_dates = test_forecasts["predict_from"].dt.date.unique().tolist()
        model_and_forecast_dates = zip(test_model_dates, test_forecast_dates)
        assert all([model_date == forecast_date for model_date, forecast_date in model_and_forecast_dates])
        all_test_models = [tuple(models.keys()) for models in test_models["N1681"].values()]
        required_models = ('AdaptiveHedge--p1_alpha:0.1--p3_multiplier:1', 'FollowTheLeader')
        assert [test_models == required_models for test_models in all_test_models]

# class TestPrimitiveModelBacktestForecaster:
#     def test_get_fit_models(self):
#         assert False
#
#     def test_get_forecasts_all_models(self):
#         assert False
#
#     def test_get_train_and_test_series(self):
#         assert False
#
#
# class TestCombinerBacktestForecaster:
#
#     def test_get_fit_models(self):
#         assert False
#
#     def test_get_forecasts_all_models(self):
#         assert False
#
#     def test_get_train_and_test_series(self):
#         assert False
#
#     def test_aggregate_forecast_horizon(self):
#         assert False
#
#     def test_filter_forecast_horizon(self):
#         assert False
#
#     def test_replace_forecasts_with_error(self):
#         assert False
