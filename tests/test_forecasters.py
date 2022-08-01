import pytest
from pytest import fixture
import configparser
from datetime import datetime
import pandas as pd
import numpy as np
import json
from backtest_forecaster.utils import model_loader
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from backtest_forecaster.models.combiner_models import AbstractCombinerModel
from backtest_forecaster.forecasters.backtest_forecaster import AbstractBacktestForecaster,\
    PrimitiveModelBacktestForecaster, CombinerBacktestForecaster
pd.options.display.width = 0


class TestAbstractBacktestForecaster:
    @pytest.fixture
    def load_config(self):
        config = configparser.ConfigParser()
        config.read("tests/test_resources/test_config.ini")
        return config

    #


class TestPrimitiveModelBacktestForecaster(TestAbstractBacktestForecaster):

    @pytest.fixture
    def load_time_series_data(self, load_config):
        config = load_config
        local_data_paths = config["local_data_paths"]
        custom_date_parser = lambda date: datetime.strptime(date, "%d/%m/%Y")
        time_series_data = pd.read_csv(
            filepath_or_buffer=local_data_paths["time_series_data_path"],
            index_col=0,
            parse_dates=["date_index"],
            date_parser=custom_date_parser
        )

        return time_series_data

    @pytest.fixture
    def load_primitive_models(self, load_config):
        config = load_config
        loader_config_path = config["loader_config_path"]
        with open(loader_config_path[f"primitive_model_config_path"], 'r') as j:
            combiner_loader_config = json.loads(j.read())
        combiners = model_loader.load_models(combiner_loader_config)

        return combiners

    @pytest.fixture
    def load_primitive_model_backtest_forecaster(self, load_time_series_data, load_primitive_models):
        time_series_data = load_time_series_data
        primitive_models = load_primitive_models
        backtest_forecaster = PrimitiveModelBacktestForecaster(
            time_series_data=time_series_data,
            models=primitive_models,
            min_train_window_len=12,
            max_train_window_len=120,
            max_windows=12,
            max_horizon=4
        )
        return backtest_forecaster

    @pytest.fixture
    def load_train_and_test_series(self, load_primitive_model_backtest_forecaster):
        primitive_model_backtest_forecaster = load_primitive_model_backtest_forecaster
        time_series_data = primitive_model_backtest_forecaster.time_series_data[
            primitive_model_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_window = primitive_model_backtest_forecaster._get_windows()[-1]
        train_series, test_series = primitive_model_backtest_forecaster._get_train_and_test_series(
            time_series_data,
            test_window
        )
        return train_series, test_series

    def test_get_windows(self, load_primitive_model_backtest_forecaster):
        combiner_backtest_forecaster = load_primitive_model_backtest_forecaster
        windows = combiner_backtest_forecaster._get_windows()
        test_train_index_lengths = [len(window.train_index) for window in windows]
        required_train_index_lengths = ([120]*6) + list(range(119, 114, -1))
        test_required_train_index_lengths = zip(test_train_index_lengths, required_train_index_lengths)
        assert all([test == required for test, required in test_required_train_index_lengths])
        test_test_index_lengths = [len(window.test_index) for window in windows]
        required_test_index_lengths = [1, 2, 3] + ([4] * 6)
        test_required_test_index_lengths = zip(test_test_index_lengths, required_test_index_lengths)
        assert all([test == required for test, required in test_required_test_index_lengths])
        assert all([max(window.train_index) < min(window.test_index) for window in windows])

    def test_get_backtest_models_and_forecasts(self, load_primitive_model_backtest_forecaster):
        primitive_model_backtest_forecaste = load_primitive_model_backtest_forecaster
        all_fit_models, primitive_model_forecasts = primitive_model_backtest_forecaste.get_backtest_models_and_forecasts()
        required_columns = [
            "predict_from",
            "date_index",
            "actuals",
            "Naive",
            "SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0",
            "SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0",
            "ExponentialSmoothing--p1_trend_type:None",
            "series_id",
        ]
        test_columns = list(primitive_model_forecasts.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required for test, required in test_required_columns])
        n_forecasts = primitive_model_forecasts["predict_from"].nunique()
        assert n_forecasts == 12
        primitive_model_forecasts["horizon"] = round(
            (primitive_model_forecasts['date_index'] - primitive_model_forecasts['predict_from']) / np.timedelta64(1, 'M')
        ).astype(int) + 1
        max_horizon = max(primitive_model_forecasts["horizon"])
        assert max_horizon == 4
        assert max(primitive_model_forecasts["predict_from"]) == max(primitive_model_forecasts["date_index"])
    #
    def test_get_model_forecasts_all_windows(self, load_primitive_model_backtest_forecaster):
        primitive_model_backtest_forecaster = load_primitive_model_backtest_forecaster
        primitive_model_backtest_forecaster.time_series_data = primitive_model_backtest_forecaster.time_series_data[
            primitive_model_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_models, test_forecasts = primitive_model_backtest_forecaster.get_backtest_models_and_forecasts()
        test_model_dates = [datetime.date(date) for date in test_models["N1681"].keys()]
        test_forecast_dates = test_forecasts["predict_from"].dt.date.unique().tolist()
        model_and_forecast_dates = zip(test_model_dates, test_forecast_dates)
        assert all([model_date == forecast_date for model_date, forecast_date in model_and_forecast_dates])
        all_test_models = [tuple(models.keys()) for models in test_models["N1681"].values()]
        required_models = (
            "Naive",
            "SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0",
            "SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0",
            "ExponentialSmoothing--p1_trend_type:None"
        )
        assert [test_models == required_models for test_models in all_test_models]
#
    def test_get_fit_models(self, load_primitive_model_backtest_forecaster, load_train_and_test_series):
        primitive_model_backtest_forecaster = load_primitive_model_backtest_forecaster
        train_series, _ = load_train_and_test_series
        fit_models = primitive_model_backtest_forecaster._get_fit_models(
            train_series
        )
        assert all([isinstance(fit_model, AbstractPrimitiveModel) for fit_model in fit_models.values()])
        assert all([fit_model.is_fit for fit_model in fit_models.values()])


    def test_get_forecasts_all_models(self, load_primitive_model_backtest_forecaster, load_train_and_test_series):
        primitive_model_backtest_forecaster = load_primitive_model_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = primitive_model_backtest_forecaster._get_fit_models(
            train_series
        )
        forecasts_all_models = primitive_model_backtest_forecaster._get_forecasts_all_models(fit_models, test_series)
        required_columns = [
            'predict_from', 'date_index', 'actuals', 'Naive',
            'SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0',
            'SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0',
            'ExponentialSmoothing--p1_trend_type:None'
        ]
        test_columns = list(forecasts_all_models.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required for test, required in test_required_columns])
        assert len(forecasts_all_models) == 4
        assert all(forecasts_all_models.notnull())


class TestCombinerBacktestForecaster(TestAbstractBacktestForecaster):


    @pytest.fixture
    def load_primitive_model_forecasts(self, load_config):
        config = load_config
        local_data_paths = config["local_data_paths"]
        custom_date_parser = lambda date: datetime.strptime(date, "%Y-%m-%d")
        primitive_model_forecasts = pd.read_csv(
            filepath_or_buffer=local_data_paths["primitive_model_backtest_forecasts_path"],
            index_col=0,
            parse_dates=["predict_from", "date_index"],
            date_parser=custom_date_parser
        )
        return primitive_model_forecasts

    @pytest.fixture
    def load_combiner_models(self, load_config):
        config = load_config
        loader_config_path = config["loader_config_path"]
        with open(loader_config_path[f"combiner_loader_config_path"], 'r') as j:
            combiner_loader_config = json.loads(j.read())
        combiners = model_loader.load_models(combiner_loader_config)

        return combiners

    @pytest.fixture
    def load_combiner_backtest_forecaster(self, load_primitive_model_forecasts, load_combiner_models):
        primitive_model_backtest_forecasts = load_primitive_model_forecasts
        combiners = load_combiner_models
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
    @pytest.fixture
    def load_train_and_test_series(self, load_combiner_backtest_forecaster):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        time_series_data = combiner_backtest_forecaster.time_series_data[
            combiner_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_window = combiner_backtest_forecaster._get_windows()[-1]
        train_series, test_series = combiner_backtest_forecaster._get_train_and_test_series(
            time_series_data,
            test_window
        )
        return train_series, test_series
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

    def test_get_fit_models(self, load_combiner_backtest_forecaster, load_train_and_test_series):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = combiner_backtest_forecaster._get_fit_models(
            train_series
        )
        assert all([isinstance(fit_model, AbstractCombinerModel) for fit_model in fit_models.values()])
        assert all([fit_model.is_fit for fit_model in fit_models.values()])

    def test_get_forecasts_all_models(self, load_combiner_backtest_forecaster, load_train_and_test_series):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = combiner_backtest_forecaster._get_fit_models(
            train_series
        )
        forecasts_all_models = combiner_backtest_forecaster._get_forecasts_all_models(fit_models, test_series)
        required_columns = [
            'predict_from', 'date_index', 'actuals',
            "AdaptiveHedge--p1_alpha:0.1--p3_multiplier:1",
            "FollowTheLeader"
        ]
        test_columns = list(forecasts_all_models.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required for test, required in test_required_columns])
        assert len(forecasts_all_models) == 4
        assert all(forecasts_all_models.notnull())
        assert False
