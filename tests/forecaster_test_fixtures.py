import pytest
from pytest import fixture
import configparser
import pandas as pd
import json
from datetime import datetime
from backtest_forecaster.utils import model_loader
from backtest_forecaster.forecasters.backtest_forecaster import \
    PrimitiveModelBacktestForecaster, \
    CombinerBacktestForecaster



class BacktestForecasterTestFixtures:
    @pytest.fixture
    def load_config(self):
        config = configparser.ConfigParser()
        config.read("tests/test_resources/test_config.ini")
        return config

    @pytest.fixture
    def load_data(
            self,
            load_config
    ):
        config = load_config
        local_data_paths = config["local_data_paths"]
        custom_date_parser = lambda date: datetime.strptime(date, "%d/%m/%Y")
        time_series_data = pd.read_csv(
            filepath_or_buffer=local_data_paths["time_series_data_path"],
            index_col=0,
            parse_dates=["date_index"],
            date_parser=custom_date_parser
        )
        custom_date_parser = lambda date: datetime.strptime(date, "%Y-%m-%d")
        primitive_model_forecasts = pd.read_csv(
            filepath_or_buffer=local_data_paths[
                "primitive_model_backtest_forecasts_path"],
            index_col=0,
            parse_dates=["predict_from", "date_index"],
            date_parser=custom_date_parser
        )
        return time_series_data, primitive_model_forecasts

    @pytest.fixture
    def load_X_train_test(self, load_train_and_test_series):
        train_series, test_series = load_train_and_test_series
        prepare_series = lambda series: series.drop(
            columns=["predict_from", "date_index", "actuals",
                     "series_id"]).apply(pd.to_numeric)
        X_train = prepare_series(train_series)
        X_test = prepare_series(test_series)
        return X_train, X_test

    @pytest.fixture
    def load_models(
            self,
            load_config
    ):
        config = load_config
        loader_config_path = config["loader_config_path"]
        def get_models(config_path):
            with open(loader_config_path[config_path], 'r') as j:
                model_loader_config = json.loads(j.read())
            models = model_loader.load_models(model_loader_config)
            return models
        primitive_models = get_models("primitive_model_config_path")
        combiners = get_models("combiner_loader_config_path")
        return primitive_models, combiners
#
class PrimitiveModelBacktestForecasterTestFixtures(
    BacktestForecasterTestFixtures
    ):
    @pytest.fixture
    def load_primitive_model_backtest_forecaster(
            self,
            load_data,
            load_models
    ):
        time_series_data, _ = load_data
        primitive_models, _ = load_models
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
    def load_train_and_test_series(
            self,
            load_primitive_model_backtest_forecaster
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        time_series_data = pm_backtest_forecaster.time_series_data[
            pm_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_window = pm_backtest_forecaster._get_windows()[-1]
        train_series, test_series = \
            pm_backtest_forecaster._get_train_and_test_series(
                time_series_data,
                test_window
            )
        return train_series, test_series


class CombinerBacktestForecasterTestFixtures(
    BacktestForecasterTestFixtures
    ):
    @pytest.fixture
    def load_combiner_backtest_forecaster(
            self,
            load_data,
            load_models
    ):
        _, primitive_model_backtest_forecasts = load_data
        _, combiners = load_models
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
    def load_train_and_test_series(
            self,
            load_combiner_backtest_forecaster
    ):
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        time_series_data = cmb_backtest_forecaster.time_series_data[
            cmb_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_window = cmb_backtest_forecaster._get_windows()[-1]
        train_series, test_series = \
            cmb_backtest_forecaster._get_train_and_test_series(
                time_series_data,
                test_window
            )
        return train_series, test_series