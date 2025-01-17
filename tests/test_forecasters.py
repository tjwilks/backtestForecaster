from datetime import datetime
import numpy as np
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from backtest_forecaster.models.combiner_models import AbstractCombinerModel
from tests.forecaster_test_fixtures import \
    PrimitiveModelBacktestForecasterTestFixtures, \
    CombinerBacktestForecasterTestFixtures


class TestPrimitiveModelBacktestForecaster(
    PrimitiveModelBacktestForecasterTestFixtures
    ):

    def test_get_windows(
            self,
            load_primitive_model_backtest_forecaster
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        windows = pm_backtest_forecaster._get_windows()
        test_train_index_lengths = [len(window.train_index)
                                    for window in windows]
        required_train_index_lengths = ([120]*6) + list(range(119, 114, -1))
        test_required_train_index_lengths = zip(
            test_train_index_lengths, required_train_index_lengths
        )
        assert all([test == required
                    for test, required
                    in test_required_train_index_lengths]), \
            "PrimitiveModelBacktestForecaster method _get_windows did not " \
            "return the correct train_index length for all windows"
        test_test_index_lengths = [len(window.test_index)
                                   for window in windows]
        required_test_index_lengths = [1, 2, 3] + ([4] * 6)
        test_required_test_index_lengths = zip(
            test_test_index_lengths, required_test_index_lengths
        )
        assert all([test == required
                    for test, required
                    in test_required_test_index_lengths]), \
            "PrimitiveModelBacktestForecaster method _get_windows did not " \
            "return the correct test_index length for all windows"
        assert all([max(window.train_index) < min(window.test_index)
                    for window in windows]), \
            "PrimitiveModelBacktestForecaster method _get_windows returned " \
            "windows with overlapping train and test indexes" \


    def test_get_backtest_models_and_forecasts(
            self,
            load_primitive_model_backtest_forecaster
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        _, primitive_model_forecasts = \
            pm_backtest_forecaster.get_backtest_models_and_forecasts()
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
        assert all([test == required
                    for test, required
                    in test_required_columns]), \
            "PrimitiveModelBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned a dataframe with " \
            "incorrect columns"
        n_forecasts = primitive_model_forecasts["predict_from"].nunique()
        assert n_forecasts == 12, \
            "PrimitiveModelBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned an incorrect " \
            "number of forecasts"
        primitive_model_forecasts["horizon"] = round(
            (primitive_model_forecasts['date_index'] -
             primitive_model_forecasts['predict_from'])
            / np.timedelta64(1, 'M')
        ).astype(int) + 1
        max_horizon = max(primitive_model_forecasts["horizon"])
        assert max_horizon == 4, \
            "PrimitiveModelBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned forecasts with an " \
            "incorrect maximum horizon"
        assert (max(primitive_model_forecasts["predict_from"]) ==
               max(primitive_model_forecasts["date_index"])), \
            "PrimitiveModelBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned forecasts beyond " \
            "the specified time period"

    def test_get_model_forecasts_all_windows(
            self,
            load_primitive_model_backtest_forecaster
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        pm_backtest_forecaster.time_series_data = \
            pm_backtest_forecaster.time_series_data[
            pm_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_models, test_forecasts = \
            pm_backtest_forecaster.get_backtest_models_and_forecasts()
        test_model_predict_froms = [datetime.date(date)
                                    for date in test_models["N1681"].keys()]
        test_forecast_predict_froms = \
            test_forecasts["predict_from"].dt.date.unique().tolist()
        model_and_forecast_predict_froms = zip(
            test_model_predict_froms, test_forecast_predict_froms
        )
        assert all(
            [model_date == forecast_date
             for model_date, forecast_date
             in model_and_forecast_predict_froms]), \
            "PrimitiveModelBacktestForecaster method " \
            "get_model_forecasts_all_windows returned forecasts and models " \
            "with differing predict_from dates"
        all_test_models = [tuple(models.keys())
                           for models in test_models["N1681"].values()]
        required_models = (
            "Naive",
            "SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0",
            "SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0",
            "ExponentialSmoothing--p1_trend_type:None"
        )
        assert [test_models == required_models
                for test_models in all_test_models], \
            "PrimitiveModelBacktestForecaster method " \
            "get_model_forecasts_all_windows returned incorrect models"

    def test_get_fit_models(
            self,
            load_primitive_model_backtest_forecaster,
            load_train_and_test_series
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        train_series, _ = load_train_and_test_series
        fit_models = pm_backtest_forecaster._get_fit_models(
            train_series
        )
        assert all([isinstance(fit_model, AbstractPrimitiveModel)
                    for fit_model in fit_models.values()]),  \
            "PrimitiveModelBacktestForecaster method " \
            "_get_fit_models returned models that are not instances of the" \
            "AbstractPrimitiveModel class"

        assert all([fit_model.is_fit for fit_model in fit_models.values()]), \
            "PrimitiveModelBacktestForecaster method " \
            "_get_fit_models returned models that have not been fit"


    def test_get_forecasts_all_models(
            self,
            load_primitive_model_backtest_forecaster,
            load_train_and_test_series
    ):
        pm_backtest_forecaster = load_primitive_model_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = pm_backtest_forecaster._get_fit_models(
            train_series
        )
        forecasts_all_models = pm_backtest_forecaster._get_forecasts_all_models(
            fit_models,
            test_series
        )
        required_columns = [
            'predict_from', 'date_index', 'actuals', 'Naive',
            'SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0',
            'SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0',
            'ExponentialSmoothing--p1_trend_type:None'
        ]
        test_columns = list(forecasts_all_models.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required
                    for test, required in test_required_columns]),  \
            "PrimitiveModelBacktestForecaster method " \
            "_get_forecasts_all_models returned a dataframe with incorrect " \
            "columns"
        assert len(forecasts_all_models) == 4,  \
            "PrimitiveModelBacktestForecaster method " \
            "_get_forecasts_all_models returned a longer than " \
            "requested forecast"
        assert all(forecasts_all_models.notnull()),  \
            "PrimitiveModelBacktestForecaster method " \
            "_get_forecasts_all_models returned a dataframe with null values"


class TestCombinerBacktestForecaster(
    CombinerBacktestForecasterTestFixtures
    ):

    def test_get_windows(
            self,
            load_combiner_backtest_forecaster
    ):
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        windows = cmb_backtest_forecaster._get_windows()
        test_train_index_lengths = [len(window.train_index)
                                    for window in windows]
        required_train_index_lengths = list(range(12, 2, -1))
        test_required_train_index_lengths = zip(
            test_train_index_lengths, required_train_index_lengths
        )
        assert all([test == required
                    for test, required
                    in test_required_train_index_lengths]), \
            "CombinerBacktestForecaster method _get_windows did not " \
            "return the correct train_index length for all windows"
        test_test_index_lengths = [len(window.test_index)
                                   for window in windows]
        required_test_index_lengths = [1, 2, 3] + ([4] * 6)
        test_required_test_index_lengths = zip(
            test_test_index_lengths, required_test_index_lengths
        )
        assert all([test == required
                    for test, required
                    in test_required_test_index_lengths]), \
            "CombinerBacktestForecaster method _get_windows did not " \
            "return the correct test_index length for all windows"

        assert all([max(window.train_index) < min(window.test_index)
                    for window in windows]), \
            "CombinerBacktestForecaster method _get_windows returned " \
            "windows with overlapping train and test indexes" \


    def test_get_backtest_models_and_forecasts(
            self,
            load_combiner_backtest_forecaster
    ):
        combiner_backtest_forecaster = load_combiner_backtest_forecaster
        _, combiner_forecasts = \
            combiner_backtest_forecaster.get_backtest_models_and_forecasts()
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
        assert all([test == required
                    for test, required
                    in test_required_columns]), \
            "CombinerBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned a dataframe with " \
            "incorrect columns"
        n_forecasts = combiner_forecasts["predict_from"].nunique()
        assert n_forecasts == 6, \
            "CombinerBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned an incorrect " \
            "number of forecasts"
        combiner_forecasts["horizon"] = round(
            (combiner_forecasts['date_index'] -
             combiner_forecasts['predict_from'])
            / np.timedelta64(1, 'M')
        ).astype(int) + 1
        max_horizon = max(combiner_forecasts["horizon"])
        assert max_horizon == 4, \
            "CombinerBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned forecasts with an " \
            "incorrect maximum horizon"
        assert (max(combiner_forecasts["predict_from"]) ==
               max(combiner_forecasts["date_index"])), \
            "CombinerBacktestForecaster method " \
            "_get_backtest_models_and_forecasts returned forecasts beyond " \
            "the specified time period"

    def test_get_model_forecasts_all_windows(
            self,
            load_combiner_backtest_forecaster
    ):
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        cmb_backtest_forecaster.time_series_data = \
            cmb_backtest_forecaster.time_series_data[
            cmb_backtest_forecaster.time_series_data["series_id"] == "N1681"
            ]
        test_models, test_forecasts = \
            cmb_backtest_forecaster.get_backtest_models_and_forecasts()
        test_model_dates = [datetime.date(date)
                            for date in test_models["N1681"].keys()]
        test_forecast_dates = \
            test_forecasts["predict_from"].dt.date.unique().tolist()
        model_and_forecast_dates = zip(test_model_dates, test_forecast_dates)
        assert all([model_date == forecast_date
                    for model_date, forecast_date
                    in model_and_forecast_dates]), \
            "CombinerBacktestForecaster method " \
            "get_model_forecasts_all_windows returned forecasts and models " \
            "with differing predict_from dates"
        all_test_models = [tuple(models.keys())
                           for models
                           in test_models["N1681"].values()]
        required_models = (
            'AdaptiveHedge--p1_alpha:0.1--p3_multiplier:1', 'FollowTheLeader'
        )
        assert [test_models == required_models
                for test_models in all_test_models], \
            "CombinerBacktestForecaster method " \
            "get_model_forecasts_all_windows returned incorrect models"

    def test_get_fit_models(
            self,
            load_combiner_backtest_forecaster,
            load_train_and_test_series
    ):
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = cmb_backtest_forecaster._get_fit_models(
            train_series
        )
        assert all([isinstance(fit_model, AbstractCombinerModel)
                    for fit_model in fit_models.values()]),  \
            "CombinerBacktestForecaster method " \
            "_get_fit_models returned models that are not instances of the" \
            "AbstractPrimitiveModel class"
        assert all([fit_model.is_fit
                    for fit_model in fit_models.values()]), \
            "CombinerBacktestForecaster method " \
            "_get_fit_models returned models that have not been fit"

    def test_get_forecasts_all_models(
            self,
            load_combiner_backtest_forecaster,
            load_train_and_test_series
    ):
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        train_series, test_series = load_train_and_test_series
        fit_models = cmb_backtest_forecaster._get_fit_models(
            train_series
        )
        forecasts_all_models = \
            cmb_backtest_forecaster._get_forecasts_all_models(
                fit_models,
                test_series
            )
        required_columns = [
            'predict_from', 'date_index', 'actuals',
            "AdaptiveHedge--p1_alpha:0.1--p3_multiplier:1",
            "FollowTheLeader"
        ]
        test_columns = list(forecasts_all_models.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required
                    for test, required in test_required_columns]),  \
            "CombinerBacktestForecaster method " \
            "_get_forecasts_all_models returned a dataframe with incorrect " \
            "columns"
        assert len(forecasts_all_models) == 4,  \
            "CombinerBacktestForecaster method " \
            "_get_forecasts_all_models returned a longer than " \
            "requested forecast"
        assert all(forecasts_all_models.notnull()),  \
            "CombinerBacktestForecaster method " \
            "_get_forecasts_all_models returned a dataframe with null values"
