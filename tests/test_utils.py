from backtest_forecaster.utils import model_loader, model_info_extrator
from tests.forecaster_test_fixtures import BacktestForecasterTestFixtures, \
    PrimitiveModelBacktestForecasterTestFixtures, \
    CombinerBacktestForecasterTestFixtures

from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from backtest_forecaster.models.combiner_models import AbstractCombinerModel
import json


class TestModelLoader(BacktestForecasterTestFixtures):
    def test_load_models(self, load_config):
        config = load_config
        loader_config_path = config["loader_config_path"]

        def get_models(config_path):
            with open(loader_config_path[config_path], 'r') as j:
                model_loader_config = json.loads(j.read())
            models = model_loader.load_models(model_loader_config)
            return models
        primitive_models = get_models("primitive_model_config_path")
        combiners = get_models("combiner_loader_config_path")
        assert all([isinstance(model, AbstractPrimitiveModel)
               for model in primitive_models.values()]), "model_loader" \
            "module load_models returned primitive models that are not " \
            "instances of AbstractPrimitiveModel"
        assert all([isinstance(model, AbstractCombinerModel)
               for model in combiners.values()]), "model_loader" \
            "function load_models returned models that are not " \
            "instances of AbstractCombinerModel"


class TestModelInfoExtractor(
    PrimitiveModelBacktestForecasterTestFixtures,
    CombinerBacktestForecasterTestFixtures
    ):

    def test_get_primitive_model_weights_all_combiners(
            self,
            load_data,
            load_combiner_backtest_forecaster
    ):
        _, primitive_model_backtest_forecasts = load_data
        cmb_backtest_forecaster = load_combiner_backtest_forecaster
        combiners, _ = \
            cmb_backtest_forecaster.get_backtest_models_and_forecasts()
        primitive_model_names = primitive_model_backtest_forecasts.drop(
            labels=["series_id", "predict_from", "actuals"],
            axis=1
        ).columns
        primitive_model_weights = \
            model_info_extrator.get_primitive_model_weights_all_combiners(
                combiners,
                primitive_model_names
            )
        required_columns = ['date_index', 'Naive',
         'SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0',
         'combiner_model_name', 'predict_from', 'series_id']
        test_columns = list(primitive_model_weights.columns)
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required
                    for test, required
                    in test_required_columns]), \
            "model_info_extractor function " \
            "get_primitive_model_weights_all_combiners returned a dataframe " \
            "with incorrect columns"

        assert primitive_model_weights['predict_from'].nunique() == 6, \
            "model_info_extractor function " \
            "get_primitive_model_weights_all_combiners returned weights for" \
            "an incorrect number of forecasts"

    def test_get_hyp_param_df(self, load_config):
        config = load_config
        loader_config_path = config["loader_config_path"]
        def get_models(config_path):
            with open(loader_config_path[config_path], 'r') as j:
                model_loader_config = json.loads(j.read())
            models = model_loader.load_models(model_loader_config)
            return models
        primitive_models = get_models("primitive_model_config_path")
        pm_hyp_param_df = model_info_extrator.get_hyp_param_df(
            primitive_models)
        test_columns = pm_hyp_param_df.columns
        test_index = pm_hyp_param_df.index
        required_columns = [
            'SARIMA-auto_regressive',
            'SARIMA-integrated',
            'SARIMA-moving_average',
            'ExponentialSmoothing-trend_type'
        ]
        required_index = [
            'Naive',
            'SARIMA--p1_auto_regressive:1--p2_integrated:0--p3_moving_average:0',
            'SARIMA--p1_auto_regressive:2--p2_integrated:0--p3_moving_average:0',
            'ExponentialSmoothing--p1_trend_type:None'
        ]
        test_required_columns = zip(required_columns, test_columns)
        assert all([test == required
                    for test, required
                    in test_required_columns]), \
            "model_info_extractor function get_hyp_param_df returned a " \
            "dataframe with incorrect columns"
        test_required_index = zip(required_index, test_index)
        assert all([test == required
                    for test, required
                    in test_required_columns]), \
            "model_info_extractor function get_hyp_param_df returned a " \
            "dataframe with incorrect index values"
