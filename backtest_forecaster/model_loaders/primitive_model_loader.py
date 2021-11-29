from sklearn.model_selection import ParameterGrid
from backtest_forecaster.models.primitive_models import Naive, SARIMA, ExponentialSmoothing
from weighted_forecast_combiner.forecast_combiner import OptimalForecastCombiner as ForecastCombiner
import inspect
import re


def load_models(model_config):
    all_models = dict()
    for model_name, grids in model_config.items():
        for grid in grids.values():
            model_grid = list(ParameterGrid(grid))
            models = {
                get_model_name(model_name, model_config):
                get_model(model_name, model_config)
                for model_config in model_grid
            }
            all_models.update(models)
    return all_models


def get_model_name(model_name, model_config):
    for key in model_config.keys():
        model_name += f"--{key}:{model_config[key]}"
    return model_name


def get_model(model_name, model_config):
    model_obj = globals()[model_name]
    model_args = model_config.values()
    model_arg_names = [re.sub(r"^p[0-9]{1}_", "", arg_name)
                       for arg_name in list(model_config.keys())]
    check_arguments(
        provided_arguments=model_arg_names,
        model=model_obj
    )
    model = model_obj(*model_args)
    return model


def check_arguments(provided_arguments, model):
    model_init_arguments = list(inspect.signature(model.__init__).parameters.keys())
    required_model_init_arguments = model_init_arguments[1:len(provided_arguments) + 1]
    assert required_model_init_arguments == provided_arguments, (
        f"arguments provided do not match those in model init method"
        f"\nrequired arguments: {required_model_init_arguments}"
        f"\nprovided arguments: {provided_arguments}"
    )
