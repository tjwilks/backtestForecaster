from sklearn.model_selection import ParameterGrid
from backtest_forecaster.models.primitive_models import Naive, SARIMA, ExponentialSmoothing
from weighted_forecast_combiner.forecast_combiner import OptimalForecastCombiner as ForecastCombiner
import inspect
import re


def load_models(model_config_grids):
    """
    Unpacks model hyper-parameter grids specified in model_config
    into dictionary of all configs

    :param model_config_grids: dictionary of multiple model hyper-parameter grids
    :returns all_models: all configs specified by all hyper-paramter
    grids in model_config
    """
    all_models = dict()
    for model_name, grids in model_config_grids.items():
        for grid in grids.values():
            all_model_configs = list(ParameterGrid(grid))
            models = {
                get_model_name(model_name, model_config):
                get_model(model_name, model_config)
                for model_config in all_model_configs
            }
            all_models.update(models)
    return all_models


def get_model_name(model_name, model_config):
    """
    Appends model config information to model name to return
    a new model name

    :param model_name: base model name without hyper-parameter
    information
    :param model_config: hyper-parameter information of model
    :returns model_name: new model name containing hyper-parameter
    information
    """
    for key in model_config.keys():
        model_name += f"--{key}:{model_config[key]}"
    return model_name


def get_model(model_name, model_config):
    """
    Initialises model with model name using arguments
    specified as values in model_config

    :param model_name: base model name without hyper-parameter
    information
    :param model_config: hyper-parameter information of model
    :returns model: initialised model with name specified by
    model_name and hyper-parameter config specified by model_config
    """
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
    """
    Checks arguments are the same as those required by model

    :param provided_arguments: intended hyper-parameter configuration of
    model
    :model model: initialised model with hyper-parameter configuration
    that is meant to be equivalent to provided arguments
    """
    model_init_arguments = list(inspect.signature(model.__init__).parameters.keys())
    required_model_init_arguments = model_init_arguments[1:len(provided_arguments) + 1]
    assert required_model_init_arguments == provided_arguments, (
        f"arguments provided do not match those in model init method"
        f"\nrequired arguments: {required_model_init_arguments}"
        f"\nprovided arguments: {provided_arguments}"
    )
