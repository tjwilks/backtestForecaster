import pandas as pd
import re
from typing import Dict, List, Union
from weighted_forecast_combiner.forecast_combiner import OptimalForecastCombiner as ForecastCombiner
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel


def get_primitive_model_weights_all_combiners(
        fit_combiners: Dict[str,
                            Dict[str,
                                 Dict[str,
                                      Union[AbstractPrimitiveModel, ForecastCombiner]]]],
        primitive_model_names: List[str]
        ) -> pd.DataFrame:
    """
    Unpacks fit_combiners to dataset containing weights given to each primitive
    model by each combiner

    :param fit_combiners: combiner objects with weights data for all primitive models
    for all series id
    :param primitive_model_names: names of primitive model that combiner has
    assigned weights to
    :return primitive_model_weights: dataset of all primitive model weights
    for all time series for all combiners
    """

    weights_datasets = []
    index_n = 0
    for series_id, fit_combiners_all_windows in fit_combiners.items():
        for predict_from, fit_combiners in fit_combiners_all_windows.items():
            for combiner_model_name, combiner_model in fit_combiners.items():
                combiner_weights = combiner_model.get_weights()
                weights_data = {
                    primitive_model_name: model_weight
                    for primitive_model_name, model_weight
                    in zip(primitive_model_names, combiner_weights)
                }
                weights_data = pd.DataFrame(weights_data, index=[index_n])
                weights_data["combiner_model_name"] = combiner_model_name
                weights_data["predict_from"] = predict_from
                weights_data["series_id"] = series_id
                weights_datasets.append(weights_data)
                index_n += 1
    primitive_model_weights_all_combiners = pd.concat(weights_datasets)
    return primitive_model_weights_all_combiners


def get_hyp_param_df(
        models: Dict[str, List[str]]
        ) -> pd.DataFrame:
    """
    Unpacks model names contained as keys in models into dataset specifying
    hyper-parameters configuration of each model

    :param models: dictionary containing all primitive models and equivalent
    model names
    :return hyp_param_df: dataset specifying hyperparameter configurations
    of all models
    """
    model_names = models.keys()
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
