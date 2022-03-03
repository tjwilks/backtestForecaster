import pandas as pd
import re


def get_primitive_model_weights_all_combiners(fit_combiners, primitive_model_names):
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


def get_hyp_param_df(models):
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