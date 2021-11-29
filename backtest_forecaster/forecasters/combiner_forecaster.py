import pandas as pd
import numpy as np


def get_model_selection_forecasts_all_models(model_selection_models, train_series, test_series):
    """
    Fit models in model_list to trainset, predict on test set.

    :param train: training data
    :param test: test data
    :param seasonal_period: the seasonal period of the data
    ...
    :returns: predictions and actuals of backtest
    """
    # Check test set starts 1 point on from train set
    assert train_series['index'].max() + 1 == test_series['index'].min()
    X_train = train_series.drop(columns=['index', 'actuals', "series_id"])
    X_train = X_train.apply(pd.to_numeric)
    X_test = test_series.drop(columns=['index', 'actuals', "series_id"])
    X_test = X_test.apply(pd.to_numeric)
    y_train = train_series['actuals']
    y_test = test_series['actuals']
    combiners_output = []
    # Copy models first to ensure fitting doesn't happen multiple times
    for model_selection_model_name, combiner_model in model_selection_models.copy().items():
        combiner_model.fit(
            x=X_train,
            y=y_train,
            epochs=100
        )
        forecast = combiner_model.predict(x=X_test)
        combiner_forecast = pd.DataFrame({
            'index': test_series['index'].tolist()[0],
            'predict_from': test_series['index'].min(),
            'actuals': y_test.tolist()[0],
            'forecast': forecast
        }, index=[0])
        hyp_param_df = pd.DataFrame({
            'method': combiner_model.regularization_type,
            'shrinkage': combiner_model.lmbda,
            'smoothing': combiner_model.alpha,
            'multiplyer': combiner_model.multiplyer,
            'n_models_to_include': X_train.shape[1],
            'train_series_len':  X_train.shape[0]
        }, index=[0])
        model_names = X_train.columns.tolist()
        model_weights = combiner_model.omega.numpy().tolist()
        model_names_weights = {model_name: weight for model_name, weight in zip(model_names, model_weights)}
        model_weights_df = pd.DataFrame(model_names_weights, index=[0])
        combiner_output = pd.concat([combiner_forecast, hyp_param_df, model_weights_df], axis=1)
        combiners_output.append(combiner_output)
    combiners_output = pd.concat(combiners_output)
    return combiners_output
