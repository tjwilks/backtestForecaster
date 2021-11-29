import numpy as np
import pandas as pd
import warnings


def get_primitive_model_forecasts_all_models(primitive_models, train_series, test_series, seasonal_period=12):
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
    horizon = len(test_series)
    primitive_model_forecasts = dict()
    primitive_model_forecasts['predict_from'] = test_series['index'].min()
    primitive_model_forecasts['index'] = test_series['index']
    primitive_model_forecasts['actuals'] = np.array(test_series['actuals'], dtype=np.float64)
    for primitive_model_name, primitive_model in primitive_models.copy().items():
        actuals_array = np.array(train_series['actuals'], dtype=np.float64)
        if len(actuals_array) < seasonal_period*2 and 'Season' in primitive_model_name:
            primitive_model_forecasts[primitive_model_name] = np.repeat([np.NaN], horizon)
            warnings.warn(f'Can not forecast seasonal primitive model {primitive_model_name}: ' +
                          f'Time series length less than two seasonal periods')
            continue
        try:
            primitive_model.fit(y=actuals_array)
            primitive_model_forecasts[primitive_model_name] = primitive_model.predict(h=horizon)
        except np.linalg.LinAlgError:
            primitive_model_forecasts[primitive_model_name] = np.repeat([np.NaN], horizon)
            warnings.warn(f'Forecasting failed for primitive model {primitive_model_name}')
    primitive_model_forecasts = pd.DataFrame(primitive_model_forecasts, index=test_series.index)
    return primitive_model_forecasts
