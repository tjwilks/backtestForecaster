import pandas as pd
import configparser
import sys
import json
from datetime import datetime

from backtest_forecaster.model_loaders.primitive_model_loader import load_models
from backtest_forecaster.forecasters.backtest_forecaster import CombinerBacktestForecaster


def main():
    config = configparser.ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)

    local_data_paths = config["local_data_paths"]
    custom_date_parser = lambda date: datetime.strptime(date, "%d/%m/%Y")
    primitive_model_backtest_forecasts = pd.read_csv(
        filepath_or_buffer=local_data_paths["primitive_model_backtest_forecasts_path"],
        index_col=0,
        parse_dates=["predict_from", "date_index"],
        date_parser=custom_date_parser
    )
    loader_config_path = config["loader_config_path"]
    with open(loader_config_path["combiner_loader_config_path"], 'r') as j:
        combiner_loader_config = json.loads(j.read())
    combiners = load_models(combiner_loader_config)
    combiner_backtest_forecaster = CombinerBacktestForecaster(
        forecast_data=primitive_model_backtest_forecasts,
        models=combiners,
        min_train_window_len=2,
        max_windows=2,
        horizon_length=1
    )
    all_fit_models, combiner_forecasts = combiner_backtest_forecaster.get_backtest_models_and_forecasts()
    primitive_model_weights = combiner_backtest_forecaster.get_primitive_model_weights_all_combiners(all_fit_models)
    print(primitive_model_weights)
    # get_hyp_param_df = combiner_backtest_forecaster.get_hyp_param_df()
    # combiner_forecasts.to_csv(local_data_paths["combiner_backtest_forecasts_path"])
    primitive_model_weights.to_csv("examples/example_data/example_primitive_model_weights.csv")
    # get_hyp_param_df.to_csv("examples/example_data/example_hyp_param_df.csv")


if __name__ == '__main__':
    main()
