import pandas as pd
import configparser
import sys
import json
from datetime import datetime

from backtest_forecaster.forecasters.backtest_forecaster import PrimitiveModelBacktestForecaster
from backtest_forecaster.utils import model_loader, model_info_extrator


def main():
    config = configparser.ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)

    local_data_paths = config["local_data_paths"]
    custom_date_parser = lambda date: datetime.strptime(date, "%d/%m/%Y")

    time_series_data = pd.read_csv(
        filepath_or_buffer=local_data_paths["time_series_data_path"],
        index_col=0,
        parse_dates=["date_index"],
        date_parser=custom_date_parser
    )
    loader_config_path = config["loader_config_path"]
    with open(loader_config_path["primitive_model_config_path"], 'r') as j:
        primitive_model_config = json.loads(j.read())
    primitive_models = model_loader.load_models(primitive_model_config)
    backtest_forecaster = PrimitiveModelBacktestForecaster(
        time_series_data=time_series_data,
        models=primitive_models,
        min_train_window_len=1,
        max_train_window_len=24,
        max_windows=13,
    )
    all_primitive_models, primitive_model_forecasts = backtest_forecaster.get_backtest_models_and_forecasts()
    get_hyp_param_df = model_info_extrator.get_hyp_param_df(primitive_models)
    primitive_model_forecasts.to_csv(local_data_paths["primitive_model_backtest_forecasts_path"])
    get_hyp_param_df.to_csv("examples/example_data/example_hyp_param_df.csv")


if __name__ == '__main__':
    main()
