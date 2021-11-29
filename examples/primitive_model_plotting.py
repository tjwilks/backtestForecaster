import pandas as pd
from backtest_forecaster.plotting.plotting import plot_forecast_vs_actuals, plot_primitive_model_error_over_time, plot_forecasts_over_time_aggregated
from backtest_forecaster.plotting.plotting import plot_model_weights_bar_plot

from datetime import datetime


def main():
    custom_date_parser = lambda date: datetime.strptime(date, "%d/%m/%Y")
    # time_series_data = pd.read_csv(
    #     filepath_or_buffer=(
    #             "examples/example_data/" +
    #             "example_time_series_data_all_1984_10_01_ids_3_ids.csv"
    #     ),
    #     index_col=0,
    #     parse_dates=["date_index"],
    #     date_parser=custom_date_parser
    # )
    # primitive_model_backtest_forecasts = pd.read_csv(
    #     filepath_or_buffer=(
    #             "examples/example_data/" +
    #             "example_primitive_model_backtest_forecasts_all_1984_10_01_ids_3_ids.csv"),
    #     index_col=0,
    #     parse_dates=["predict_from", "date_index"],
    #     date_parser=custom_date_parser
    # )
    # time_series_data = time_series_data[
    #     time_series_data["series_id"] == "N1681"
    # ]
    # primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
    #     primitive_model_backtest_forecasts["series_id"] == "N1681"
    # ]
    #
    # ########################################
    # for predict_from in primitive_model_backtest_forecasts["predict_from"].unique()[2:]:
    #     plot_forecast_vs_actuals(
    #         time_series_data,
    #         primitive_model_backtest_forecasts,
    #         3,
    #         predict_from
    #     )
    #
    # ########################################
    # plot_forecasts_over_time_aggregated(primitive_model_backtest_forecasts, 3)
    #
    # ########################################
    # primitive_model_backtest_error = primitive_model_backtest_forecasts.copy()
    # for model in primitive_model_backtest_error.drop(
    #     columns=["series_id", "predict_from", "date_index", "actuals"],
    #     axis=1).columns:
    #     primitive_model_backtest_error[model] = abs(
    #         primitive_model_backtest_error[model] -
    #         primitive_model_backtest_error["actuals"]
    #     )
    # primitive_model_backtest_error = primitive_model_backtest_error.drop(
    #     labels="actuals",
    #     axis=1
    # )
    # plot_primitive_model_error_over_time(primitive_model_backtest_error, 3)

    ########################################
    weights_data = pd.read_csv(
        filepath_or_buffer=(
                "examples/example_data/" +
                "example_primitive_model_weights.csv"
        ),
        index_col=0,
    )
    weights_data = weights_data[
        weights_data["series_id"] == "N1681"
    ]
    weights_data = weights_data[
        weights_data["predict_from"] == "1995-01-01"
        ]
    weights_data = weights_data[
        weights_data["combiner_model_name"] == "ForecastCombiner--p1_lmbda:0.1--p2_alpha:0.1--p3_multiplyer:1000000000--p4_regularization_type:ridge"
        ]
    plot_model_weights_bar_plot(weights_data)


if __name__ == "__main__":
    main()
