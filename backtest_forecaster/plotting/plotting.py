import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import DateOffset


def plot_forecast_vs_actuals(time_series_data,
                             primitive_model_backtest_forecasts,
                             forecast_horizon,
                             date_to_plot=None,
                             max_months_to_plot=24
                             ):
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["predict_from"] == date_to_plot
    ]
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts.melt(
        id_vars=["series_id", "predict_from", "date_index"],
        var_name="primitive_model",
        value_name="forecast"
    )
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        (primitive_model_backtest_forecasts["date_index"] <
         primitive_model_backtest_forecasts["predict_from"] + DateOffset(months=forecast_horizon)
         )
    ]
    time_series_data = time_series_data[
        time_series_data["date_index"] < date_to_plot
        ]
    first_date_to_plot_from = time_series_data["date_index"].max() + DateOffset(months=-max_months_to_plot)
    time_series_data = time_series_data[
        time_series_data["date_index"] > first_date_to_plot_from
    ]
    datasets_to_insert = []
    for primitive_model in primitive_model_backtest_forecasts["primitive_model"].unique():
        data_to_insert = time_series_data[
            time_series_data["date_index"] == time_series_data["date_index"].max()
        ]
        data_to_insert["predict_from"] = date_to_plot
        data_to_insert["primitive_model"] = primitive_model
        data_to_insert = data_to_insert.rename({"actuals": "forecast"}, axis=1)
        datasets_to_insert.append(data_to_insert)
    datasets_to_insert = pd.concat(datasets_to_insert)
    primitive_model_backtest_forecasts = pd.concat([primitive_model_backtest_forecasts, datasets_to_insert])
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts.sort_values(
        by=["series_id", "primitive_model", "predict_from", "date_index"]
    )
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts.reset_index(drop=True)
    primitive_model_backtest_actuals = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["primitive_model"] == "actuals"
        ]
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["primitive_model"] != "actuals"
        ]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set(rc={"lines.linewidth": 1.5})
    sns.lineplot(
        x="date_index",
        y="actuals",
        color="black",
        data=time_series_data
    )
    sns.lineplot(
        x="date_index",
        y="forecast",
        linestyle="dashed",
        color="black",
        data=primitive_model_backtest_actuals
    )
    sns.lineplot(
        x="date_index",
        y="forecast",
        hue="primitive_model",
        data=primitive_model_backtest_forecasts
    )
    ax.axvline(pd.to_datetime(date_to_plot) + DateOffset(months=-1), ls='--', c='red')
    # plt.legend(loc=None)
    plt.xlabel("Time (Months)")
    plt.ylabel("Actuals and Forecast ($M)")
    plt.title('Actuals vs Forecast Over Time')
    date = pd.to_datetime(str(date_to_plot)).strftime('%Y.%m.%d')
    plot_path = f"examples/example_data/plots/actuals_vs_forecast_plot_{date}.png"
    plt.savefig(plot_path)


def plot_forecasts_over_time_aggregated(primitive_model_backtest_forecasts,
                                         forecast_horizon):
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts.melt(
        id_vars=["series_id", "predict_from", "date_index"],
        var_name="primitive_model",
        value_name="forecast"
    )
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        (primitive_model_backtest_forecasts["date_index"] <
         primitive_model_backtest_forecasts["predict_from"] + DateOffset(months=forecast_horizon)
         )
    ]
    predict_froms_to_keep = primitive_model_backtest_forecasts["predict_from"].unique()[:(-forecast_horizon + 1)]
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["predict_from"].isin(predict_froms_to_keep)
    ]

    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts.drop(
        labels="date_index",
        axis=1
    ).groupby(by=["series_id", "predict_from", "primitive_model"], as_index=False).sum()
    primitive_model_backtest_actuals = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["primitive_model"] == "actuals"
        ]
    primitive_model_backtest_forecasts = primitive_model_backtest_forecasts[
        primitive_model_backtest_forecasts["primitive_model"] != "actuals"
        ]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x="predict_from",
        y="forecast",
        color="black",
        data=primitive_model_backtest_actuals
    )
    sns.lineplot(
        x="predict_from",
        y="forecast",
        hue="primitive_model",
        data=primitive_model_backtest_forecasts
    )
    plt.xlabel("Time (Months)")
    plt.ylabel("Aggregated Actuals and Forecasts ($M)")
    plt.title("Forecast Error Over Time")
    plot_path = f"examples/example_data/plots/forecasts_over_time_aggregated.png"
    plt.savefig(plot_path)


def plot_primitive_model_error_over_time(primitive_model_backtest_error,
                                         forecast_horizon):
    primitive_model_backtest_error = primitive_model_backtest_error.melt(
        id_vars=["series_id", "predict_from", "date_index"],
        var_name="primitive_model",
        value_name="error"
    )
    primitive_model_backtest_error = primitive_model_backtest_error[
        (primitive_model_backtest_error["date_index"] <
         primitive_model_backtest_error["predict_from"] + DateOffset(months=forecast_horizon)
         )
    ]
    predict_froms_to_keep = primitive_model_backtest_error["predict_from"].unique()[:(-forecast_horizon+1)]
    primitive_model_backtest_error = primitive_model_backtest_error[
        primitive_model_backtest_error["predict_from"].isin(predict_froms_to_keep)
    ]
    primitive_model_backtest_error = primitive_model_backtest_error.drop(
        labels="date_index",
        axis=1
    ).groupby(by=["series_id", "predict_from", "primitive_model"], as_index=False).sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x="predict_from",
        y="error",
        hue="primitive_model",
        data=primitive_model_backtest_error
    )
    # plt.legend(loc='upper left')
    plt.xlabel("Time (Months)")
    plt.ylabel("Forecast Error ($M)")
    plt.title("Forecast Error Over Time")
    plot_path = f"examples/example_data/plots/error_over_time_plot.png"
    plt.savefig(plot_path)


def plot_model_weights_bar_plot(weights_data):
    weights_data = weights_data.melt(
        id_vars=["series_id", "predict_from", "combiner_model_name"],
        var_name="primitive_model",
        value_name="weight"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="primitive_model",
        y="weight",
        color="royalblue",
        data=weights_data
    )
    plt.xlabel("Model")
    plt.ylabel("Weight")
    plt.title("Model Weights For Forecast Window")
    plot_path = f"examples/example_data/plots/model_weights_plot.png"
    plt.savefig(plot_path)

