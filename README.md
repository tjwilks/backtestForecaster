# BacktestForecaster 
BacktestForecaster is a python API for backtesting time-series forecasting 
models. It provides the ability for users to easily produce forecasts for a 
pre-specified set of models for multiple cross validation windows across 
multiple time series. The resulting forecasts can then be compared to real 
values to get a backtest of model errors over time.

BacktestForecaster currently supports both primitive models that train on 
original time-series values (e.g. ARIMA models) and combiner models that 
train on the errors of multiple primitive models to learn the optimal way to 
combine them together to produce a final stacked-ensemble forecast (e.g. 
AdaptiveHedge). 


### Step-by-step setup and run example scripts from command line 
1) Configure config files <br />
&nbsp; a) config/config.ini <br />
&nbsp; &nbsp; i) "loader_config_paths" - paths to json files that specify 
models for forecasts to be produced with. <br />
&nbsp; &nbsp; ii) "local_data_paths" - paths to original time series 
csv files and forecast csv files that forecasts will be both saved to and read 
from. <br />
&nbsp; b) config/primitive_model_config.json and 
config/combiner_model_config.json - files that specify models and 
hyperparameter grids to be used to generate set of models produce forecasts 
with.
2) Ensure package directory is working directory
3) Build a docker image of environment: `$docker build -t [IMAGE_TAG] .`
4) Run container based on docker image in interactive mode and access bash 
command line inside container: `docker run -it --mount "type=bind,source=$(pwd),target=/backtestForecaster" 
[IMAGE_TAG] bash`
4) Run example scripts: `python examples/primitive_model_backtest_forecasting.py config/config.ini`
or `python examples/combiner_model_backtest_forecasting.py config/config.ini`
if primitive model forecasts have already been generated. 

### Package development requirements
1) Documentation improvements - Docstring and inline comments
2) Error catching
3) TimeSeries and Forecast Classes



