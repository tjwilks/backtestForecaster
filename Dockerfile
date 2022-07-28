FROM python:3.8
COPY requirements.txt /backtest_forecaster/requirements.txt
COPY setup.py /backtest_forecaster/setup.py
WORKDIR /backtest_forecaster
RUN pip install -r requirements.txt
RUN python setup.py develop
COPY . /backtest_forecaster
