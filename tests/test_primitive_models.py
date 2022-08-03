from backtest_forecaster.models.primitive_models import \
    Naive, ExponentialSmoothing, SARIMA
from tests.forecaster_test_fixtures import \
    PrimitiveModelBacktestForecasterTestFixtures


class TestNaive(PrimitiveModelBacktestForecasterTestFixtures):

    def test_fit(self, load_train_and_test_series):
        train_series, _ = load_train_and_test_series
        naive_model = Naive()
        naive_model.fit(y=train_series["actuals"])
        assert naive_model.is_fit, "Naive fit method did not set is_fit " \
           "class variable to True"
        assert naive_model.stored is not None, "Naive fit method did not set" \
           "'stored' class variable"

    def test_predict(self, load_train_and_test_series):
        train_series, test_series = load_train_and_test_series
        naive_model = Naive()
        naive_model.fit(y=train_series["actuals"])
        test_prediction = naive_model.predict(h=len(test_series))
        assert len(test_prediction) == len(test_series), \
            "Naive predict method did not return prediction for " \
            "same horizon length as inputted X_test"
        required_prediction = [
            3740,
            3740,
            3740,
            3740
        ]
        test_required_prediction = zip(test_prediction, required_prediction)
        assert all([
            round(test, 3) == round(required, 3)
            for test, required
            in test_required_prediction
        ]), "Naive predict method did not return the correct " \
            "prediction"


class TestExponentialSmoothing(PrimitiveModelBacktestForecasterTestFixtures):

    def test_fit(self, load_train_and_test_series):
        train_series, _ = load_train_and_test_series
        exps_model = ExponentialSmoothing()
        exps_model.fit(y=train_series["actuals"])
        assert exps_model.is_fit, "ExponentialSmoothing fit method did not " \
          "set is_fit class variable to True"
        assert exps_model.model is not None, "ExponentialSmoothing fit " \
         "method did not set 'model' class variable"

    def test_predict(self, load_train_and_test_series):
        train_series, test_series = load_train_and_test_series
        exps_model = ExponentialSmoothing()
        exps_model.fit(y=train_series["actuals"])
        test_prediction = exps_model.predict(h=len(test_series))
        assert len(test_prediction) == len(test_series), \
            "ExponentialSmoothing predict method did not return prediction " \
            "for same horizon length as inputted X_test"
        required_prediction = [
            3140.924483,
            3140.924483,
            3140.924483,
            3140.924483,
        ]
        test_required_prediction = zip(test_prediction, required_prediction)
        assert all([
            round(test, 3) == round(required, 3)
            for test, required
            in test_required_prediction
        ]), "ExponentialSmoothing predict method did not return the correct " \
            "prediction"


class TestSARIMA(PrimitiveModelBacktestForecasterTestFixtures):

    def test_fit(self, load_train_and_test_series):
        train_series, _ = load_train_and_test_series
        arima_model = SARIMA()
        arima_model.fit(y=train_series["actuals"])
        assert arima_model.is_fit, "SARIMA fit method did not set is_fit " \
            "class variable to True"
        assert arima_model.model is not None, "SARIMA fit method did not " \
            "set 'model' class variable"

    def test_predict(self, load_train_and_test_series):
        train_series, test_series = load_train_and_test_series
        arima_model = SARIMA()
        arima_model.fit(y=train_series["actuals"])
        test_prediction = arima_model.predict(h=len(test_series))
        assert len(test_prediction) == len(test_series), \
            "SARIMA predict method did not return prediction " \
            "for same horizon length as inputted X_test"
        required_prediction = [
            3484.930531,
            3189.554856,
            3138.691939,
            3019.272565
        ]
        test_required_prediction = zip(test_prediction, required_prediction)
        assert all([
            round(test, 3) == round(required, 3)
            for test, required
            in test_required_prediction
        ]), "SARIMA predict method did not return the correct " \
            "prediction"
