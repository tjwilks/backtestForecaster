from backtest_forecaster.models.combiner_models import \
    AdaptiveHedge, FollowTheLeader
from tests.forecaster_test_fixtures import\
    CombinerBacktestForecasterTestFixtures


class TestAdaptiveHedge(CombinerBacktestForecasterTestFixtures):

    def test_fit(self, load_X_train_test):
        X_train, _ = load_X_train_test
        test_adaptive_hedge = AdaptiveHedge(alpha=0.8, multiplier=2)
        test_adaptive_hedge.fit(X_train)
        weights = test_adaptive_hedge.get_weights()
        assert int(sum(weights)) == 1,\
            "AdaptiveHedge method fit returned weights that do not sum to 1"
        assert len(weights) == 3, \
            "AdaptiveHedge method fit returned the incorrect number of weights"
        assert all([0 <= weight <= 1 for weight in weights]),\
            "AdaptiveHedge method fit returned weights outside of 0 to 1 range"
        assert test_adaptive_hedge.is_fit, "AdaptiveHedge method fit did " \
            "not set is_fit class variable to True"

    def test_predict(self, load_X_train_test):
        X_train, X_test = load_X_train_test
        test_adaptive_hedge = AdaptiveHedge(alpha=0.8, multiplier=2)
        test_adaptive_hedge.fit(X_train)
        test_prediction = test_adaptive_hedge.predict(X_test)
        assert len(test_prediction) == len(X_test), \
            "AdaptiveHedge predict method did not return prediction for " \
            "same horizon length as inputted X_test"
        required_prediction = [
            2264.78397732,
            2447.1898076,
            2738.8593603,
            2751.62836572
        ]
        test_required_prediction = zip(test_prediction, required_prediction)
        assert all([
            round(test, 3) == round(required, 3)
            for test, required
            in test_required_prediction
        ]), "AdaptiveHedge predict method did not return the correct " \
            "prediction"


class TestFollowTheLeader(CombinerBacktestForecasterTestFixtures):

    def test_fit(self, load_X_train_test):
        X_train, _ = load_X_train_test
        test_adaptive_hedge = FollowTheLeader()
        test_adaptive_hedge.fit(X_train)
        weights = test_adaptive_hedge.get_weights()
        assert int(sum(weights)) == 1, \
            "FollowTheLeader method fit returned weights that do not sum to 1"
        assert len(weights) == 3, \
            "FollowTheLeader method fit returned the incorrect number of " \
            "weights"
        assert all([0 <= weight <= 1 for weight in weights]), \
            "FollowTheLeader method fit returned weights outside of 0 to 1 " \
            "range"
        assert len([weight == 1 for weight in weights]), \
            "FollowTheLeader method fit returned weights with more than one " \
            "weight equal to 1"
        assert test_adaptive_hedge.is_fit, \
            "FollowTheLeader method fit did not set is_fit class " \
            "variable to True"

    def test_predict(self, load_X_train_test):
        X_train, X_test = load_X_train_test
        test_adaptive_hedge = FollowTheLeader()
        test_adaptive_hedge.fit(X_train)
        test_prediction = test_adaptive_hedge.predict(X_test)
        assert len(test_prediction) == len(X_test), \
            "FollowTheLeader predict method did not return prediction for " \
            "same horizon length as inputted X_test"
        required_prediction = [
            3140,
            3140,
            3140,
            3140
        ]
        test_required_prediction = zip(test_prediction, required_prediction)
        print(test_prediction)
        assert all([
            round(test, 3) == round(required, 3)
            for test, required
            in test_required_prediction
        ]), "FollowTheLeader predict method did not return the correct " \
            "prediction"
