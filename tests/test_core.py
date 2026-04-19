"""Core module tests for stockbot."""
import pytest
import numpy as np
import pandas as pd

from stockbot.features import with_indicators
from stockbot.model import EnsemblePriceForecaster
from stockbot.strategies import (
    SmaCrossoverStrategy, RsiMeanReversionStrategy, BreakoutStrategy,
    MacdCrossoverStrategy, BollingerMeanReversionStrategy, MomentumStrategy,
    StochasticStrategy, default_strategy_stack,
)
from stockbot.risk import RiskConfig, RiskManager
from stockbot.brokers import SimulatedBroker, Position


def _sample_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=dates)


class TestFeatures:
    def test_with_indicators_returns_dataframe(self):
        df = _sample_df()
        result = with_indicators(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_all_feature_columns_present(self):
        df = _sample_df()
        result = with_indicators(df)
        for col in EnsemblePriceForecaster.feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_no_nan_in_features(self):
        df = _sample_df()
        result = with_indicators(df)
        feat_cols = EnsemblePriceForecaster.feature_columns()
        assert not result[feat_cols].isnull().any().any()


class TestModel:
    def test_fit_and_predict(self):
        df = _sample_df()
        feat = with_indicators(df)
        model = EnsemblePriceForecaster()
        model.fit(feat)
        pred = model.predict_next_return(feat)
        assert -0.2 <= pred <= 0.2

    def test_predict_detailed(self):
        df = _sample_df()
        feat = with_indicators(df)
        model = EnsemblePriceForecaster()
        model.fit(feat)
        result = model.predict_detailed(feat)
        assert "predictions" in result
        assert "weights" in result
        assert "ensemble" in result
        assert len(result["predictions"]) == 4

    def test_save_load(self, tmp_path):
        df = _sample_df()
        feat = with_indicators(df)
        model = EnsemblePriceForecaster()
        model.fit(feat)
        pred1 = model.predict_next_return(feat)
        model.save(tmp_path / "model")
        model2 = EnsemblePriceForecaster()
        model2.load(tmp_path / "model")
        pred2 = model2.predict_next_return(feat)
        assert abs(pred1 - pred2) < 1e-6

    def test_untrained_raises(self):
        model = EnsemblePriceForecaster()
        with pytest.raises(RuntimeError):
            model.predict_next_return(_sample_df())


class TestStrategies:
    @pytest.mark.parametrize("strategy_cls", [
        SmaCrossoverStrategy, RsiMeanReversionStrategy, BreakoutStrategy,
        MacdCrossoverStrategy, BollingerMeanReversionStrategy, MomentumStrategy,
        StochasticStrategy,
    ])
    def test_strategy_generates_signal(self, strategy_cls):
        df = _sample_df()
        feat = with_indicators(df)
        sig = strategy_cls().generate(feat)
        assert sig.signal in [-1, 0, 1]
        assert 0 <= sig.confidence <= 1.0

    def test_ensemble_generates_signal(self):
        df = _sample_df()
        feat = with_indicators(df)
        sig = default_strategy_stack().generate(feat)
        assert sig.signal in [-1, 0, 1]


class TestRisk:
    def test_stop_loss(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.05))
        assert rm.check_stop_loss("AAPL", 100, 94) is True
        assert rm.check_stop_loss("AAPL", 100, 96) is False

    def test_take_profit(self):
        rm = RiskManager(RiskConfig(take_profit_pct=0.10))
        assert rm.check_take_profit("AAPL", 100, 111) is True
        assert rm.check_take_profit("AAPL", 100, 108) is False

    def test_drawdown_halt(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.10))
        rm.state.peak_equity = 100000
        assert rm.check_drawdown(89000) is True
        assert rm.check_drawdown(95000) is False

    def test_position_limit(self):
        rm = RiskManager(RiskConfig(max_position_pct=0.25))
        # qty=100 shares * price=$300 = $30k position > 25% of $100k
        assert rm.check_position_limit("AAPL", 100, 300, 100000) is True
        # qty=100 * price=$200 = $20k position < 25% of $100k
        assert rm.check_position_limit("AAPL", 100, 200, 100000) is False

    def test_volatility_scaling(self):
        rm = RiskManager(RiskConfig(volatility_position_scale=True))
        scaled = rm.scale_position_by_volatility(1000, 0.005)
        assert scaled >= 1000
        scaled = rm.scale_position_by_volatility(1000, 0.05)
        assert scaled < 1000

    def test_evaluate_allows_normal_buy(self):
        rm = RiskManager()
        allowed, reason = rm.evaluate("AAPL", "buy", 10, 150, 100000, 0, 1000000, 0.02)
        assert allowed

    def test_evaluate_blocks_halted(self):
        rm = RiskManager()
        rm.state.halted = True
        rm.state.halt_reason = "Test halt"
        allowed, reason = rm.evaluate("AAPL", "buy", 10, 150, 100000, 0, 1000000, 0.02)
        assert not allowed


class TestBroker:
    def test_simulated_buy_sell(self):
        broker = SimulatedBroker(cash=10000)
        broker.buy("AAPL", 150, 10)
        assert broker.cash == 10000 - 150 * 10
        assert broker.positions["AAPL"].qty == 10
        broker.sell("AAPL", 160, 5)
        assert broker.positions["AAPL"].qty == 5
        assert broker.cash == 10000 - 150 * 10 + 160 * 5

    def test_insufficient_cash(self):
        broker = SimulatedBroker(cash=100)
        with pytest.raises(ValueError):
            broker.buy("AAPL", 150, 10)

    def test_insufficient_position(self):
        broker = SimulatedBroker(cash=10000)
        with pytest.raises(ValueError):
            broker.sell("AAPL", 150, 10)

    def test_equity(self):
        broker = SimulatedBroker(cash=10000)
        broker.buy("AAPL", 150, 10)
        equity = broker.equity({"AAPL": 160})
        assert equity == 10000 - 150 * 10 + 160 * 10

    def test_commission(self):
        broker = SimulatedBroker(cash=10000, commission_per_trade=1.0)
        broker.buy("AAPL", 150, 10)
        assert broker.cash == 10000 - 150 * 10 - 1.0
