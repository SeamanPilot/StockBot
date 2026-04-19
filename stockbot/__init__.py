"""StockBot — Ensemble ML paper trading with risk management and backtesting."""

from .trader import PaperTradingEngine
from .model import EnsemblePriceForecaster, DeepPriceForecaster
from .backtest import BacktestEngine, BacktestResult
from .risk import RiskConfig, RiskManager
from .strategies import default_strategy_stack

__all__ = [
    "PaperTradingEngine",
    "EnsemblePriceForecaster",
    "DeepPriceForecaster",
    "BacktestEngine",
    "BacktestResult",
    "RiskConfig",
    "RiskManager",
    "default_strategy_stack",
]
