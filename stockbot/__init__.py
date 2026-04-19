"""StockBot — Ensemble ML paper trading with risk management and backtesting."""

from .trader import PaperTradingEngine
from .model import EnsemblePriceForecaster, DeepPriceForecaster
from .backtest import BacktestEngine, BacktestResult
from .risk import RiskConfig, RiskManager
from .strategies import default_strategy_stack
from .cache import get_ohlcv, clear_cache, cache_status
from .alerts import AlertManager
from .scheduler import TradeScheduler
from .multi_tf import analyze_multi_timeframe
from .feature_importance import permutation_importance, gbr_feature_importance, rf_feature_importance

__all__ = [
    "PaperTradingEngine",
    "EnsemblePriceForecaster",
    "DeepPriceForecaster",
    "BacktestEngine",
    "BacktestResult",
    "RiskConfig",
    "RiskManager",
    "default_strategy_stack",
    "get_ohlcv",
    "clear_cache",
    "cache_status",
    "AlertManager",
    "TradeScheduler",
    "analyze_multi_timeframe",
    "permutation_importance",
    "gbr_feature_importance",
    "rf_feature_importance",
]
