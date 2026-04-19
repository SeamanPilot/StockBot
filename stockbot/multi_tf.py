from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .cache import get_ohlcv
from .features import with_indicators
from .model import EnsemblePriceForecaster
from .strategies import default_strategy_stack, EnsembleStrategy


TIMEFRAMES = ['1h', '1d', '1wk']


@dataclass
class TimeframeSignal:
    timeframe: str
    predicted_return: float
    strategy_signal: int
    strategy_confidence: float
    combined_score: float
    volatility: float
    model_predictions: dict


@dataclass
class MultiTimeframeResult:
    symbol: str
    signals: dict
    aggregate_score: float
    aggregate_action: str
    dominant_timeframe: str


def analyze_multi_timeframe(
    symbol: str,
    model: EnsemblePriceForecaster,
    strategy: Optional[EnsembleStrategy] = None,
    timeframes: Optional[list] = None,
):
    strategy = strategy or default_strategy_stack()
    timeframes = timeframes or TIMEFRAMES
    tf_weights = {'1h': 0.2, '1d': 0.5, '1wk': 0.8}
    signals = {}
    periods = {'1h': '1mo', '1d': '6mo', '1wk': '2y'}

    for tf in timeframes:
        try:
            df = get_ohlcv(symbol, period=periods.get(tf, '6mo'), interval=tf)
            feat = with_indicators(df)
            pred = model.predict_next_return(feat)
            detailed = model.predict_detailed(feat)
            strat = strategy.generate(feat)
            weights = model.model_weights
            max_w = max(weights.values())
            combined = strat.signal * strat.confidence + pred * 6 * max_w
            vol = float(feat.iloc[-1].get('volatility_20', 0.02))
            signals[tf] = TimeframeSignal(
                timeframe=tf, predicted_return=pred,
                strategy_signal=strat.signal, strategy_confidence=strat.confidence,
                combined_score=combined, volatility=vol, model_predictions=detailed['predictions'],
            )
        except Exception:
            continue

    if not signals:
        raise ValueError(f'No timeframe data available for {symbol}')

    total_weight = sum(tf_weights.get(tf, 0.3) for tf in signals)
    aggregate_score = sum(
        signals[tf].combined_score * tf_weights.get(tf, 0.3) for tf in signals
    ) / total_weight

    if aggregate_score > 0.2:
        action = 'buy'
    elif aggregate_score < -0.2:
        action = 'sell'
    else:
        action = 'hold'

    dominant = max(signals, key=lambda tf: abs(signals[tf].combined_score))

    return MultiTimeframeResult(
        symbol=symbol, signals=signals, aggregate_score=aggregate_score,
        aggregate_action=action, dominant_timeframe=dominant,
    )

