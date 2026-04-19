from __future__ import annotations

import numpy as np
import pandas as pd


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["returns"] = out["close"].pct_change().fillna(0)

    # Moving averages
    out["sma_fast"] = out["close"].rolling(10).mean()
    out["sma_slow"] = out["close"].rolling(30).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()

    # RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["signal"]

    # Bollinger Bands
    out["bb_mid"] = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + (2 * std)
    out["bb_lower"] = out["bb_mid"] - (2 * std)
    out["bb_pct"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, 1e-9)

    # ATR (Average True Range)
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = true_range.rolling(14).mean()
    out["atr_pct"] = out["atr"] / out["close"].replace(0, 1e-9)

    # Stochastic Oscillator
    low_14 = out["low"].rolling(14).min()
    high_14 = out["high"].rolling(14).max()
    out["stoch_k"] = 100 * (out["close"] - low_14) / (high_14 - low_14).replace(0, 1e-9)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    # ADX (Average Directional Index)
    plus_dm = out["high"].diff().clip(lower=0)
    minus_dm = (-out["low"].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_14 = out["atr"]
    plus_di = 100 * plus_dm.rolling(14).mean() / atr_14.replace(0, 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr_14.replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    out["adx"] = dx.rolling(14).mean()

    # OBV (On-Balance Volume)
    direction = np.where(out["close"] > out["close"].shift(1), 1, np.where(out["close"] < out["close"].shift(1), -1, 0))
    out["obv"] = (direction * out["volume"]).cumsum()
    out["obv_sma"] = out["obv"].rolling(20).mean()

    # Williams %R
    high_14 = out["high"].rolling(14).max()
    low_14 = out["low"].rolling(14).min()
    out["willr"] = -100 * (high_14 - out["close"]) / (high_14 - low_14).replace(0, 1e-9)

    # Lagged returns (momentum context)
    out["return_1d"] = out["close"].pct_change(1).fillna(0)
    out["return_5d"] = out["close"].pct_change(5).fillna(0)
    out["return_10d"] = out["close"].pct_change(10).fillna(0)

    # Volatility
    out["volatility_10"] = out["returns"].rolling(10).std().fillna(0)
    out["volatility_20"] = out["returns"].rolling(20).std().fillna(0)

    return out.dropna()
