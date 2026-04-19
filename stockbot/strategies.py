from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class StrategySignal:
    name: str
    signal: int
    confidence: float


class BaseStrategy:
    name = "base"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError


class SmaCrossoverStrategy(BaseStrategy):
    name = "sma_crossover"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        sig = 1 if last["sma_fast"] > last["sma_slow"] else -1
        spread = abs(last["sma_fast"] - last["sma_slow"]) / max(last["close"], 1e-9)
        return StrategySignal(self.name, sig, float(min(1.0, spread * 50)))


class RsiMeanReversionStrategy(BaseStrategy):
    name = "rsi_mean_reversion"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 30:
            return StrategySignal(self.name, 1, (30 - rsi) / 30)
        if rsi > 70:
            return StrategySignal(self.name, -1, (rsi - 70) / 30)
        return StrategySignal(self.name, 0, 0.2)


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        highs = df["high"].tail(20)
        lows = df["low"].tail(20)
        if last["close"] >= highs.max():
            return StrategySignal(self.name, 1, 0.85)
        if last["close"] <= lows.min():
            return StrategySignal(self.name, -1, 0.85)
        return StrategySignal(self.name, 0, 0.3)


class MacdCrossoverStrategy(BaseStrategy):
    """Buy when MACD crosses above signal line, sell on cross below."""
    name = "macd_crossover"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 2:
            return StrategySignal(self.name, 0, 0.0)
        prev = df.iloc[-2]
        last = df.iloc[-1]
        prev_hist = prev["macd_hist"]
        last_hist = last["macd_hist"]
        if prev_hist <= 0 and last_hist > 0:
            strength = min(abs(last_hist) / max(last["close"], 1e-9) * 1000, 1.0)
            return StrategySignal(self.name, 1, max(0.5, strength))
        if prev_hist >= 0 and last_hist < 0:
            strength = min(abs(last_hist) / max(last["close"], 1e-9) * 1000, 1.0)
            return StrategySignal(self.name, -1, max(0.5, strength))
        return StrategySignal(self.name, 0, 0.2)


class BollingerMeanReversionStrategy(BaseStrategy):
    """Buy near lower band, sell near upper band."""
    name = "bb_mean_reversion"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        pct = float(df.iloc[-1]["bb_pct"])
        if pct < 0.1:
            return StrategySignal(self.name, 1, min(1.0, (0.1 - pct) * 5))
        if pct > 0.9:
            return StrategySignal(self.name, -1, min(1.0, (pct - 0.9) * 5))
        return StrategySignal(self.name, 0, 0.2)


class MomentumStrategy(BaseStrategy):
    """Follow short-term momentum using lagged returns."""
    name = "momentum"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        r1 = float(last.get("return_1d", 0))
        r5 = float(last.get("return_5d", 0))
        r10 = float(last.get("return_10d", 0))
        momentum = r1 * 0.5 + r5 * 0.3 + r10 * 0.2
        if momentum > 0.02:
            return StrategySignal(self.name, 1, min(1.0, momentum * 5))
        if momentum < -0.02:
            return StrategySignal(self.name, -1, min(1.0, abs(momentum) * 5))
        return StrategySignal(self.name, 0, 0.2)


class StochasticStrategy(BaseStrategy):
    """Buy on oversold Stochastic, sell on overbought."""
    name = "stochastic"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        k = float(last["stoch_k"])
        d = float(last["stoch_d"])
        if k < 20 and k > d:
            return StrategySignal(self.name, 1, min(1.0, (20 - k) / 20))
        if k > 80 and k < d:
            return StrategySignal(self.name, -1, min(1.0, (k - 80) / 20))
        return StrategySignal(self.name, 0, 0.2)


class EnsembleStrategy(BaseStrategy):
    name = "ensemble"

    def __init__(self, strategies: list[BaseStrategy]):
        self.strategies = strategies

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        signals = [s.generate(df) for s in self.strategies]
        weighted = sum(s.signal * s.confidence for s in signals)
        confidence = min(1.0, abs(weighted) / max(len(signals), 1))
        if weighted > 0.15:
            sig = 1
        elif weighted < -0.15:
            sig = -1
        else:
            sig = 0
        return StrategySignal(self.name, sig, confidence)


def default_strategy_stack() -> EnsembleStrategy:
    return EnsembleStrategy([
        SmaCrossoverStrategy(),
        RsiMeanReversionStrategy(),
        BreakoutStrategy(),
        MacdCrossoverStrategy(),
        BollingerMeanReversionStrategy(),
        MomentumStrategy(),
        StochasticStrategy(),
    ])
