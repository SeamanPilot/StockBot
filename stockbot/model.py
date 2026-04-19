from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


class EnsemblePriceForecaster:
    """Ensemble price forecaster combining MLP, Gradient Boosting, Ridge, and RandomForest.

    Each sub-model predicts next-close return independently.
    The final prediction is a weighted average that adapts based on
    walk-forward validation performance of each model.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()

        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )

        self.gbr = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )

        self.ridge = Ridge(alpha=1.0)

        self.rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        self.trained = False
        self._weights = np.array([0.3, 0.3, 0.15, 0.25])  # MLP, GBR, Ridge, RF
        self._validation_scores: dict[str, float] = {}

    @staticmethod
    def feature_columns() -> list[str]:
        return [
            # Price / returns
            "returns",
            "return_1d",
            "return_5d",
            "return_10d",
            # Trend
            "sma_fast",
            "sma_slow",
            "ema_12",
            "ema_26",
            "macd",
            "signal",
            "macd_hist",
            "adx",
            # Momentum
            "rsi",
            "stoch_k",
            "stoch_d",
            "willr",
            # Volatility
            "atr_pct",
            "volatility_10",
            "volatility_20",
            "bb_pct",
            # Volume
            "volume",
            "obv",
            "obv_sma",
            # Price level
            "close",
        ]

    def _walk_forward_validate(self, x: np.ndarray, y: np.ndarray, n_splits: int = 3) -> dict[str, float]:
        """Time-series walk-forward cross-validation to score each model."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {"mlp": [], "gbr": [], "ridge": [], "rf": []}
        models = {"mlp": self.mlp, "gbr": self.gbr, "ridge": self.ridge, "rf": self.rf}

        for train_idx, test_idx in tscv.split(x):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            sc = StandardScaler().fit(x_train)
            x_train_s = sc.transform(x_train)
            x_test_s = sc.transform(x_test)

            for name, model in models.items():
                try:
                    clone = type(model)(**model.get_params())
                    clone.fit(x_train_s, y_train)
                    pred = clone.predict(x_test_s)
                    scores[name].append(max(r2_score(y_test, pred), 0.01))
                except Exception:
                    scores[name].append(0.01)

        return {name: float(np.mean(v)) for name, v in scores.items()}

    def fit(self, df: pd.DataFrame) -> None:
        feat_cols = self.feature_columns()
        x = df[feat_cols].values[:-1]
        y = (df["close"].shift(-1) / df["close"] - 1).values[:-1]

        x_scaled = self.scaler.fit_transform(x)

        # Walk-forward validation for adaptive weighting
        self._validation_scores = self._walk_forward_validate(x_scaled, y)
        total = sum(self._validation_scores.values())
        if total > 0:
            self._weights = np.array([
                self._validation_scores["mlp"],
                self._validation_scores["gbr"],
                self._validation_scores["ridge"],
                self._validation_scores["rf"],
            ]) / total

        # Train on full dataset
        self.mlp.fit(x_scaled, y)
        self.gbr.fit(x_scaled, y)
        self.ridge.fit(x_scaled, y)
        self.rf.fit(x_scaled, y)
        self.trained = True

    def predict_next_return(self, df: pd.DataFrame) -> float:
        if not self.trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        row = df[self.feature_columns()].tail(1).values
        row_scaled = self.scaler.transform(row)

        mlp_pred = self.mlp.predict(row_scaled)[0]
        gbr_pred = self.gbr.predict(row_scaled)[0]
        ridge_pred = self.ridge.predict(row_scaled)[0]
        rf_pred = self.rf.predict(row_scaled)[0]

        preds = np.array([mlp_pred, gbr_pred, ridge_pred, rf_pred])
        ensemble_pred = float(np.dot(self._weights, preds))
        return float(np.clip(ensemble_pred, -0.2, 0.2))

    def predict_detailed(self, df: pd.DataFrame) -> dict:
        """Return individual model predictions alongside ensemble."""
        if not self.trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        row = df[self.feature_columns()].tail(1).values
        row_scaled = self.scaler.transform(row)

        preds = {
            "mlp": float(self.mlp.predict(row_scaled)[0]),
            "gbr": float(self.gbr.predict(row_scaled)[0]),
            "ridge": float(self.ridge.predict(row_scaled)[0]),
            "rf": float(self.rf.predict(row_scaled)[0]),
        }
        weights = {"mlp": float(self._weights[0]), "gbr": float(self._weights[1]),
                   "ridge": float(self._weights[2]), "rf": float(self._weights[3])}
        ensemble = float(np.clip(sum(preds[k] * weights[k] for k in preds), -0.2, 0.2))
        return {"predictions": preds, "weights": weights, "ensemble": ensemble}

    @property
    def model_weights(self) -> dict[str, float]:
        return {
            "mlp": float(self._weights[0]),
            "gbr": float(self._weights[1]),
            "ridge": float(self._weights[2]),
            "rf": float(self._weights[3]),
        }

    def save(self, path: str | Path) -> None:
        """Save trained model, scaler, and weights to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "scaler": pickle.dumps(self.scaler),
            "mlp": pickle.dumps(self.mlp),
            "gbr": pickle.dumps(self.gbr),
            "ridge": pickle.dumps(self.ridge),
            "rf": pickle.dumps(self.rf),
            "weights": self._weights.tolist(),
            "validation_scores": self._validation_scores,
        }
        (path / "model.pkl").write_bytes(pickle.dumps(data))

    def load(self, path: str | Path) -> None:
        """Load trained model, scaler, and weights from disk."""
        path = Path(path)
        data = pickle.loads((path / "model.pkl").read_bytes())
        self.scaler = pickle.loads(data["scaler"])
        self.mlp = pickle.loads(data["mlp"])
        self.gbr = pickle.loads(data["gbr"])
        self.ridge = pickle.loads(data["ridge"])
        self.rf = pickle.loads(data["rf"])
        self._weights = np.array(data["weights"])
        self._validation_scores = data.get("validation_scores", {})
        self.trained = True


# Backward-compatible alias
DeepPriceForecaster = EnsemblePriceForecaster
