from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .model import EnsemblePriceForecaster
from .data import fetch_ohlcv
from .features import with_indicators
from .cache import get_ohlcv as cached_fetch


def permutation_importance(
    model: EnsemblePriceForecaster,
    df: pd.DataFrame,
    n_repeats: int = 10,
    n_samples: int = 200,
) -> pd.DataFrame:
    """Compute feature importance via permutation on trained ensemble model.

    Shuffles each feature column and measures impact on prediction.
    Higher score = more important feature.
    """
    if not model.trained:
        raise RuntimeError("Model must be trained first.")

    feat_cols = model.feature_columns()
    x = df[feat_cols].values
    sample_idx = np.random.choice(len(x), size=min(n_samples, len(x)), replace=False)
    x_sample = x[sample_idx]
    x_scaled = model.scaler.transform(x_sample)

    # Baseline predictions
    base_preds = {}
    for name, mdl in [("mlp", model.mlp), ("gbr", model.gbr), ("ridge", model.ridge), ("rf", model.rf)]:
        base_preds[name] = mdl.predict(x_scaled)

    baseline = sum(
        np.mean(np.abs(base_preds[k])) * model._weights[i]
        for i, k in enumerate(["mlp", "gbr", "ridge", "rf"])
    )

    results = {}
    for col_idx, col_name in enumerate(feat_cols):
        deltas = []
        for _ in range(n_repeats):
            x_permuted = x_sample.copy()
            np.random.shuffle(x_permuted[:, col_idx])
            x_perm_scaled = model.scaler.transform(x_permuted)

            perm_preds = {}
            for name, mdl in [("mlp", model.mlp), ("gbr", model.gbr), ("ridge", model.ridge), ("rf", model.rf)]:
                perm_preds[name] = mdl.predict(x_perm_scaled)

            perm_score = sum(
                np.mean(np.abs(perm_preds[k])) * model._weights[i]
                for i, k in enumerate(["mlp", "gbr", "ridge", "rf"])
            )
            deltas.append(abs(perm_score - baseline))

        results[col_name] = {
            "importance": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
        }

    result_df = pd.DataFrame(results).T.sort_values("importance", ascending=False)
    result_df.index.name = "feature"
    return result_df


def gbr_feature_importance(model: EnsemblePriceForecaster) -> pd.DataFrame:
    """Quick GBR native feature importance (no permutation needed)."""
    if not model.trained:
        raise RuntimeError("Model must be trained first.")
    feat_cols = model.feature_columns()
    importances = model.gbr.feature_importances_
    return pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).set_index("feature")


def rf_feature_importance(model: EnsemblePriceForecaster) -> pd.DataFrame:
    """Quick RF native feature importance."""
    if not model.trained:
        raise RuntimeError("Model must be trained first.")
    feat_cols = model.feature_columns()
    importances = model.rf.feature_importances_
    return pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).set_index("feature")
