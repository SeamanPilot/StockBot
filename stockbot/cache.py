from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import fetch_ohlcv


CACHE_DIR = Path(os.environ.get("STOCKBOT_CACHE_DIR", Path(__file__).parent.parent / "stockbot_cache"))
DEFAULT_TTL = 3600  # 1 hour


def _cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}_{period}_{interval}.parquet"


def _meta_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}_{period}_{interval}.json"


def get_ohlcv(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    ttl: int = DEFAULT_TTL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data with local disk cache.

    Returns cached data if fresh (within ttl seconds).
    Falls back to yfinance on miss or stale data.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ck = CACHE_DIR / _cache_key(symbol, period, interval)
    mk = CACHE_DIR / _meta_key(symbol, period, interval)

    if not force_refresh and ck.exists() and mk.exists():
        meta = json.loads(mk.read_text(encoding="utf-8"))
        age = time.time() - meta.get("fetched_at", 0)
        if age < ttl:
            df = pd.read_parquet(ck)
            # Filter to requested period freshness
            return df

    # Cache miss or stale
    df = fetch_ohlcv(symbol, period=period, interval=interval)
    df.to_parquet(ck, index=True)
    meta = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "fetched_at": time.time(),
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
    }
    mk.write_text(json.dumps(meta), encoding="utf-8")
    return df


def clear_cache(symbol: Optional[str] = None, older_than: int = 0) -> int:
    """Remove cached data. If symbol given, only that symbol. If older_than, only entries older than N seconds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for f in list(CACHE_DIR.glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        if symbol and meta.get("symbol") != symbol:
            continue
        age = time.time() - meta.get("fetched_at", 0)
        if older_than and age < older_than:
            continue
        cache_file = CACHE_DIR / _cache_key(meta["symbol"], meta["period"], meta["interval"])
        cache_file.unlink(missing_ok=True)
        f.unlink()
        removed += 1
    return removed


def cache_status() -> list[dict]:
    """List all cache entries with metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(CACHE_DIR.glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        meta["age_hours"] = round((time.time() - meta.get("fetched_at", 0)) / 3600, 1)
        meta["size_kb"] = round((CACHE_DIR / _cache_key(meta["symbol"], meta["period"], meta["interval"])).stat().st_size / 1024, 1)
        entries.append(meta)
    return entries
