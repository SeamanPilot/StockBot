from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("stockbot.scheduler")


class TradeScheduler:
    """Run trading cycles on a fixed interval in a background thread."""

    def __init__(
        self,
        callback: Callable[[], dict],
        interval_seconds: int = 3600,
        run_on_start: bool = True,
    ) -> None:
        self.callback = callback
        self.interval = interval_seconds
        self.run_on_start = run_on_start
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_result: Optional[dict] = None
        self._run_count = 0
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started (interval=%ds)", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def _loop(self) -> None:
        if self.run_on_start:
            self._run_cycle()
        while not self._stop_event.wait(timeout=self.interval):
            self._run_cycle()

    def _run_cycle(self) -> None:
        try:
            logger.info("Scheduler cycle started at %s", datetime.now(timezone.utc).isoformat())
            self._last_result = self.callback()
            self._run_count += 1
            logger.info("Scheduler cycle %d complete", self._run_count)
        except Exception as e:
            logger.error("Scheduler cycle failed: %s", e)
            self._last_result = {"error": str(e)}

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "interval": self.interval,
            "run_count": self._run_count,
            "last_result": self._last_result is not None,
        }
