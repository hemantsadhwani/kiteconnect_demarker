"""
Real-Time Market Sentiment Manager (v5).
Uses NiftySentimentAnalyzer (CPR + Type 2 bands, NCP state machine).
When cpr_today is provided by the workflow, reuses it to avoid duplicate CPR computation.
"""

import logging
import yaml
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Optional, List, Any

import pandas as pd

from .trading_sentiment_analyzer import NiftySentimentAnalyzer

logger = logging.getLogger(__name__)

MARKET_START_TIME = dt_time(9, 15)
NIFTY_TOKEN = 256265


def _prev_day_ohlc_from_cpr_today(cpr_today: Dict[str, Any]) -> Dict[str, float]:
    """Build synthetic previous-day OHLC from workflow cpr_today (P, R1, S1) so CPR levels match."""
    p = float(cpr_today["P"])
    r1 = float(cpr_today["R1"])
    s1 = float(cpr_today["S1"])
    # R1 = 2*P - L => L = 2*P - R1; S1 = 2*P - H => H = 2*P - S1
    high = 2 * p - s1
    low = 2 * p - r1
    return {"high": high, "low": low, "close": p}


class RealTimeMarketSentimentManager:
    """Real-time v5 sentiment: NiftySentimentAnalyzer, optional cpr_today from workflow."""

    def __init__(self, config_path: str, kite=None, cpr_today: Optional[Dict] = None):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.kite = kite
        self.config = self._load_config()
        self.cpr_today = cpr_today
        self.analyzer: Optional[NiftySentimentAnalyzer] = None
        self.current_date: Optional[datetime.date] = None
        self.is_initialized = False
        self._candle_history: List[Dict] = []
        self._last_sentiment: Optional[str] = None
        if cpr_today:
            self._init_from_cpr_today()
        logger.info("RealTimeMarketSentimentManager (v5) initialized (cpr_today=%s)", cpr_today is not None)

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _init_from_cpr_today(self) -> bool:
        """Initialize analyzer from workflow cpr_today (no Kite fetch)."""
        try:
            prev_day_ohlc = _prev_day_ohlc_from_cpr_today(self.cpr_today)
            self.analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
            self.current_date = datetime.now().date()
            self.is_initialized = True
            self._candle_history = []
            logger.info("v5 analyzer initialized from workflow cpr_today (no duplicate CPR computation)")
            return True
        except Exception as e:
            logger.error("Failed to init v5 from cpr_today: %s", e, exc_info=True)
            return False

    def _get_previous_day_ohlc(self, candle_date: datetime.date) -> tuple:
        """Fetch previous trading day OHLC from Kite. Returns (high, low, close) or (None, None, None)."""
        if not self.kite:
            return None, None, None
        prev = candle_date - timedelta(days=1)
        for _ in range(7):
            if prev.weekday() >= 5:
                prev -= timedelta(days=1)
                continue
            try:
                data = self.kite.historical_data(
                    instrument_token=NIFTY_TOKEN,
                    from_date=prev,
                    to_date=prev,
                    interval="day",
                )
                if data and len(data) > 0:
                    c = data[0]
                    return float(c["high"]), float(c["low"]), float(c["close"])
            except Exception as e:
                logger.debug("Kite OHLC fetch for %s: %s", prev, e)
            prev -= timedelta(days=1)
        return None, None, None

    def _ensure_initialized(self, candle_date: datetime.date, first_candle_ohlc: Optional[Dict] = None) -> bool:
        """Initialize analyzer if not yet (from cpr_today or Kite)."""
        if self.is_initialized and self.current_date == candle_date:
            return True
        if self.cpr_today and not self.analyzer:
            return self._init_from_cpr_today()
        if not self.cpr_today:
            high, low, close = self._get_previous_day_ohlc(candle_date)
            if high is None and first_candle_ohlc:
                high = float(first_candle_ohlc.get("high", 0)) + 100
                low = float(first_candle_ohlc.get("low", 0)) - 100
                close = float(first_candle_ohlc.get("close", 0))
            if high is not None and low is not None and close is not None:
                prev_day_ohlc = {"high": high, "low": low, "close": close}
                self.analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
                self.current_date = candle_date
                self.is_initialized = True
                self._candle_history = []
                logger.info("v5 analyzer initialized from Kite OHLC for %s", candle_date)
                return True
        return False

    def process_candle(self, ohlc: Dict, timestamp: datetime) -> Optional[str]:
        """Process a completed 1-minute NIFTY candle; return current sentiment."""
        candle_date = timestamp.date()
        if not self._ensure_initialized(candle_date, ohlc):
            logger.warning("v5: cannot init analyzer for %s", candle_date)
            return None
        if self.current_date and candle_date != self.current_date:
            self._candle_history = []
            self.current_date = candle_date
            if self.cpr_today:
                self._init_from_cpr_today()
            else:
                if not self._ensure_initialized(candle_date, ohlc):
                    return None
        row = {
            "open": float(ohlc["open"]),
            "high": float(ohlc["high"]),
            "low": float(ohlc["low"]),
            "close": float(ohlc["close"]),
            "date": timestamp,
        }
        self._candle_history.append(row)
        df = pd.DataFrame(self._candle_history)
        out = self.analyzer.apply_sentiment_logic(df)
        self._last_sentiment = str(out.iloc[-1]["market_sentiment"]).strip().upper()
        logger.info(
            "[%s] Market Sentiment (v5): %s | OHLC: O=%.2f H=%.2f L=%.2f C=%.2f",
            timestamp.strftime("%H:%M:%S"),
            self._last_sentiment,
            row["open"],
            row["high"],
            row["low"],
            row["close"],
        )
        return self._last_sentiment

    def get_current_sentiment(self) -> Optional[str]:
        if self._last_sentiment is not None:
            return self._last_sentiment
        return None

    def get_cpr_width_for_date(self, candle_date: datetime.date) -> Optional[float]:
        """CPR width |TC - BC| for the given date. Uses cpr_today when available."""
        if self.cpr_today and self.cpr_today.get("P") is not None:
            p = float(self.cpr_today["P"])
            r1 = float(self.cpr_today["R1"])
            s1 = float(self.cpr_today["S1"])
            high = 2 * p - s1
            low = 2 * p - r1
            close = p
            pivot = (high + low + close) / 3
            bc = (high + low) / 2
            tc = 2 * pivot - bc
            return abs(tc - bc)
        high, low, close = self._get_previous_day_ohlc(candle_date)
        if None in (high, low, close):
            return None
        pivot = (high + low + close) / 3
        bc = (high + low) / 2
        tc = 2 * pivot - bc
        return abs(tc - bc)
