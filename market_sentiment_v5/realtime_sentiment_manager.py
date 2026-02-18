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

    def _backfill_candle_history(self, candle_date: datetime.date, first_live_timestamp: datetime) -> None:
        """
        Backfill _candle_history from market open to (first_live_timestamp - 1 min).
        Ensures production state matches test_realtime_sentiment (same prior candles => same sentiment).
        """
        if not self.kite or not self.analyzer:
            return
        from_dt = datetime.combine(candle_date, MARKET_START_TIME)
        to_dt = first_live_timestamp - timedelta(minutes=1)
        if to_dt < from_dt:
            return
        if self.config.get("backfill_from_market_open") is False:
            return
        data: List[Dict] = []
        chunk_hours = 2
        chunk_delta = timedelta(hours=chunk_hours)
        current_start = from_dt
        while current_start <= to_dt:
            current_end = min(current_start + chunk_delta, to_dt)
            try:
                chunk = self.kite.historical_data(
                    instrument_token=NIFTY_TOKEN,
                    from_date=current_start,
                    to_date=current_end,
                    interval="minute",
                )
                if chunk:
                    data.extend(chunk)
            except Exception as e:
                logger.warning("v5 backfill: Kite historical_data failed %s to %s: %s", current_start, current_end, e)
            current_start = current_end + timedelta(minutes=1)
            if current_start <= to_dt:
                import time
                time.sleep(0.3)
        if not data:
            return
        def _parse_date(d):
            dt = d["date"]
            if isinstance(dt, str):
                try:
                    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                    except Exception:
                        return None
            if hasattr(dt, "to_pydatetime"):
                dt = dt.to_pydatetime()
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            return dt

        data_sorted = sorted(data, key=lambda x: x["date"])
        for c in data_sorted:
            dt = _parse_date(c)
            if dt is None:
                continue
            self._candle_history.append({
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "date": dt,
            })
        logger.info(
            "v5 backfill: loaded %d candles from %s to %s (so sentiment matches test/full history)",
            len(self._candle_history),
            from_dt.strftime("%H:%M"),
            to_dt.strftime("%H:%M"),
        )

    def process_candle(self, ohlc: Dict, timestamp: datetime) -> Optional[str]:
        """Process a completed 1-minute NIFTY candle; return current sentiment."""
        candle_date = timestamp.date()
        if not self._ensure_initialized(candle_date, ohlc):
            logger.warning(
                "v5: cannot init analyzer for %s (cpr_today=%s). Sentiment will stay NEUTRAL until init succeeds.",
                candle_date,
                self.cpr_today is not None,
            )
            return None
        if self.current_date and candle_date != self.current_date:
            self._candle_history = []
            self.current_date = candle_date
            if self.cpr_today:
                self._init_from_cpr_today()
            else:
                if not self._ensure_initialized(candle_date, ohlc):
                    return None
        if len(self._candle_history) == 0 and self.kite:
            self._backfill_candle_history(candle_date, timestamp)
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
        last_row = out.iloc[-1]
        self._last_sentiment = str(last_row["market_sentiment"]).strip().upper()
        ncp = float(last_row["ncp"])
        in_any = self.analyzer.is_in_band(ncp, self.analyzer.bands)
        band_containing = self.analyzer.band_containing(ncp, self.analyzer.bands)
        pivot = self.analyzer.pivot
        logger.info(
            "[%s] Market Sentiment (v5): %s | OHLC: O=%.2f H=%.2f L=%.2f C=%.2f",
            timestamp.strftime("%H:%M:%S"),
            self._last_sentiment,
            row["open"],
            row["high"],
            row["low"],
            row["close"],
        )
        if self._last_sentiment == "NEUTRAL":
            if in_any and band_containing is not None:
                low_b, high_b = band_containing
                logger.info(
                    "[%s] v5 NEUTRAL reason: NCP=%.2f inside band [%.2f, %.2f] (pivot=%.2f; outside bands would be BULLISH if NCP>pivot else BEARISH)",
                    timestamp.strftime("%H:%M:%S"), ncp, low_b, high_b, pivot,
                )
            else:
                logger.info(
                    "[%s] v5 NEUTRAL reason: NCP=%.2f (pivot=%.2f) in_any=%s band_containing=%s (state from prior candle)",
                    timestamp.strftime("%H:%M:%S"), ncp, pivot, in_any, band_containing,
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
