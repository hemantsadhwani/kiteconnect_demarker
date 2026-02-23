"""
Intraday Sentiment Analyzer for NIFTY using 1-minute OHLC and CPR + Fibonacci band logic.
Encapsulates NiftySentimentAnalyzer with exact CPR formulas, band generation, NCP, and state-machine sentiment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Helper: Fibonacci level between two prices
# ---------------------------------------------------------------------------
def calc_fib(p1: float, p2: float, ratio: float) -> float:
    """Return p1 + (p2 - p1) * ratio."""
    return p1 + (p2 - p1) * ratio


# ---------------------------------------------------------------------------
# Nifty Sentiment Analyzer (new logic)
# ---------------------------------------------------------------------------
class NiftySentimentAnalyzer:
    """
    Determines market state (BULLISH, BEARISH, NEUTRAL) from 1-minute NIFTY OHLC
    using CPR levels, Fib-derived bands, and NCP-based state machine.
    """

    def __init__(self, prev_day_ohlc: Dict[str, float]):
        """
        Args:
            prev_day_ohlc: dict with keys 'high', 'low', 'close' (PDH, PDL, PDC).
        """
        self.prev_day_ohlc = prev_day_ohlc
        self.cpr_levels = self.calculate_cpr(prev_day_ohlc)
        self.bands: List[Tuple[float, float]] = self.generate_bands(self.cpr_levels)
        # Type 1 = first 8 (gray Fib retracement S4-S3..R3-R4), Type 2 = next 9 (colored Pivot, S1..S4, R1..R4)
        self.bands_type1 = self.bands[:8]
        self.bands_type2 = self.bands[8:17]
        self.pivot = self.cpr_levels["Pivot"]

    def calculate_cpr(self, prev_day_ohlc: Dict[str, float]) -> Dict[str, float]:
        """
        Base CPR levels from previous day High, Low, Close.
        Uses exact formulas: P, TC, BC, R1–R4, S1–S4 (matches Pine Script / TradingView).
        """
        pdh = prev_day_ohlc["high"]
        pdl = prev_day_ohlc["low"]
        pdc = prev_day_ohlc["close"]

        p = (pdh + pdl + pdc) / 3
        prev_range = pdh - pdl
        tc = (pdh + pdl) / 2
        bc = (p - tc) + p
        r1 = (2 * p) - pdl
        s1 = (2 * p) - pdh
        r2 = p + prev_range
        s2 = p - prev_range
        r3 = pdh + 2 * (p - pdl)
        s3 = pdl - 2 * (pdh - p)
        r4 = r3 + (r2 - r1)
        s4 = s3 - (s1 - s2)

        return {
            "Pivot": p,
            "TC": tc,
            "BC": bc,
            "R1": r1,
            "S1": s1,
            "R2": r2,
            "S2": s2,
            "R3": r3,
            "S3": s3,
            "R4": r4,
            "S4": s4,
        }

    def generate_bands(self, cpr_levels: Dict[str, float]) -> List[Tuple[float, float]]:
        """
        Returns a single list of (lower_bound, upper_bound) for all Type 1 and Type 2 bands.
        Matches Pine Script "CPR + Fibs + Blue Smoothed Line (Filled) Extended".
        Type 1: 8 gray Fib retracement zones (S4-S3, S3-S2, S2-S1, S1-P, P-R1, R1-R2, R2-R3, R3-R4).
        Type 2: 9 colored bands (Pivot, S1, S2, S3, S4, R1, R2, R3, R4); S4/R4 use approximated outer level.
        """
        p = cpr_levels["Pivot"]
        r1, r2, r3, r4 = cpr_levels["R1"], cpr_levels["R2"], cpr_levels["R3"], cpr_levels["R4"]
        s1, s2, s3, s4 = cpr_levels["S1"], cpr_levels["S2"], cpr_levels["S3"], cpr_levels["S4"]

        bands: List[Tuple[float, float]] = []

        # Type 1: CPR Fib retracement (Gray) – 8 zones
        zones_type1 = [
            (s4, s3),
            (s3, s2),
            (s2, s1),
            (s1, p),
            (p, r1),
            (r1, r2),
            (r2, r3),
            (r3, r4),
        ]
        for level1, level2 in zones_type1:
            low_b = calc_fib(level1, level2, 0.382)
            high_b = calc_fib(level1, level2, 0.618)
            bands.append((min(low_b, high_b), max(low_b, high_b)))

        # Type 2: CPR colored bands – 9 bands (Pivot, S1, S2, S3, S4, R1, R2, R3, R4)
        # 1. Pivot band (Orange)
        low_ref = calc_fib(s1, p, 0.5)
        high_ref = calc_fib(p, r1, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 2. S1 band (Green)
        low_ref = calc_fib(s2, s1, 0.5)
        high_ref = calc_fib(s1, p, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 3. S2 band (Green)
        low_ref = calc_fib(s3, s2, 0.5)
        high_ref = calc_fib(s2, s1, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 4. S3 band (Green) – lower: s4-s3 0.5, upper: s3-s2 0.5
        low_ref = calc_fib(s4, s3, 0.5)
        high_ref = calc_fib(s3, s2, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 5. S4 band (Green) – s5_approx = s4 - (s3 - s4); lower: s5_approx-s4 0.5, upper: s4-s3 0.5
        s5_approx = s4 - (s3 - s4)
        low_ref = calc_fib(s5_approx, s4, 0.5)
        high_ref = calc_fib(s4, s3, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 6. R1 band (Red)
        low_ref = calc_fib(p, r1, 0.5)
        high_ref = calc_fib(r1, r2, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 7. R2 band (Red)
        low_ref = calc_fib(r1, r2, 0.5)
        high_ref = calc_fib(r2, r3, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 8. R3 band (Red) – lower: r2-r3 0.5, upper: r3-r4 0.5
        low_ref = calc_fib(r2, r3, 0.5)
        high_ref = calc_fib(r3, r4, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))
        # 9. R4 band (Red) – r5_approx = r4 + (r4 - r3); lower: r3-r4 0.5, upper: r4-r5_approx 0.5
        r5_approx = r4 + (r4 - r3)
        low_ref = calc_fib(r3, r4, 0.5)
        high_ref = calc_fib(r4, r5_approx, 0.5)
        bands.append((calc_fib(low_ref, high_ref, 0.382), calc_fib(low_ref, high_ref, 0.618)))

        return bands

    @staticmethod
    def is_in_band(price: float, bands_list: List[Tuple[float, float]]) -> bool:
        """True if price is inside ANY band (lower <= price <= upper)."""
        for (low_b, high_b) in bands_list:
            if low_b <= price <= high_b:
                return True
        return False

    @staticmethod
    def band_containing(price: float, bands_list: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Return the first band (low, high) that contains price, or None."""
        for band in bands_list:
            low_b, high_b = band
            if low_b <= price <= high_b:
                return band
        return None

    def calculate_ncp(self, row) -> float:
        """
        Nifty Calculated Price: do not use raw OHLC for decision.
        Bullish (Close >= Open): NCP = (High + Close) / 2
        Bearish (Close < Open): NCP = (Low + Close) / 2
        """
        open_ = row["open"]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        if close >= open_:
            return (high + close) / 2
        return (low + close) / 2

    def apply_sentiment_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stateful sentiment per row. Adds columns: ncp, market_sentiment.
        Uses efficient iteration (e.g. itertuples) for correct state transitions.
        """
        # Ensure we have open, high, low, close
        if df.index.name == "datetime" or "datetime" in df.columns:
            date_col = "datetime" if "datetime" in df.columns else df.index
        else:
            date_col = None
        required = ["open", "high", "low", "close"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"DataFrame must have column: {c}")

        ncp_arr = np.empty(len(df), dtype=float)
        sentiment_arr = np.empty(len(df), dtype=object)

        bands_list = self.bands  # All bands (Type 1 + Type 2) for Rule 2/3/4
        pivot = self.pivot

        # State
        current_sentiment = "NEUTRAL"
        last_neutral_band: Optional[Tuple[float, float]] = None

        for i in range(len(df)):
            row = df.iloc[i]
            ncp = self.calculate_ncp(row)
            ncp_arr[i] = ncp

            in_any = self.is_in_band(ncp, bands_list)

            if i == 0:
                # Rule 1: Opening bias (first candle) – use only CPR bands (Type 2), not Fib retracement (Type 1).
                # So if NCP is in a gray zone but not in any Pivot/S1..S4/R1..R4 band → BULLISH/BEARISH by pivot.
                in_any_cpr_bands = self.is_in_band(ncp, self.bands_type2)
                if not in_any_cpr_bands:
                    if ncp > pivot:
                        current_sentiment = "BULLISH"
                    else:
                        current_sentiment = "BEARISH"
                else:
                    current_sentiment = "NEUTRAL"
                    last_neutral_band = self.band_containing(ncp, bands_list)
            else:
                prev_ncp = float(ncp_arr[i - 1])
                # Rule 2: Inside any band -> NEUTRAL
                if in_any:
                    current_sentiment = "NEUTRAL"
                    last_neutral_band = self.band_containing(ncp, bands_list)
                else:
                    # Rule 2b: Cross without being inside – transition directly to BULLISH/BEARISH
                    # If NCP was above a band and is now below it (without being inside) -> BEARISH
                    # If NCP was below a band and is now above it (without being inside) -> BULLISH
                    cross_handled = False
                    for (low_b, high_b) in bands_list:
                        if prev_ncp > high_b and ncp < low_b:
                            current_sentiment = "BEARISH"
                            last_neutral_band = None
                            cross_handled = True
                            break
                        if prev_ncp < low_b and ncp > high_b:
                            current_sentiment = "BULLISH"
                            last_neutral_band = None
                            cross_handled = True
                            break
                    if not cross_handled:
                        # Rule 3: Breakouts from NEUTRAL
                        if current_sentiment == "NEUTRAL" and last_neutral_band is not None:
                            low_b, high_b = last_neutral_band
                            if ncp < low_b:
                                current_sentiment = "BEARISH"
                            elif ncp > high_b:
                                current_sentiment = "BULLISH"
                            last_neutral_band = None
                        # Rule 4: Continuation (prev BULLISH/BEARISH, not in band, no cross -> unchanged)

            sentiment_arr[i] = current_sentiment

        out = df.copy()
        out["ncp"] = ncp_arr
        out["market_sentiment"] = sentiment_arr
        return out


# ---------------------------------------------------------------------------
# Mock data generator (random walk) for demonstration
# ---------------------------------------------------------------------------
def generate_mock_ohlc(
    n_bars: int = 100,
    start_price: float = 25700.0,
    sigma: float = 15.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate 1-minute OHLC DataFrame for testing (random walk)."""
    if seed is not None:
        np.random.seed(seed)
    dates = pd.date_range("2026-01-16 09:15:00", periods=n_bars, freq="1min", tz="Asia/Kolkata")
    returns = np.random.randn(n_bars) * sigma
    close = start_price + np.cumsum(returns)
    high = close + np.abs(np.random.randn(n_bars) * sigma * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * sigma * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close}
    )


def run_mock_demo():
    """Run analyzer on mock data when no CSV is provided (demonstration)."""
    mock_df = generate_mock_ohlc(n_bars=50, start_price=25700.0, sigma=12.0, seed=42)
    prev_day_ohlc = {
        "high": 25750.0,
        "low": 25650.0,
        "close": 25700.0,
    }
    analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
    out = analyzer.apply_sentiment_logic(mock_df)
    print("Mock demo - first 10 rows (ncp, market_sentiment):")
    print(out[["date", "ncp", "market_sentiment"]].head(10).to_string())
    print(f"\nSentiment counts: {out['market_sentiment'].value_counts().to_dict()}")
    return out


# ---------------------------------------------------------------------------
# Backward compatibility: TradingSentimentAnalyzer (original) kept for plot.py
# Plot and get_swing_bands_from_sentiment_analyzer still use the old analyzer.
# ---------------------------------------------------------------------------
class TradingSentimentAnalyzer:
    """Legacy analyzer used by plot.py and swing band extraction. Kept for compatibility."""

    def __init__(self, config: dict, cpr_levels: dict):
        self.config = config
        self.cpr_levels = cpr_levels
        self.sentiment = "NEUTRAL"
        self.candles: List[dict] = []
        self.current_candle_index = -1
        self.sentiment_history: List[str] = []
        self.transition_window = self.config.get("SENTIMENT_TRANSITION_WINDOW", 5)
        self.sentiment_transition_status = "STABLE"
        self.horizontal_bands = {"resistance": [], "support": []}
        self.enable_dynamic_swing_bands = self.config.get("ENABLE_DYNAMIC_SWING_BANDS", True)
        self.enable_default_cpr_mid_bands = self.config.get("ENABLE_DEFAULT_CPR_MID_BANDS", True)
        self.cpr_band_states = self._init_cpr_states()
        self._init_default_horizontal_bands()
        self.verbose_swing_logging = self.config.get("VERBOSE_SWING_LOGGING", False)
        self.detected_swing_highs: List[dict] = []
        self.detected_swing_lows: List[dict] = []
        self.sentiment_results: List[dict] = []

    def _init_cpr_states(self):
        states = {}
        for level_name in ["R4", "R3", "R2", "R1", "PIVOT", "S1", "S2", "S3", "S4"]:
            states[level_name] = {
                "bullish_neutralized": False,
                "bearish_neutralized": False,
                "bullish_neutralized_at": -1,
                "bearish_neutralized_at": -1,
            }
        return states

    def _init_default_horizontal_bands(self):
        if not self.enable_default_cpr_mid_bands:
            return
        levels = [
            ("R4", self.cpr_levels["R4"]),
            ("R3", self.cpr_levels["R3"]),
            ("R2", self.cpr_levels["R2"]),
            ("R1", self.cpr_levels["R1"]),
            ("PIVOT", self.cpr_levels["PIVOT"]),
            ("S1", self.cpr_levels["S1"]),
            ("S2", self.cpr_levels["S2"]),
            ("S3", self.cpr_levels["S3"]),
            ("S4", self.cpr_levels["S4"]),
        ]
        threshold = self.config.get("CPR_PAIR_WIDTH_THRESHOLD", 80.0)
        width = self.config.get("HORIZONTAL_BAND_WIDTH", 5.0)
        for i in range(len(levels) - 1):
            upper_name, upper_val = levels[i]
            lower_name, lower_val = levels[i + 1]
            pair_width = upper_val - lower_val
            if pair_width > threshold:
                midpoint = (upper_val + lower_val) / 2
                band = [midpoint - width, midpoint + width]
                self.horizontal_bands["resistance"].append(band)
                self.horizontal_bands["support"].append(band)

    def process_new_candle(self, candle: dict) -> dict:
        calc_price = (
            (candle["low"] + candle["close"]) / 2 + (candle["high"] + candle["open"]) / 2
        ) / 2
        candle["calculated_price"] = calc_price
        self.candles.append(candle)
        self.current_candle_index += 1
        if self.current_candle_index == 0:
            self._run_initial_sentiment_logic(candle)
        else:
            self._run_ongoing_sentiment_logic(candle)
        self.sentiment_history.append(self.sentiment)
        if len(self.sentiment_history) > self.transition_window + 1:
            self.sentiment_history.pop(0)
        self.sentiment_transition_status = self._detect_sentiment_transition()
        result = {
            "date": candle["date"],
            "sentiment": self.sentiment,
            "sentiment_transition": self.sentiment_transition_status,
            "calculated_price": calc_price,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
        }
        self.sentiment_results.append(result)
        return result

    def _run_initial_sentiment_logic(self, candle):
        pivot_value = self.cpr_levels.get("PIVOT", self.cpr_levels.get("pivot"))
        cp = candle["calculated_price"]
        self.sentiment = "BULLISH" if cp > pivot_value else "BEARISH" if cp < pivot_value else "NEUTRAL"

    def _run_ongoing_sentiment_logic(self, candle):
        pivot_value = self.cpr_levels.get("PIVOT", self.cpr_levels.get("pivot"))
        cp = candle["calculated_price"]
        self.sentiment = "BULLISH" if cp > pivot_value else "BEARISH" if cp < pivot_value else "NEUTRAL"

    def _detect_sentiment_transition(self) -> str:
        if len(self.sentiment_history) < 2:
            return "STABLE"
        if self.sentiment_history[-1] != self.sentiment_history[-2]:
            return "JUST_CHANGED"
        if len(self.sentiment_history) >= self.transition_window:
            recent = self.sentiment_history[-self.transition_window :]
            if len(set(recent)) > 1:
                return "TRANSITIONING"
        return "STABLE"

    def print_swing_summary(self):
        print("\n[Legacy TradingSentimentAnalyzer - swing summary skipped for NiftySentimentAnalyzer path]")


if __name__ == "__main__":
    run_mock_demo()
