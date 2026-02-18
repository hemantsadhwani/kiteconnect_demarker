"""
Intraday Sentiment Analyzer for NIFTY using 1-minute OHLC and CPR + Fibonacci band logic.
Production v5: NiftySentimentAnalyzer with exact CPR formulas, band generation, NCP state machine.
Ported from backtesting_st50/grid_search_tools/cpr_market_sentiment_v5.
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
# Nifty Sentiment Analyzer (v5 logic)
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
        self.bands_type1 = self.bands[:8]
        self.bands_type2 = self.bands[8:17]
        self.pivot = self.cpr_levels["Pivot"]

    def calculate_cpr(self, prev_day_ohlc: Dict[str, float]) -> Dict[str, float]:
        """Base CPR levels from previous day High, Low, Close. Matches Pine Script / TradingView."""
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
            "Pivot": p, "TC": tc, "BC": bc,
            "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3, "R4": r4, "S4": s4,
        }

    def generate_bands(self, cpr_levels: Dict[str, float]) -> List[Tuple[float, float]]:
        """Type 1 (8 gray) + Type 2 (9 colored) bands. Matches Pine Script CPR + Fibs."""
        p = cpr_levels["Pivot"]
        r1, r2, r3, r4 = cpr_levels["R1"], cpr_levels["R2"], cpr_levels["R3"], cpr_levels["R4"]
        s1, s2, s3, s4 = cpr_levels["S1"], cpr_levels["S2"], cpr_levels["S3"], cpr_levels["S4"]
        bands: List[Tuple[float, float]] = []
        for (level1, level2) in [
            (s4, s3), (s3, s2), (s2, s1), (s1, p), (p, r1), (r1, r2), (r2, r3), (r3, r4),
        ]:
            low_b = calc_fib(level1, level2, 0.382)
            high_b = calc_fib(level1, level2, 0.618)
            bands.append((min(low_b, high_b), max(low_b, high_b)))
        s5_approx = s4 - (s3 - s4)
        r5_approx = r4 + (r4 - r3)
        for (lo, hi) in [
            (calc_fib(s1, p, 0.5), calc_fib(p, r1, 0.5)),
            (calc_fib(s2, s1, 0.5), calc_fib(s1, p, 0.5)),
            (calc_fib(s3, s2, 0.5), calc_fib(s2, s1, 0.5)),
            (calc_fib(s4, s3, 0.5), calc_fib(s3, s2, 0.5)),
            (calc_fib(s5_approx, s4, 0.5), calc_fib(s4, s3, 0.5)),
            (calc_fib(p, r1, 0.5), calc_fib(r1, r2, 0.5)),
            (calc_fib(r1, r2, 0.5), calc_fib(r2, r3, 0.5)),
            (calc_fib(r2, r3, 0.5), calc_fib(r3, r4, 0.5)),
            (calc_fib(r3, r4, 0.5), calc_fib(r4, r5_approx, 0.5)),
        ]:
            bands.append((calc_fib(lo, hi, 0.382), calc_fib(lo, hi, 0.618)))
        return bands

    @staticmethod
    def is_in_band(price: float, bands_list: List[Tuple[float, float]]) -> bool:
        for (low_b, high_b) in bands_list:
            if low_b <= price <= high_b:
                return True
        return False

    @staticmethod
    def band_containing(price: float, bands_list: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        for band in bands_list:
            low_b, high_b = band
            if low_b <= price <= high_b:
                return band
        return None

    def calculate_ncp(self, row) -> float:
        """Nifty Calculated Price: Bullish (C>=O) -> (H+C)/2, Bearish -> (L+C)/2."""
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        return (h + c) / 2 if c >= o else (l + c) / 2

    def apply_sentiment_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stateful sentiment per row. Adds columns: ncp, market_sentiment."""
        for c in ["open", "high", "low", "close"]:
            if c not in df.columns:
                raise ValueError(f"DataFrame must have column: {c}")
        ncp_arr = np.empty(len(df), dtype=float)
        sentiment_arr = np.empty(len(df), dtype=object)
        current_sentiment = "NEUTRAL"
        last_neutral_band: Optional[Tuple[float, float]] = None
        for i in range(len(df)):
            row = df.iloc[i]
            ncp = self.calculate_ncp(row)
            ncp_arr[i] = ncp
            in_any = self.is_in_band(ncp, self.bands)
            if i == 0:
                in_any_cpr_bands = self.is_in_band(ncp, self.bands_type2)
                if not in_any_cpr_bands:
                    current_sentiment = "BULLISH" if ncp > self.pivot else "BEARISH"
                else:
                    current_sentiment = "NEUTRAL"
                    last_neutral_band = self.band_containing(ncp, self.bands)
            else:
                prev_ncp = float(ncp_arr[i - 1])
                if in_any:
                    current_sentiment = "NEUTRAL"
                    last_neutral_band = self.band_containing(ncp, self.bands)
                else:
                    cross_handled = False
                    for (low_b, high_b) in self.bands:
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
                    if not cross_handled and current_sentiment == "NEUTRAL" and last_neutral_band is not None:
                        low_b, high_b = last_neutral_band
                        if ncp < low_b:
                            current_sentiment = "BEARISH"
                        elif ncp > high_b:
                            current_sentiment = "BULLISH"
                        last_neutral_band = None
            sentiment_arr[i] = current_sentiment
        out = df.copy()
        out["ncp"] = ncp_arr
        out["market_sentiment"] = sentiment_arr
        return out
