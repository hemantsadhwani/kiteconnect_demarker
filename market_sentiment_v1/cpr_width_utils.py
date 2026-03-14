"""
Utility functions for calculating CPR_PIVOT_WIDTH and determining dynamic CPR_BAND_WIDTH.
Ported from backtesting/grid_search_tools/cpr_market_sentiment_v1.
"""


def calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close):
    """
    Calculate CPR_PIVOT_WIDTH (TC - BC) from previous day OHLC.
    Returns: (cpr_pivot_width, tc, bc, pivot)
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    bc = (prev_day_high + prev_day_low) / 2
    tc = 2 * pivot - bc
    return abs(tc - bc), tc, bc, pivot


def get_dynamic_cpr_band_width(cpr_pivot_width, config):
    """Determine dynamic CPR_BAND_WIDTH based on CPR_PIVOT_WIDTH from config."""
    default_band_width = 10.0
    filter_config = config.get('CPR_PIVOT_WIDTH_FILTER', {})
    if not filter_config.get('ENABLED', False):
        return default_band_width
    ranges = filter_config.get('RANGES', [])
    if not ranges:
        return default_band_width
    for range_config in ranges:
        max_width = range_config.get('MAX_WIDTH')
        band_width = range_config.get('CPR_BAND_WIDTH', default_band_width)
        if max_width is None or cpr_pivot_width < max_width:
            return band_width
    return default_band_width
