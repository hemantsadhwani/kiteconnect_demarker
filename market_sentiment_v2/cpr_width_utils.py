"""
Utility functions for calculating CPR_PIVOT_WIDTH and determining dynamic CPR_BAND_WIDTH
"""
import yaml
import os


def calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close):
    """
    Calculate CPR_PIVOT_WIDTH (TC - BC) from previous day OHLC.
    
    Formula:
    - Pivot = (High + Low + Close) / 3
    - BC (Bottom Central Pivot) = (High + Low) / 2
    - TC (Top Central Pivot) = 2*Pivot - BC
    - CPR_PIVOT_WIDTH = |TC - BC|
    
    Returns:
        tuple: (cpr_pivot_width, tc, bc, pivot) where:
            - cpr_pivot_width: float (always positive)
            - tc: Top Central Pivot
            - bc: Bottom Central Pivot
            - pivot: Standard pivot point
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
    tc = 2 * pivot - bc  # Top Central Pivot
    cpr_pivot_width = abs(tc - bc)  # Always positive
    return cpr_pivot_width, tc, bc, pivot


def get_dynamic_cpr_band_width(cpr_pivot_width, config):
    """
    Determine dynamic CPR_BAND_WIDTH based on CPR_PIVOT_WIDTH from config.
    
    Args:
        cpr_pivot_width: Calculated CPR_PIVOT_WIDTH (TC - BC)
        config: Configuration dictionary (loaded from config.yaml)
    
    Returns:
        float: Dynamic CPR_BAND_WIDTH based on ranges in config
    """
    default_band_width = 10.0  # Default fallback
    
    # Check if CPR_PIVOT_WIDTH filter is enabled
    filter_config = config.get('CPR_PIVOT_WIDTH_FILTER', {})
    if not filter_config.get('ENABLED', False):
        # Filter disabled - return default
        return default_band_width
    
    # Read ranges from config and apply dynamic CPR_BAND_WIDTH
    ranges = filter_config.get('RANGES', [])
    if not ranges:
        return default_band_width
    
    prev_max = 0.0
    for range_config in ranges:
        max_width = range_config.get('MAX_WIDTH')
        band_width = range_config.get('CPR_BAND_WIDTH', default_band_width)
        
        if max_width is None:
            # No upper limit - this is the last/default range
            return band_width
        elif cpr_pivot_width < max_width:
            return band_width
        prev_max = max_width
    
    # Should not reach here, but return default if we do
    return default_band_width

