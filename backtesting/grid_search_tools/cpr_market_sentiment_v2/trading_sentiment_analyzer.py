import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class TradingSentimentAnalyzer:
    def __init__(self, config: dict, cpr_levels: dict):
        self.config = config
        self.cpr_levels = cpr_levels
        
        # State Variables
        self.sentiment = "NEUTRAL"
        self.candles = []  # History of candles
        self.current_candle_index = -1
        
        # Band Storage
        # Store bands as dicts with timestamp: {'band': [low, high], 'timestamp': datetime}
        self.horizontal_bands = {
            'resistance': [], # List of {'band': [low, high], 'timestamp': datetime}
            'support': []     # List of {'band': [low, high], 'timestamp': datetime}
        }

        # Feature flags (default to True to preserve original behaviour)
        # When False, dynamic horizontal S/R bands from swings are not created/used.
        self.enable_dynamic_swing_bands: bool = self.config.get('ENABLE_DYNAMIC_SWING_BANDS', True)
        # When False, default 50% CPR midpoint horizontal bands are not created/used.
        self.enable_default_cpr_mid_bands: bool = self.config.get('ENABLE_DEFAULT_CPR_MID_BANDS', True)

        # CPR Band State (Tracks Neutralization)
        # Structure: { 'R1': {'bullish_neutralized': False, 'bearish_neutralized': False, 
        #                      'bullish_neutralized_at': -1, 'bearish_neutralized_at': -1}, ... }
        # neutralized_at tracks the candle index where neutralization happened (swing detection point)
        self.cpr_band_states = self._init_cpr_states()
        
        # Initialize Default Horizontal Bands (50% rule)
        self._init_default_horizontal_bands()
        
        # Swing Detection Logging
        self.verbose_swing_logging = self.config.get('VERBOSE_SWING_LOGGING', False)
        self.detected_swing_highs: List[Dict] = []  # {'price': float, 'timestamp': str, 'status': str, 'reason': str}
        self.detected_swing_lows: List[Dict] = []   # {'price': float, 'timestamp': str, 'status': str, 'reason': str}
        
        # Store sentiment results for reprocessing after neutralization
        self.sentiment_results: List[Dict] = []  # Store results for each candle

    def _init_cpr_states(self):
        """Initialize the neutralization state for all CPR levels."""
        states = {}
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            states[level_name] = {
                'bullish_neutralized': False,
                'bearish_neutralized': False,
                'bullish_neutralized_at': -1,  # Candle index where neutralization happened
                'bearish_neutralized_at': -1   # Candle index where neutralization happened
            }
        return states

    def _get_band_values(self, band_entry):
        """Helper to extract band [low, high] from either old format [low, high] or new format {'band': [low, high], 'timestamp': ...}"""
        if isinstance(band_entry, dict):
            return band_entry['band']
        else:
            # Legacy format
            return band_entry

    def _init_default_horizontal_bands(self):
        """Create 50% bands if CPR pair width > Threshold (and feature enabled)."""
        if not self.enable_default_cpr_mid_bands:
            # Do not create default midpoint bands when disabled.
            return
        levels = [
            ('R4', self.cpr_levels['R4']), ('R3', self.cpr_levels['R3']),
            ('R2', self.cpr_levels['R2']), ('R1', self.cpr_levels['R1']),
            ('PIVOT', self.cpr_levels['PIVOT']),
            ('S1', self.cpr_levels['S1']), ('S2', self.cpr_levels['S2']),
            ('S3', self.cpr_levels['S3']), ('S4', self.cpr_levels['S4'])
        ]
        
        threshold = self.config['CPR_PAIR_WIDTH_THRESHOLD']
        width = self.config['HORIZONTAL_BAND_WIDTH']
        
        for i in range(len(levels) - 1):
            upper_name, upper_val = levels[i]
            lower_name, lower_val = levels[i+1]
            
            pair_width = upper_val - lower_val
            if pair_width > threshold:
                midpoint = (upper_val + lower_val) / 2
                band = [midpoint - width, midpoint + width]
                # Add to both lists as it acts as both S/R initially
                # Default bands have no timestamp (created at initialization, before any candles)
                self.horizontal_bands['resistance'].append({'band': band, 'timestamp': None})
                self.horizontal_bands['support'].append({'band': band, 'timestamp': None})

    def process_new_candle(self, candle: dict) -> dict:
        """
        Main entry point. Processes a single new candle.
        """
        self.candles.append(candle)
        self.current_candle_index += 1
        
        # 1. Calculate Price
        # Formula: ((L+C)/2 + (H+O)/2) / 2
        calc_price = ((candle['low'] + candle['close']) / 2 + 
                      (candle['high'] + candle['open']) / 2) / 2
        candle['calculated_price'] = calc_price

        # 2. Detect Swings (Delayed by N candles)
        neutralization_occurred = self._process_delayed_swings()

        # 3. Determine Sentiment
        if self.current_candle_index == 0:
            self._run_initial_sentiment_logic(candle)
        else:
            self._run_ongoing_sentiment_logic(candle)
        
        # Store result
        result = {
            'date': candle['date'],
            'sentiment': self.sentiment,
            'calculated_price': calc_price,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close']
        }
        self.sentiment_results.append(result)
        
        # 4. If neutralization occurred, reprocess affected candles
        if neutralization_occurred:
            self._reprocess_after_neutralization(neutralization_occurred)

        return result

    def _process_delayed_swings(self):
        """
        Checks if the candle at (Current - N) was a swing.
        If so, adds a horizontal band or neutralizes a CPR band.
        
        Returns:
            Dict with neutralization info if neutralization occurred, None otherwise
            Format: {'neutralized_at': int, 'level_name': str, 'zone_type': 'bullish'|'bearish'}
        """
        n = self.config['SWING_CONFIRMATION_CANDLES']
        target_idx = self.current_candle_index - n
        
        if target_idx < n:
            return None  # Not enough history yet

        target_candle = self.candles[target_idx]
        
        # Get timestamp for logging and plotting
        # Use the actual swing detection timestamp (target_candle), not the confirmation timestamp
        swing_detection_timestamp = target_candle.get('date', '')
        timestamp = swing_detection_timestamp
        if timestamp and ' ' in str(timestamp):
            time_part = str(timestamp).split(' ')[1] if ' ' in str(timestamp) else str(timestamp)
        else:
            time_part = str(timestamp) if timestamp else ""
        
        neutralization_info = None
        
        # Check Swing High (using calculated_price)
        if self._is_swing_high(target_idx, n):
            neutralization_info = self._handle_new_swing(target_candle['calculated_price'], 'high', time_part, swing_detection_timestamp)
            
        # Check Swing Low (using calculated_price)
        if self._is_swing_low(target_idx, n):
            neutralization_info = self._handle_new_swing(target_candle['calculated_price'], 'low', time_part, swing_detection_timestamp)
        
        return neutralization_info

    def _is_swing_high(self, idx, n):
        # Use calculated_price for swing detection
        val = self.candles[idx]['calculated_price']
        # Check N before
        for i in range(idx - n, idx):
            if self.candles[i]['calculated_price'] >= val: return False
        # Check N after
        for i in range(idx + 1, idx + n + 1):
            if self.candles[i]['calculated_price'] >= val: return False
        return True

    def _is_swing_low(self, idx, n):
        # Use calculated_price for swing detection
        val = self.candles[idx]['calculated_price']
        # Check N before
        for i in range(idx - n, idx):
            if self.candles[i]['calculated_price'] <= val: return False
        # Check N after
        for i in range(idx + 1, idx + n + 1):
            if self.candles[i]['calculated_price'] <= val: return False
        return True

    def _handle_new_swing(self, price, swing_type, timestamp="", swing_detection_datetime=None):
        """
        Decides whether to create a horizontal band or neutralize a CPR zone.
        Uses calculated_price for swing detection (passed as 'price' parameter).
        
        Args:
            price: The swing price (calculated_price from the swing candle)
            swing_type: 'high' or 'low'
            timestamp: String timestamp for logging (e.g., "09:30:00")
            swing_detection_datetime: The actual datetime when the swing occurred (for plotting)
        
        Returns:
            Dict with neutralization info if neutralization occurred, None otherwise
            Format: {'neutralized_at': int, 'level_name': str, 'zone_type': 'bullish'|'bearish'}
        """
        cpr_width = self.config['CPR_BAND_WIDTH']
        ignore_buffer = self.config['CPR_IGNORE_BUFFER']
        
        # First pass: Check if swing is INSIDE any CPR zone (neutralize takes priority)
        for name, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            
            # 1. Check if INSIDE Bullish Zone -> Neutralize
            if bullish_zone[0] <= price <= bullish_zone[1]:
                # Neutralize from the swing detection point (target_idx), not confirmation point
                n = self.config['SWING_CONFIRMATION_CANDLES']
                swing_detection_idx = self.current_candle_index - n
                self.cpr_band_states[name]['bullish_neutralized'] = True
                self.cpr_band_states[name]['bullish_neutralized_at'] = swing_detection_idx
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - NEUTRALIZED CPR {name} bullish zone [{bullish_zone[0]:.2f}, {bullish_zone[1]:.2f}]")
                # Track neutralized swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bullish zone"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bullish zone"
                    })
                return {'neutralized_at': swing_detection_idx, 'level_name': name, 'zone_type': 'bullish'}  # Do not create horizontal band
                
            # 2. Check if INSIDE Bearish Zone -> Neutralize
            if bearish_zone[0] <= price <= bearish_zone[1]:
                # Neutralize from the swing detection point (target_idx), not confirmation point
                n = self.config['SWING_CONFIRMATION_CANDLES']
                swing_detection_idx = self.current_candle_index - n
                self.cpr_band_states[name]['bearish_neutralized'] = True
                self.cpr_band_states[name]['bearish_neutralized_at'] = swing_detection_idx
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - NEUTRALIZED CPR {name} bearish zone [{bearish_zone[0]:.2f}, {bearish_zone[1]:.2f}]")
                # Track neutralized swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bearish zone"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bearish zone"
                    })
                return # Do not create horizontal band

        # Second pass: Check Ignore Buffer (ONLY if NOT inside any zone)
        # If swing is outside all zones but within buffer, ignore it
        for name, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            
            # Check if swing is within buffer of this level (but not inside any zone)
            # Distance to level
            if abs(price - level) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} level {level:.2f}, outside zones)")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} level (outside zones)"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} level (outside zones)"
                    })
                return
            
            # Distance to zone edges (only if not inside the zone)
            # Check distance to bullish zone upper edge
            if price < bullish_zone[0] and abs(price - bullish_zone[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} bullish zone lower edge {bullish_zone[0]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone lower edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone lower edge"
                    })
                return
            
            # Check distance to bullish zone upper edge
            if price > bullish_zone[1] and abs(price - bullish_zone[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} bullish zone upper edge {bullish_zone[1]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone upper edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone upper edge"
                    })
                return
            
            # Check distance to bearish zone lower edge
            if price < bearish_zone[0] and abs(price - bearish_zone[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} bearish zone lower edge {bearish_zone[0]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone lower edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone lower edge"
                    })
                return
            
            # Check distance to bearish zone upper edge
            if price > bearish_zone[1] and abs(price - bearish_zone[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} bearish zone upper edge {bearish_zone[1]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone upper edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone upper edge"
                    })
                return

        # If we reached here, create/merge horizontal band (only if dynamic swing bands are enabled)
        if self.enable_dynamic_swing_bands:
            self._add_horizontal_band(price, swing_type, timestamp, swing_detection_datetime)
        return None  # No neutralization occurred
    
    def _reprocess_after_neutralization(self, neutralization_info):
        """
        Reprocess candles from neutralization point to current candle
        to update sentiment based on neutralized zone.
        
        Args:
            neutralization_info: Dict with 'neutralized_at', 'level_name', 'zone_type'
        """
        if not neutralization_info:
            return
        
        neutralized_at = neutralization_info['neutralized_at']
        
        # Reprocess candles from neutralization point to current (including current, as it needs to be updated too)
        for idx in range(neutralized_at, self.current_candle_index + 1):
            if idx < len(self.candles) and idx < len(self.sentiment_results):
                candle = self.candles[idx]
                
                # Recalculate sentiment for this candle
                if idx == 0:
                    self._run_initial_sentiment_logic(candle)
                else:
                    # Temporarily set current_candle_index to reprocess
                    # This ensures _run_ongoing_sentiment_logic uses the correct previous candle
                    original_index = self.current_candle_index
                    self.current_candle_index = idx
                    self._run_ongoing_sentiment_logic(candle)
                    self.current_candle_index = original_index
                
                # Update stored result
                self.sentiment_results[idx]['sentiment'] = self.sentiment

    def _bands_overlap(self, band_a, band_b):
        """Return True if two bands overlap (inclusive)."""
        return not (band_a[1] < band_b[0] or band_b[1] < band_a[0])

    def _add_horizontal_band(self, price, swing_type, timestamp="", swing_detection_datetime=None):
        width = self.config['HORIZONTAL_BAND_WIDTH']
        tolerance = self.config['MERGE_TOLERANCE']
        
        target_list = self.horizontal_bands['resistance'] if swing_type == 'high' else self.horizontal_bands['support']
        band_type = 'RESISTANCE' if swing_type == 'high' else 'SUPPORT'
        
        # Use the swing detection datetime (when the swing actually occurred), not the confirmation time
        # This ensures the band is drawn from the actual peak/low point, not from when it was confirmed
        detection_datetime = swing_detection_datetime
        if detection_datetime is None:
            # Fallback: use current candle if swing_detection_datetime not provided (shouldn't happen)
            if self.current_candle_index >= 0 and len(self.candles) > 0:
                detection_datetime = self.candles[self.current_candle_index].get('date')
        
        # Build the proposed band range for overlap checks
        proposed_band = [price - width, price + width]

        # 1) Ignore if this band overlaps any CPR zone (bullish or bearish) for any level
        cpr_width = self.config['CPR_BAND_WIDTH']
        for _, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            if self._bands_overlap(proposed_band, bullish_zone) or self._bands_overlap(proposed_band, bearish_zone):
                if self.verbose_swing_logging:
                    print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: overlaps CPR zone")
                return

        # 2) Ignore if this band overlaps any pre-initialized default horizontal band (timestamp is None)
        for default_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
            if isinstance(default_entry, dict):
                default_band = default_entry['band']
                default_ts = default_entry.get('timestamp')
            else:
                default_band = default_entry
                default_ts = None
            # Consider only default (pre-init) bands
            if default_ts is None and self._bands_overlap(proposed_band, default_band):
                if self.verbose_swing_logging:
                    print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: overlaps default midpoint band [{default_band[0]:.2f}, {default_band[1]:.2f}]")
                    print(f"       Proposed band: [{proposed_band[0]:.2f}, {proposed_band[1]:.2f}] (center {price:.2f})")
                return

        # 3) Before merging, check if merging would create a band that overlaps default midpoint bands
        # Try to merge
        for i, band_entry in enumerate(target_list):
            # Handle both old format [low, high] and new format {'band': [low, high], 'timestamp': ...}
            if isinstance(band_entry, dict):
                band = band_entry['band']
            else:
                # Legacy format: convert to new format
                band = band_entry
                band_entry = {'band': band, 'timestamp': None}
                target_list[i] = band_entry
            
            center = (band[0] + band[1]) / 2
            if abs(price - center) <= tolerance:
                # Merge: Expand band to cover both original band and new price
                # This ensures merged band covers the full range of both bands
                old_band = band.copy()
                # Calculate the expanded range that covers both bands
                min_bound = min(band[0], price - width)
                max_bound = max(band[1], price + width)
                # Create expanded band centered on the average, but wide enough to cover both
                new_center = (price + center) / 2
                # Ensure the band covers both original ranges
                expanded_min = min(min_bound, new_center - width)
                expanded_max = max(max_bound, new_center + width)
                merged_band = [expanded_min, expanded_max]
                
                # Check if the merged band would overlap any default midpoint band
                # If so, ignore this merge and don't create the band
                existing_timestamp = band_entry.get('timestamp') if isinstance(band_entry, dict) else None
                # Only check overlap if we're merging with a dynamic band (has timestamp)
                # If merging with a default band (no timestamp), we should ignore it
                if existing_timestamp is None:
                    # We're trying to merge with a default band - this should be ignored
                    if self.verbose_swing_logging:
                        print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: would merge with default midpoint band [{old_band[0]:.2f}, {old_band[1]:.2f}]")
                        print(f"       Proposed band: [{proposed_band[0]:.2f}, {proposed_band[1]:.2f}] (center {price:.2f})")
                    return
                
                # Check if merged band overlaps any default midpoint band
                for default_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                    if isinstance(default_entry, dict):
                        default_band = default_entry['band']
                        default_ts = default_entry.get('timestamp')
                    else:
                        default_band = default_entry
                        default_ts = None
                    # Check overlap with default bands only
                    if default_ts is None and self._bands_overlap(merged_band, default_band):
                        if self.verbose_swing_logging:
                            print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: merged band would overlap default midpoint band [{default_band[0]:.2f}, {default_band[1]:.2f}]")
                            print(f"       Merged band: [{merged_band[0]:.2f}, {merged_band[1]:.2f}] (center {new_center:.2f})")
                        return
                
                # Keep the earliest timestamp (when the band was first created)
                target_list[i] = {'band': merged_band, 'timestamp': existing_timestamp if existing_timestamp else detection_datetime}
                if self.verbose_swing_logging:
                    print(f"    -> MERGED with existing {band_type.lower()} band {i+1} @{timestamp}: [{old_band[0]:.2f}, {old_band[1]:.2f}] (center {center:.2f})")
                    print(f"       New merged band: [{merged_band[0]:.2f}, {merged_band[1]:.2f}] (center {new_center:.2f})")
                # Track valid swing (merged)
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'VALID',
                        'reason': f"merged with existing {band_type.lower()} band"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'VALID',
                        'reason': f"merged with existing {band_type.lower()} band"
                    })
                return

        # Create new
        new_band = [price - width, price + width]
        target_list.append({'band': new_band, 'timestamp': detection_datetime})
        if self.verbose_swing_logging:
            print(f"    -> CREATED new {band_type.lower()} band @{timestamp}: [{new_band[0]:.2f}, {new_band[1]:.2f}] (center {price:.2f})")
        # Track valid swing (created)
        if swing_type == 'high':
            self.detected_swing_highs.append({
                'price': price,
                'timestamp': timestamp,
                'status': 'VALID',
                'reason': f"created new {band_type.lower()} band"
            })
        else:
            self.detected_swing_lows.append({
                'price': price,
                'timestamp': timestamp,
                'status': 'VALID',
                'reason': f"created new {band_type.lower()} band"
            })

    def _run_initial_sentiment_logic(self, candle):
        """
        Determines sentiment based on the very first candle of the day.
        Uses robust logic with calculated_price, raw high/low, touch-and-move detection,
        and horizontal band cross detection for better edge case handling.
        """
        high = candle['high']
        low = candle['low']
        close = candle['close']
        open_price = candle['open']
        calculated_price = candle['calculated_price']
        cpr_width = self.config['CPR_BAND_WIDTH']
        
        # Track all band interactions (to use the last one)
        last_interaction = None
        last_interaction_type = None
        sentiment_determined = False
        
        # Helper to get CPR level name in lowercase (for compatibility with old logic)
        def get_level_name_lower(name):
            """Convert 'R4' -> 'r4', 'PIVOT' -> 'pivot', etc."""
            return name.lower() if name != 'PIVOT' else 'pivot'
        
        # Check CPR bands - prioritize bearish/bullish zones
        # CPR bands (R4, R3, R2, R1, PIVOT): if calculated_price is in bearish zone → BEARISH
        cpr_levels_upper = ['R4', 'R3', 'R2', 'R1', 'PIVOT']
        for level_name in cpr_levels_upper:
            if level_name not in self.cpr_levels:
                continue
            level = self.cpr_levels[level_name]
            state = self.cpr_band_states[level_name]
            bearish_zone = [level - cpr_width, level]
            
            # Check if calculated_price is inside bearish zone of CPR band
            if bearish_zone[0] <= calculated_price <= bearish_zone[1]:
                # Check if neutralized AND current candle is at or after neutralization point
                if state['bearish_neutralized'] and self.current_candle_index >= state.get('bearish_neutralized_at', -1):
                    self.sentiment = 'NEUTRAL'
                else:
                    self.sentiment = 'BEARISH'
                sentiment_determined = True
                last_interaction = get_level_name_lower(level_name)
                last_interaction_type = 'CPR_BEARISH'
                break
        
        # CPR bands (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4): if calculated_price is in bullish zone → BULLISH
        # IMPORTANT: PIVOT must be included - it works exactly like any other CPR band
        if not sentiment_determined:
            # Check all CPR bands for bullish zones (including PIVOT)
            all_cpr_levels_bullish = ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']
            for level_name in all_cpr_levels_bullish:
                if level_name not in self.cpr_levels:
                    continue
                level = self.cpr_levels[level_name]
                state = self.cpr_band_states[level_name]
                bullish_zone = [level, level + cpr_width]
                
                # Check if calculated_price is inside bullish zone of CPR band
                if bullish_zone[0] <= calculated_price <= bullish_zone[1]:
                    # Check if neutralized AND current candle is at or after neutralization point
                    if state['bullish_neutralized'] and self.current_candle_index >= state.get('bullish_neutralized_at', -1):
                        self.sentiment = 'NEUTRAL'
                    else:
                        self.sentiment = 'BULLISH'
                    sentiment_determined = True
                    last_interaction = get_level_name_lower(level_name)
                    last_interaction_type = 'CPR_BULLISH'
                    break
        
        # Check if high or low falls inside any CPR band zone (fallback check using raw OHLC)
        # This is a secondary check - if calculated_price didn't trigger, check raw values
        if not sentiment_determined:
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                if level_name not in self.cpr_levels:
                    continue
                level = self.cpr_levels[level_name]
                state = self.cpr_band_states[level_name]
                bullish_zone = [level, level + cpr_width]
                bearish_zone = [level - cpr_width, level]
                
                # Check if high/low is inside bearish zone → BEARISH
                if (bearish_zone[0] <= high <= bearish_zone[1] or 
                    bearish_zone[0] <= low <= bearish_zone[1]):
                    if state['bearish_neutralized']:
                        self.sentiment = 'NEUTRAL'
                    else:
                        self.sentiment = 'BEARISH'
                    sentiment_determined = True
                    break
                # Check if high/low is inside bullish zone → BULLISH
                elif (bullish_zone[0] <= high <= bullish_zone[1] or 
                      bullish_zone[0] <= low <= bullish_zone[1]):
                    if state['bullish_neutralized']:
                        self.sentiment = 'NEUTRAL'
                    else:
                        self.sentiment = 'BULLISH'
                    sentiment_determined = True
                    break
        
        # Check horizontal bands - check all and track the last one
        # NEUTRAL only occurs from horizontal band interactions, not CPR bands
        if not sentiment_determined:
            for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                band = self._get_band_values(band_entry)
                # Check if calculated_price is inside the band
                if band[0] <= calculated_price <= band[1]:
                    last_interaction = 'horizontal_band'
                    last_interaction_type = 'HORIZONTAL'
        
        # If inside horizontal band and sentiment not determined, set to NEUTRAL
        # Note: CPR band interactions always set BEARISH/BULLISH, never NEUTRAL
        if last_interaction and not sentiment_determined:
            self.sentiment = 'NEUTRAL'
        else:
            # Check if price touched a band and moved away
            touched_and_moved = False
            last_touched_band = None
            last_touched_type = None
            
            # Check CPR bands - check all and track the last one
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                if level_name not in self.cpr_levels:
                    continue
                level = self.cpr_levels[level_name]
                bullish_zone = [level, level + cpr_width]
                bearish_zone = [level - cpr_width, level]
                
                # Check if high or low touched a band boundary
                touched = False
                if (low <= bullish_zone[1] <= high) or (low <= bearish_zone[0] <= high):
                    touched = True
                    last_touched_band = get_level_name_lower(level_name)
                    last_touched_type = 'CPR'
                    
                    # Check where close is relative to the band
                    if close < bearish_zone[0]:
                        self.sentiment = 'BEARISH'
                        touched_and_moved = True
                    elif close > bullish_zone[1]:
                        self.sentiment = 'BULLISH'
                        touched_and_moved = True
            
            # Check horizontal bands if not already determined
            if not touched_and_moved:
                for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                    band = self._get_band_values(band_entry)
                    band_lower = band[0]
                    band_upper = band[1]
                    
                    # Check if price crossed the band completely
                    if (open_price < band_lower and close > band_upper) or (open_price > band_upper and close < band_lower):
                        last_touched_band = 'horizontal_band'
                        last_touched_type = 'HORIZONTAL'
                        
                        if close < band_lower:
                            self.sentiment = 'BEARISH'
                            touched_and_moved = True
                            break
                        elif close > band_upper:
                            self.sentiment = 'BULLISH'
                            touched_and_moved = True
                            break
            
            # If still not determined, use pivot-based logic
            if not touched_and_moved:
                pivot_value = self.cpr_levels['PIVOT']
                if low > pivot_value:
                    self.sentiment = 'BULLISH'
                elif high < pivot_value:
                    self.sentiment = 'BEARISH'
                else:
                    self.sentiment = 'NEUTRAL'

    def _run_ongoing_sentiment_logic(self, candle):
        """
        Logic for every subsequent candle.
        Implements the 'Sticky' state machine and Hybrid Price Checks.
        """
        calc_price = candle['calculated_price']
        high = candle['high']
        low = candle['low']
        cpr_width = self.config['CPR_BAND_WIDTH']
        
        # --- 1. Check CPR Band Interactions ---
        # PRIORITY 1: Check for high/low touching zones (resistance/support rejection)
        # This takes highest priority because it's a stronger signal
        # IMPORTANT: When sentiment is BEARISH, prioritize checking for reversal to BULLISH (low touching bullish zones)
        # When sentiment is BULLISH, prioritize checking for reversal to BEARISH (high touching bearish zones)
        
        # First pass: Check for reversal based on current sentiment
        if self.sentiment == "BEARISH":
            # When BEARISH, look for reversal to BULLISH: check if low OR calculated_price touches bullish zone
            # According to Hybrid Price Check: Trigger points are low AND calculated_price
            # If either touches a bullish zone, flip to BULLISH
            for name, level in self.cpr_levels.items():
                state = self.cpr_band_states[name]
                bull_zone = [level, level + cpr_width]
                
                # Check if low touches Bullish Zone (Support) -> BULLISH
                # When low hits a bullish zone, it's a bounce from support
                if bull_zone[0] <= low <= bull_zone[1]:
                    # IMPORTANT: Neutralization does NOT affect touch detection.
                    # Touching a (previously bullish) zone low still represents a bullish bounce.
                    self.sentiment = "BULLISH"
                    return  # Low touching support takes priority when BEARISH
                
                # Also check if calculated_price touches Bullish Zone -> BULLISH (unless neutralized)
                # This is part of Hybrid Price Check: either low OR calculated_price can trigger reversal
                if bull_zone[0] <= calc_price <= bull_zone[1]:
                    # Check if neutralized AND current candle is at or after neutralization point
                    if state['bullish_neutralized'] and self.current_candle_index >= state.get('bullish_neutralized_at', -1):
                        self.sentiment = "NEUTRAL"
                    else:
                        self.sentiment = "BULLISH"
                    return  # Calculated price touching support takes priority when BEARISH
        
        elif self.sentiment == "BULLISH":
            # When BULLISH, look for reversal to BEARISH: check if high touches bearish zone
            for name, level in self.cpr_levels.items():
                state = self.cpr_band_states[name]
                bear_zone = [level - cpr_width, level]
                
                # High touches Bearish Zone (Resistance) -> BEARISH
                # When high hits a bearish zone, it's a rejection at resistance
                if bear_zone[0] <= high <= bear_zone[1]:
                    # IMPORTANT: Neutralization does NOT affect touch detection.
                    # Touching a (previously bearish) zone high still represents a bearish rejection.
                    self.sentiment = "BEARISH"
                    return  # High touching resistance takes priority when BULLISH
        
        # Second pass: Check all zones (for NEUTRAL sentiment or if no reversal detected)
        for name, level in self.cpr_levels.items():
            state = self.cpr_band_states[name]
            bull_zone = [level, level + cpr_width]
            bear_zone = [level - cpr_width, level]
            
            # A. High touches Bearish Zone (Resistance) -> BEARISH
            # When high hits a bearish zone, it's a rejection at resistance.
            # IMPORTANT: Neutralization does NOT affect touch detection.
            if bear_zone[0] <= high <= bear_zone[1]:
                self.sentiment = "BEARISH"
                return  # High touching resistance takes priority
            
            # B. Low touches Bullish Zone (Support) -> BULLISH
            # When low hits a bullish zone, it's a bounce from support.
            # IMPORTANT: Neutralization does NOT affect touch detection.
            if bull_zone[0] <= low <= bull_zone[1]:
                self.sentiment = "BULLISH"
                return  # Low touching support takes priority
        
        # Collect all horizontal bands once (used later after CPR checks)
        all_horiz = self.horizontal_bands['resistance'] + self.horizontal_bands['support']

        # PRIORITY 2: Check if calculated_price is INSIDE any CPR zone
        # (Neutralized CPR zones turn "inside" to NEUTRAL, but do not block breakouts)
        for name, level in self.cpr_levels.items():
            state = self.cpr_band_states[name]
            
            # Define Zones
            bull_zone = [level, level + cpr_width]
            bear_zone = [level - cpr_width, level]
            
            # C. Bullish Zone: [level, level + CPR_BAND_WIDTH]
            # Check if calculated_price is INSIDE bullish zone
            if bull_zone[0] <= calc_price <= bull_zone[1]:
                # Check if neutralized AND current candle is at or after neutralization point
                if state['bullish_neutralized'] and self.current_candle_index >= state.get('bullish_neutralized_at', -1):
                    self.sentiment = "NEUTRAL"
                else:
                    self.sentiment = "BULLISH"
                return  # Found a match, exit early
            
            # D. Bearish Zone: [level - CPR_BAND_WIDTH, level]
            # Check if calculated_price is INSIDE bearish zone
            if bear_zone[0] <= calc_price <= bear_zone[1]:
                # Check if neutralized AND current candle is at or after neutralization point
                if state['bearish_neutralized'] and self.current_candle_index >= state.get('bearish_neutralized_at', -1):
                    self.sentiment = "NEUTRAL"
                else:
                    self.sentiment = "BEARISH"
                return  # Found a match, exit early
        
        # PRIORITY 3: Check for breakout/breakdown (price crossed ABOVE or BELOW zones)
        # Use current_candle_index - 1 instead of -2 to respect reprocessing
        prev_idx = self.current_candle_index - 1
        if prev_idx >= 0 and prev_idx < len(self.candles):
            prev_calc_price = self.candles[prev_idx]['calculated_price']
        else:
            prev_calc_price = calc_price  # Fallback if no previous candle
        for name, level in self.cpr_levels.items():
            bull_zone = [level, level + cpr_width]
            bear_zone = [level - cpr_width, level]
            
            # E. Breakout: Price crosses ABOVE the Bullish Zone -> BULLISH
            # Using hybrid check: calculated_price OR high crosses above
            if (calc_price > bull_zone[1] or high > bull_zone[1]) and prev_calc_price <= bull_zone[1]:
                self.sentiment = "BULLISH"
                return
            
            # F. Breakdown: Price crosses BELOW the Bearish Zone -> BEARISH
            # Using hybrid check: calculated_price OR low crosses below
            if (calc_price < bear_zone[0] or low < bear_zone[0]) and prev_calc_price >= bear_zone[0]:
                self.sentiment = "BEARISH"
                return 

        # PRIORITY 4: Check Horizontal Band Interactions (inside horizontal bands)
        # Only apply horizontal "inside = NEUTRAL" after giving CPR zones a chance
        for band_entry in all_horiz:
            band = self._get_band_values(band_entry)
            if band[0] <= calc_price <= band[1]:
                self.sentiment = "NEUTRAL"
                return

        # --- Check Horizontal Band Crosses (after CPR breakout/breakdown and inside checks) ---
        # Check for horizontal band crosses (breakout/breakdown)
        for band_entry in all_horiz:
            band = self._get_band_values(band_entry)
            
            # Break Above -> BULLISH
            # We check if we crossed it in this candle
            # Use current_candle_index - 1 instead of -2 to respect reprocessing
            prev_idx = self.current_candle_index - 1
            if prev_idx >= 0 and prev_idx < len(self.candles):
                prev_band_price = self.candles[prev_idx]['calculated_price']
            else:
                prev_band_price = calc_price  # Fallback if no previous candle
                
            if prev_band_price <= band[1] and calc_price > band[1]:
                self.sentiment = "BULLISH"
                return
                
            # Break Below -> BEARISH
            if prev_band_price >= band[0] and calc_price < band[0]:
                self.sentiment = "BEARISH"
                return

        # --- 3. Implicit Crossing (Gap/Jump Logic) ---
        # This runs last to catch jumps over bands
        # Use current_candle_index - 1 instead of -2 to respect reprocessing
        prev_idx = self.current_candle_index - 1
        if prev_idx >= 0 and prev_idx < len(self.candles):
            prev_price = self.candles[prev_idx]['calculated_price']
        else:
            prev_price = calc_price  # Fallback if no previous candle
        curr_pair = self._get_cpr_pair(calc_price)
        prev_pair = self._get_cpr_pair(prev_price)
        
        if curr_pair and prev_pair and curr_pair != prev_pair:
            # Determine direction of pairs (R4 is highest, S4 is lowest)
            # Simple index comparison
            order = ['R4_R3', 'R3_R2', 'R2_R1', 'R1_PIVOT', 'PIVOT_S1', 'S1_S2', 'S2_S3', 'S3_S4']
            try:
                curr_idx = order.index(curr_pair)
                prev_idx = order.index(prev_pair)
                
                # Lower index = Higher Price (R4 is top)
                if curr_idx < prev_idx: # Moved Up
                    self.sentiment = "BULLISH"
                elif curr_idx > prev_idx: # Moved Down
                    self.sentiment = "BEARISH"
            except ValueError:
                pass # Pair not in standard list

    def _get_cpr_pair(self, price):
        # Helper to find which pair the price is in
        # Note: This assumes standard ordering R4 > R3 ... > S4
        levels = [
            ('R4', self.cpr_levels['R4']), ('R3', self.cpr_levels['R3']),
            ('R2', self.cpr_levels['R2']), ('R1', self.cpr_levels['R1']),
            ('PIVOT', self.cpr_levels['PIVOT']),
            ('S1', self.cpr_levels['S1']), ('S2', self.cpr_levels['S2']),
            ('S3', self.cpr_levels['S3']), ('S4', self.cpr_levels['S4'])
        ]
        
        for i in range(len(levels) - 1):
            upper = levels[i][1]
            lower = levels[i+1][1]
            # Handle case where levels might be inverted or messy, but standard CPR is ordered
            if lower <= price <= upper:
                return f"{levels[i][0]}_{levels[i+1][0]}"
        return None
    
    def print_swing_summary(self):
        """Print a consolidated summary of all detected swings (both valid and ignored)."""
        print("\n" + "=" * 80)
        print("SWING HIGH/LOW DETECTION SUMMARY")
        print("=" * 80)
        
        # Count valid vs ignored vs neutralized swings
        valid_highs = [s for s in self.detected_swing_highs if s['status'] == 'VALID']
        ignored_highs = [s for s in self.detected_swing_highs if s['status'] == 'IGNORED']
        neutralized_highs = [s for s in self.detected_swing_highs if s['status'] == 'NEUTRALIZED']
        
        valid_lows = [s for s in self.detected_swing_lows if s['status'] == 'VALID']
        ignored_lows = [s for s in self.detected_swing_lows if s['status'] == 'IGNORED']
        neutralized_lows = [s for s in self.detected_swing_lows if s['status'] == 'NEUTRALIZED']
        
        print(f"\nTotal Swing Highs Detected: {len(self.detected_swing_highs)}")
        print(f"  Valid: {len(valid_highs)}")
        print(f"  Ignored: {len(ignored_highs)}")
        print(f"  Neutralized: {len(neutralized_highs)}")
        
        print(f"\nTotal Swing Lows Detected: {len(self.detected_swing_lows)}")
        print(f"  Valid: {len(valid_lows)}")
        print(f"  Ignored: {len(ignored_lows)}")
        print(f"  Neutralized: {len(neutralized_lows)}")
        
        # Print all swing highs
        if self.detected_swing_highs:
            print("\n" + "-" * 80)
            print("SWING HIGHS (All Detected)")
            print("-" * 80)
            for i, swing in enumerate(self.detected_swing_highs, 1):
                status_marker = f"[{swing['status']}]"
                print(f"  {i}. {status_marker} @{swing['timestamp']} Price: {swing['price']:.2f} - {swing['reason']}")
        
        # Print all swing lows
        if self.detected_swing_lows:
            print("\n" + "-" * 80)
            print("SWING LOWS (All Detected)")
            print("-" * 80)
            for i, swing in enumerate(self.detected_swing_lows, 1):
                status_marker = f"[{swing['status']}]"
                print(f"  {i}. {status_marker} @{swing['timestamp']} Price: {swing['price']:.2f} - {swing['reason']}")
        
        # Print neutralized CPR zones summary
        neutralized_zones = []
        for name, state in self.cpr_band_states.items():
            if state['bullish_neutralized'] or state['bearish_neutralized']:
                zones = []
                if state['bullish_neutralized']:
                    zones.append('bullish')
                if state['bearish_neutralized']:
                    zones.append('bearish')
                neutralized_zones.append(f"{name} ({', '.join(zones)})")
        
        if neutralized_zones:
            print("\n" + "-" * 80)
            print("NEUTRALIZED CPR ZONES")
            print("-" * 80)
            for zone in neutralized_zones:
                print(f"  - {zone}")
        
        print("=" * 80)