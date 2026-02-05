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
        self.horizontal_bands = {
            'resistance': [], # List of [low, high]
            'support': []     # List of [low, high]
        }

        # Config flags
        self.enable_dynamic_horizontal_bands: bool = self.config.get(
            'ENABLE_DYNAMIC_HORIZONTAL_BANDS', True
        )
        
        # Initialize Default Horizontal Bands (50% rule)
        self._init_default_horizontal_bands()
        
        # Swing Detection Logging
        self.verbose_swing_logging = self.config.get('VERBOSE_SWING_LOGGING', False)
        self.detected_swing_highs: List[Dict] = []  # {'price': float, 'timestamp': str, 'status': str, 'reason': str}
        self.detected_swing_lows: List[Dict] = []   # {'price': float, 'timestamp': str, 'status': str, 'reason': str}
        
    def _init_default_horizontal_bands(self):
        """Create 50% bands if CPR pair width > Threshold."""
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
                self.horizontal_bands['resistance'].append(band)
                self.horizontal_bands['support'].append(band)

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
        self._process_delayed_swings()

        # 3. Determine Sentiment
        if self.current_candle_index == 0:
            self._run_initial_sentiment_logic(candle)
        else:
            self._run_ongoing_sentiment_logic(candle)
        
        result = {
            'date': candle['date'],
            'sentiment': self.sentiment,
            'calculated_price': calc_price,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close']
        }

        return result

    def _process_delayed_swings(self):
        """
        Checks if the candle at (Current - N) was a swing.
        If enabled, decides whether to create/merge a horizontal band or ignore it
        based on CPR proximity rules.
        """
        # Short-circuit if dynamic horizontal bands are disabled
        if not self.enable_dynamic_horizontal_bands:
            return None
        n = self.config['SWING_CONFIRMATION_CANDLES']
        target_idx = self.current_candle_index - n
        
        if target_idx < n:
            return None  # Not enough history yet

        target_candle = self.candles[target_idx]
        
        # Get timestamp for logging
        timestamp = target_candle.get('date', '')
        if timestamp and ' ' in str(timestamp):
            time_part = str(timestamp).split(' ')[1] if ' ' in str(timestamp) else str(timestamp)
        else:
            time_part = str(timestamp) if timestamp else ""
        
        # Check Swing High (using calculated_price)
        if self._is_swing_high(target_idx, n):
            self._handle_new_swing(target_candle['calculated_price'], 'high', time_part)
            
        # Check Swing Low (using calculated_price)
        if self._is_swing_low(target_idx, n):
            self._handle_new_swing(target_candle['calculated_price'], 'low', time_part)

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

    def _handle_new_swing(self, price, swing_type, timestamp=""):
        """
        Decides whether to create a horizontal band or ignore the swing
        based on its proximity to CPR NEUTRAL zones.
        Uses calculated_price for swing detection (passed as 'price' parameter).
        """
        cpr_width = self.config['CPR_BAND_WIDTH']
        ignore_buffer = self.config['CPR_IGNORE_BUFFER']
        
        # First pass: If swing is INSIDE any CPR NEUTRAL zone, ignore it.
        # CPR zones are already neutral consolidation areas; we generally do not
        # create additional horizontal bands inside them.
        for name, level in self.cpr_levels.items():
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]
            
            if upper_neutral[0] <= price <= upper_neutral[1] or lower_neutral[0] <= price <= lower_neutral[1]:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (inside CPR NEUTRAL zone around {name})")
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"inside CPR NEUTRAL zone around {name}"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"inside CPR NEUTRAL zone around {name}"
                    })
                return  # Do not create horizontal band

        # Second pass: Check Ignore Buffer (ONLY if NOT inside any zone)
        # If swing is outside all zones but within buffer, ignore it
        for name, level in self.cpr_levels.items():
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]
            
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
            # Check distance to upper neutral zone lower edge
            if price < upper_neutral[0] and abs(price - upper_neutral[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} upper NEUTRAL zone lower edge {upper_neutral[0]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} upper NEUTRAL zone lower edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone lower edge"
                    })
                return
            
            # Check distance to upper neutral zone upper edge
            if price > upper_neutral[1] and abs(price - upper_neutral[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} upper NEUTRAL zone upper edge {upper_neutral[1]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} upper NEUTRAL zone upper edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bullish zone upper edge"
                    })
                return
            
            # Check distance to lower neutral zone lower edge
            if price < lower_neutral[0] and abs(price - lower_neutral[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} lower NEUTRAL zone lower edge {lower_neutral[0]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} lower NEUTRAL zone lower edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone lower edge"
                    })
                return
            
            # Check distance to lower neutral zone upper edge
            if price > lower_neutral[1] and abs(price - lower_neutral[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name} lower NEUTRAL zone upper edge {lower_neutral[1]:.2f})")
                # Track ignored swing
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} lower NEUTRAL zone upper edge"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price,
                        'timestamp': timestamp,
                        'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} bearish zone upper edge"
                    })
                return

        # If we reached here, create/merge horizontal band
        self._add_horizontal_band(price, swing_type, timestamp)
        return None

    def _add_horizontal_band(self, price, swing_type, timestamp=""):
        # If dynamic bands are disabled, do nothing
        if not self.enable_dynamic_horizontal_bands:
            return

        width = self.config['HORIZONTAL_BAND_WIDTH']
        tolerance = self.config['MERGE_TOLERANCE']
        
        target_list = self.horizontal_bands['resistance'] if swing_type == 'high' else self.horizontal_bands['support']
        band_type = 'RESISTANCE' if swing_type == 'high' else 'SUPPORT'
        
        # Try to merge
        for i, band in enumerate(target_list):
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
                new_band = [expanded_min, expanded_max]
                target_list[i] = new_band
                if self.verbose_swing_logging:
                    print(f"    -> MERGED with existing {band_type.lower()} band {i+1} @{timestamp}: [{old_band[0]:.2f}, {old_band[1]:.2f}] (center {center:.2f})")
                    print(f"       New merged band: [{new_band[0]:.2f}, {new_band[1]:.2f}] (center {new_center:.2f})")
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
        target_list.append(new_band)
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
        
        # Step 1: Check if calculated_price is INSIDE any CPR NEUTRAL zone
        # (Inside detection uses calculated_price only – consistent with v2 design)
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            if level_name not in self.cpr_levels:
                continue
            level = self.cpr_levels[level_name]
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]

            if upper_neutral[0] <= calculated_price <= upper_neutral[1] or \
               lower_neutral[0] <= calculated_price <= lower_neutral[1]:
                self.sentiment = 'NEUTRAL'
                sentiment_determined = True
                last_interaction = get_level_name_lower(level_name)
                last_interaction_type = 'CPR_NEUTRAL'
                break

        # Check horizontal bands - check all and track the last one
        # NEUTRAL only occurs from horizontal band interactions, not CPR bands
        if not sentiment_determined:
            for band in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                # Check if calculated_price is inside the band
                if band[0] <= calculated_price <= band[1]:
                    last_interaction = 'horizontal_band'
                    last_interaction_type = 'HORIZONTAL'
        
        # If inside horizontal band and sentiment not determined, set to NEUTRAL
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
                upper_neutral = [level, level + cpr_width]
                lower_neutral = [level - cpr_width, level]
                
                # Check if high or low touched a CPR band boundary
                touched = False
                # Upper band boundary: upper_neutral[1]
                # Lower band boundary: lower_neutral[0]
                if (low <= upper_neutral[1] <= high) or (low <= lower_neutral[0] <= high):
                    touched = True
                    last_touched_band = get_level_name_lower(level_name)
                    last_touched_type = 'CPR'
                    
                    # Check where close is relative to the band
                    # If close moves below lower neutral zone → BEARISH
                    if close < lower_neutral[0]:
                        self.sentiment = 'BEARISH'
                        touched_and_moved = True
                    # If close moves above upper neutral zone → BULLISH
                    elif close > upper_neutral[1]:
                        self.sentiment = 'BULLISH'
                        touched_and_moved = True
            
            # Check horizontal bands if not already determined
            if not touched_and_moved:
                for band in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
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
        
        # --- 1. Check CPR Band Touch-and-Move Rejection (Highest Priority) ---
        # High touching lower CPR NEUTRAL zone [level - w, level] with CLOSE below band → BEARISH
        # Low touching upper CPR NEUTRAL zone [level, level + w] with CLOSE above band → BULLISH
        for name, level in self.cpr_levels.items():
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]

            # High touches lower neutral zone -> potential BEARISH rejection
            if lower_neutral[0] <= high <= lower_neutral[1]:
                # Rejection only if candle CLOSES below the lower neutral boundary
                if candle['close'] < lower_neutral[0]:
                    self.sentiment = "BEARISH"
                    return

            # Low touches upper neutral zone -> potential BULLISH rejection
            if upper_neutral[0] <= low <= upper_neutral[1]:
                # Rejection only if candle CLOSES above the upper neutral boundary
                if candle['close'] > upper_neutral[1]:
                    self.sentiment = "BULLISH"
                    return
        
        # PRIORITY 2: Check if calculated_price is INSIDE any CPR NEUTRAL zone
        # (Inside detection uses calculated_price only)
        for name, level in self.cpr_levels.items():
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]

            if upper_neutral[0] <= calc_price <= upper_neutral[1] or \
               lower_neutral[0] <= calc_price <= lower_neutral[1]:
                self.sentiment = "NEUTRAL"
                return  # Found a match, exit early
        
        # PRIORITY 3: Check for breakout/breakdown (price crossed ABOVE or BELOW CPR NEUTRAL zones)
        # Use current_candle_index - 1 instead of -2 to respect reprocessing
        prev_idx = self.current_candle_index - 1
        if prev_idx >= 0 and prev_idx < len(self.candles):
            prev_calc_price = self.candles[prev_idx]['calculated_price']
        else:
            prev_calc_price = calc_price  # Fallback if no previous candle
        for name, level in self.cpr_levels.items():
            upper_neutral = [level, level + cpr_width]
            lower_neutral = [level - cpr_width, level]
            
            # E. Breakout: Price crosses ABOVE the upper NEUTRAL zone -> BULLISH
            # Using hybrid check: calculated_price OR high crosses above
            if (calc_price > upper_neutral[1] or high > upper_neutral[1]) and prev_calc_price <= upper_neutral[1]:
                self.sentiment = "BULLISH"
                return
            
            # F. Breakdown: Price crosses BELOW the lower NEUTRAL zone -> BEARISH
            # Using hybrid check: calculated_price OR low crosses below
            if (calc_price < lower_neutral[0] or low < lower_neutral[0]) and prev_calc_price >= lower_neutral[0]:
                self.sentiment = "BEARISH"
                return
        
        # PRIORITY 4: Check Horizontal Band Interactions (inside horizontal bands)
        # Only apply horizontal "inside = NEUTRAL" after giving CPR zones a chance
        all_horiz = self.horizontal_bands['resistance'] + self.horizontal_bands['support']
        for band in all_horiz:
            # Inside Band -> NEUTRAL (only if not already determined by CPR breakout/breakdown)
            if band[0] <= calc_price <= band[1]:
                self.sentiment = "NEUTRAL"
                return 

        # --- Check Horizontal Band Crosses (after CPR breakout/breakdown) ---
        # Check for horizontal band crosses (breakout/breakdown)
        for band in all_horiz:
            
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
        
        print("=" * 80)