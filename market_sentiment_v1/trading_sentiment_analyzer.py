"""
Trading Sentiment Analyzer
Analyzes 1-minute OHLC data to determine market sentiment (BULLISH, BEARISH, NEUTRAL)
based on CPR and dynamic Horizontal support/resistance bands.
"""

import yaml
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class TradingSentimentAnalyzer:
    """
    Analyzes 1-minute OHLC data to determine market sentiment (BULLISH, BEARISH, NEUTRAL)
    based on CPR and dynamic Horizontal support/resistance bands.
    """

    def __init__(self, config_path: str, cpr_levels: dict):
        """
        Initializes the analyzer with configuration settings and daily CPR levels.
        
        Args:
            config_path: Path to the YAML configuration file
            cpr_levels: Dictionary with CPR levels {'R4': value, 'R3': value, ..., 'S4': value}
        """
        # 1. Load configuration from the YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cpr_band_width = self.config.get('CPR_BAND_WIDTH', 5.0)
        self.horizontal_band_width = self.config.get('HORIZONTAL_BAND_WIDTH', 5.0)
        self.cpr_ignore_buffer = self.config.get('CPR_IGNORE_BUFFER', 5.0)
        self.cpr_pair_width_threshold = self.config.get('CPR_PAIR_WIDTH_THRESHOLD', 80.0)
        self.swing_confirmation_candles = self.config.get('SWING_CONFIRMATION_CANDLES', 2)
        self.merge_tolerance = self.config.get('MERGE_TOLERANCE', 10.0)
        self.enable_swing_midpoint_validation = self.config.get('ENABLE_SWING_MIDPOINT_VALIDATION', True)
        
        # 2. Initialize state variables
        self.sentiment = None
        self.candle_history: List[Dict] = []
        self.candle_timestamps: List[str] = []  # Track timestamps for each candle
        self.is_first_candle = True
        
        # 3. Initialize CPR bands and their bullish/bearish zones
        self._initialize_cpr_bands(cpr_levels)
        
        # 4. Initialize data structures for Horizontal Bands
        self.horizontal_bands: Dict[str, Dict[str, List]] = {}
        # Track default bands (initialized at start) vs dynamic bands (from swings)
        self.default_bands: Dict[str, Dict[str, List]] = {}
        
        # 5. Create the default 50% Horizontal Bands for wide CPR pairs
        self._initialize_horizontal_bands()
        
        # Store CPR levels for reference
        self.cpr_levels = cpr_levels
        
        # Logging
        self.sentiment_log: List[Tuple[str, str, str]] = []  # (timestamp, sentiment, reason)
        
        # Support/Resistance levels log file
        self.levels_log_file = 'output/support_resistance_levels.txt'
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.levels_log_file), exist_ok=True)
        
        # Log initial CPR levels and default horizontal bands
        self._log_support_resistance_levels("Initialization")

    def _initialize_cpr_bands(self, cpr_levels: dict):
        """Helper to create CPR band structures with their bullish/bearish zones."""
        self.cpr_bands = {}
        
        # Normalize keys to lowercase
        normalized_levels = {}
        for key, value in cpr_levels.items():
            normalized_levels[key.lower()] = value
        
        # Define all CPR levels in order
        level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
        
        for level_name in level_names:
            if level_name in normalized_levels:
                level_value = normalized_levels[level_name]
                # Bullish Zone: [level, level + CPR_BAND_WIDTH]
                # Bearish Zone: [level - CPR_BAND_WIDTH, level]
                self.cpr_bands[level_name] = {
                    'level': level_value,
                    'bullish_zone': [level_value, level_value + self.cpr_band_width],
                    'bearish_zone': [level_value - self.cpr_band_width, level_value]
                }

    def _initialize_horizontal_bands(self):
        """Helper to create default 50% horizontal bands."""
        # Define CPR pairs: (upper_level, lower_level, pair_name)
        level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
        level_values = {name: self.cpr_bands[name]['level'] for name in level_names if name in self.cpr_bands}
        
        pairs = []
        for i in range(len(level_names) - 1):
            upper_name = level_names[i]
            lower_name = level_names[i + 1]
            if upper_name in level_values and lower_name in level_values:
                pair_name = f"{upper_name}_{lower_name}"
                pairs.append((upper_name, lower_name, pair_name))
        
        # Initialize horizontal bands structure
        for _, _, pair_name in pairs:
            self.horizontal_bands[pair_name] = {
                'resistance': [],
                'support': []
            }
            # Initialize default bands tracking
            self.default_bands[pair_name] = {
                'resistance': [],
                'support': []
            }
        
        # Create default 50% horizontal bands for wide pairs
        for upper_name, lower_name, pair_name in pairs:
            upper_value = level_values[upper_name]
            lower_value = level_values[lower_name]
            distance = abs(upper_value - lower_value)
            
            if distance > self.cpr_pair_width_threshold:
                # Calculate midpoint
                midpoint = (upper_value + lower_value) / 2.0
                
                # Create default horizontal band
                default_band = [
                    midpoint - self.horizontal_band_width,
                    midpoint + self.horizontal_band_width
                ]
                
                # Add to both resistance and support lists
                self.horizontal_bands[pair_name]['resistance'].append(default_band)
                self.horizontal_bands[pair_name]['support'].append(default_band)
                # Track as default bands
                self.default_bands[pair_name]['resistance'].append(default_band)
                self.default_bands[pair_name]['support'].append(default_band)

    def process_new_candle(self, ohlc: dict, timestamp: str = None):
        """
        Processes a new 1-minute candle and updates the market sentiment.
        This is the main logic handler.

        Args:
            ohlc: A dictionary containing 'open', 'high', 'low', 'close' for the candle.
            timestamp: Optional timestamp string for logging
        """
        # 1. Add the new candle to self.candle_history
        self.candle_history.append(ohlc.copy())
        # Store timestamp for this candle
        self.candle_timestamps.append(timestamp if timestamp else "")
        
        # 2. If this is the first candle of the day, run the Initial Sentiment Logic
        if self.is_first_candle:
            self._run_initial_sentiment_logic(ohlc, timestamp)
            self.is_first_candle = False
        else:
            # 3. For all subsequent candles, run the Ongoing Sentiment Logic
            self._run_ongoing_sentiment_logic(ohlc, timestamp)
        
        # 4. After sentiment is updated, run the Swing Detection and Band Management logic
        self._detect_and_manage_swings()

    def _run_initial_sentiment_logic(self, ohlc: dict, timestamp: str = None):
        """Determines sentiment based on the very first candle of the day."""
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        open_price = ohlc['open']
        calculated_price = self._calculate_price(ohlc)
        
        # Track all band interactions (to use the last one)
        last_interaction = None
        last_interaction_type = None
        sentiment_determined = False
        
        # Check CPR bands - prioritize bearish/bullish zones
        # CPR bands (R4, R3, R2, R1, PIVOT): if calculated_price is in bearish zone → BEARISH
        cpr_levels_upper = ['r4', 'r3', 'r2', 'r1', 'pivot']
        for level_name in cpr_levels_upper:
            if level_name not in self.cpr_bands:
                continue
            band_data = self.cpr_bands[level_name]
            bearish_zone = band_data['bearish_zone']
            
            # Check if calculated_price is inside bearish zone of CPR band
            if bearish_zone[0] <= calculated_price <= bearish_zone[1]:
                self.sentiment = 'BEARISH'
                reason = f"First candle calculated_price inside CPR {level_name} bearish zone"
                sentiment_determined = True
                last_interaction = level_name
                last_interaction_type = 'CPR_BEARISH'
                break
        
        # CPR bands (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4): if calculated_price is in bullish zone → BULLISH
        # IMPORTANT: PIVOT must be included - it works exactly like any other CPR band
        if not sentiment_determined:
            # Check all CPR bands for bullish zones (including PIVOT)
            all_cpr_levels_bullish = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            for level_name in all_cpr_levels_bullish:
                if level_name not in self.cpr_bands:
                    continue
                band_data = self.cpr_bands[level_name]
                bullish_zone = band_data['bullish_zone']
                
                # Check if calculated_price is inside bullish zone of CPR band
                # Bullish zone: [level, level + CPR_BAND_WIDTH]
                # For PIVOT: [pivot, pivot + CPR_BAND_WIDTH]
                if bullish_zone[0] <= calculated_price <= bullish_zone[1]:
                    self.sentiment = 'BULLISH'
                    reason = f"First candle calculated_price inside CPR {level_name} bullish zone [{bullish_zone[0]:.2f}, {bullish_zone[1]:.2f}]"
                    sentiment_determined = True
                    last_interaction = level_name
                    last_interaction_type = 'CPR_BULLISH'
                    break
        
        # Check if high or low falls inside any CPR band zone (fallback check using raw OHLC)
        # This is a secondary check - if calculated_price didn't trigger, check raw values
        if not sentiment_determined:
            for level_name, band_data in self.cpr_bands.items():
                bullish_zone = band_data['bullish_zone']
                bearish_zone = band_data['bearish_zone']
                
                # Check if high/low is inside bearish zone → BEARISH
                if (bearish_zone[0] <= high <= bearish_zone[1] or 
                    bearish_zone[0] <= low <= bearish_zone[1]):
                    self.sentiment = 'BEARISH'
                    reason = f"First candle high/low inside CPR {level_name} bearish zone"
                    sentiment_determined = True
                    break
                # Check if high/low is inside bullish zone → BULLISH
                elif (bullish_zone[0] <= high <= bullish_zone[1] or 
                      bullish_zone[0] <= low <= bullish_zone[1]):
                    self.sentiment = 'BULLISH'
                    reason = f"First candle high/low inside CPR {level_name} bullish zone"
                    sentiment_determined = True
                    break
        
        # Check horizontal bands - check all and track the last one
        # NEUTRAL only occurs from horizontal band interactions, not CPR bands
        if not sentiment_determined:
            for pair_name, bands in self.horizontal_bands.items():
                for band in bands['resistance'] + bands['support']:
                    # Check if calculated_price is inside the band
                    if band[0] <= calculated_price <= band[1]:
                        last_interaction = pair_name
                        last_interaction_type = 'HORIZONTAL'
        
        # If inside horizontal band and sentiment not determined, set to NEUTRAL
        # Note: CPR band interactions always set BEARISH/BULLISH, never NEUTRAL
        if last_interaction and not sentiment_determined:
            self.sentiment = 'NEUTRAL'
            reason = f"First candle inside {last_interaction_type} band: {last_interaction}"
        else:
            # Check if price touched a band and moved away
            touched_and_moved = False
            last_touched_band = None
            last_touched_type = None
            
            # Check CPR bands - check all and track the last one
            for level_name, band_data in self.cpr_bands.items():
                bullish_zone = band_data['bullish_zone']
                bearish_zone = band_data['bearish_zone']
                
                # Check if high or low touched a band boundary
                touched = False
                if (low <= bullish_zone[1] <= high) or (low <= bearish_zone[0] <= high):
                    touched = True
                    last_touched_band = level_name
                    last_touched_type = 'CPR'
                    
                    # Check where close is relative to the band
                    if close < bearish_zone[0]:
                        self.sentiment = 'BEARISH'
                        reason = f"First candle touched CPR {level_name} and closed below"
                        touched_and_moved = True
                    elif close > bullish_zone[1]:
                        self.sentiment = 'BULLISH'
                        reason = f"First candle touched CPR {level_name} and closed above"
                        touched_and_moved = True
            
            # Check horizontal bands if not already determined
            if not touched_and_moved:
                for pair_name, bands in self.horizontal_bands.items():
                    for band in bands['resistance'] + bands['support']:
                        band_lower = band[0]
                        band_upper = band[1]
                        
                        # Check if price crossed the band completely
                        if (open_price < band_lower and close > band_upper) or (open_price > band_upper and close < band_lower):
                            last_touched_band = pair_name
                            last_touched_type = 'HORIZONTAL'
                            
                            if close < band_lower:
                                self.sentiment = 'BEARISH'
                                reason = f"First candle crossed horizontal band {pair_name} and closed below"
                                touched_and_moved = True
                                break
                            elif close > band_upper:
                                self.sentiment = 'BULLISH'
                                reason = f"First candle crossed horizontal band {pair_name} and closed above"
                                touched_and_moved = True
                                break
                    if touched_and_moved:
                        break
            
            # If still not determined, use pivot-based logic
            if not touched_and_moved:
                pivot_value = self.cpr_bands['pivot']['level']
                if low > pivot_value:
                    self.sentiment = 'BULLISH'
                    reason = "First candle low above Pivot"
                elif high < pivot_value:
                    self.sentiment = 'BEARISH'
                    reason = "First candle high below Pivot"
                else:
                    self.sentiment = 'NEUTRAL'
                    reason = "First candle around Pivot"
        
        # Log the sentiment change
        if timestamp:
            self.sentiment_log.append((timestamp, self.sentiment, reason))

    def _run_ongoing_sentiment_logic(self, ohlc: dict, timestamp: str = None):
        """Updates sentiment based on the current state and new candle data."""
        if self.sentiment is None:
            return
        
        # When NEUTRAL: Check horizontal bands FIRST (they represent immediate support/resistance)
        # For BULLISH/BEARISH: Check CPR bands first (immediate flips)
        if self.sentiment == 'NEUTRAL':
            # Check horizontal bands first - they take priority when NEUTRAL
            self._check_horizontal_band_interaction(ohlc, timestamp)
            # If horizontal bands didn't change sentiment, then check CPR bands
            if self.sentiment == 'NEUTRAL':
                self._check_cpr_band_interaction(ohlc, timestamp)
        else:
            # For BULLISH/BEARISH: Check CPR bands first (immediate flips)
            cpr_changed = self._check_cpr_band_interaction(ohlc, timestamp)
            
            if not cpr_changed:
                # Check horizontal band interactions
                self._check_horizontal_band_interaction(ohlc, timestamp)

    def _check_cpr_band_interaction(self, ohlc: dict, timestamp: str = None) -> bool:
        """
        Checks for price interaction with CPR bands using Hybrid Price Check.
        Uses both raw OHLC values and calculated_price for robustness.
        """
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        calculated_price = self._calculate_price(ohlc)
        
        if self.sentiment == 'BULLISH':
            # First, check if price rises above bullish zone upper bound - maintain BULLISH
            # Check CPR bands (S1, S2, S3, S4) - if price rises above bullish zone, stay BULLISH
            cpr_levels_lower = ['s1', 's2', 's3', 's4']
            for level_name in cpr_levels_lower:
                if level_name not in self.cpr_bands:
                    continue
                band_data = self.cpr_bands[level_name]
                bullish_zone = band_data['bullish_zone']
                bullish_zone_upper = bullish_zone[1]  # level + CPR_BAND_WIDTH
                
                # If calculated_price or high rises above bullish zone upper bound, maintain BULLISH
                if calculated_price > bullish_zone_upper or high > bullish_zone_upper:
                    # Explicitly maintain BULLISH (already BULLISH, but log for clarity)
                    if timestamp:
                        trigger_point = 'high' if high > bullish_zone_upper else 'calculated_price'
                        reason = f"BULLISH maintained: {trigger_point} above CPR {level_name} bullish zone upper bound"
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    # Don't return True - continue to check for reversals
            
            # Hybrid Price Check: Check BOTH high AND calculated_price against bearish zone
            # Check ALL CPR bands for bearish zones
            # This is important because high can touch bearish zone of any CPR band
            
            # Check ALL CPR bands (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4) for bearish zones
            all_cpr_levels = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            for level_name in all_cpr_levels:
                if level_name not in self.cpr_bands:
                    continue
                    
                band_data = self.cpr_bands[level_name]
                bearish_zone = band_data['bearish_zone']
                
                # Check if high OR calculated_price touches the bearish zone
                if (bearish_zone[0] <= high <= bearish_zone[1] or 
                    bearish_zone[0] <= calculated_price <= bearish_zone[1]):
                    self.sentiment = 'BEARISH'
                    trigger_point = 'high' if bearish_zone[0] <= high <= bearish_zone[1] else 'calculated_price'
                    reason = f"BULLISH -> BEARISH: {trigger_point} touched CPR {level_name} bearish zone"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
        
        elif self.sentiment == 'BEARISH':
            # First, check if price falls below bearish zone lower bound - maintain BEARISH
            # Check CPR bands (R4, R3, R2, R1) - if price falls below bearish zone, stay BEARISH
            cpr_levels_upper = ['r4', 'r3', 'r2', 'r1']
            for level_name in cpr_levels_upper:
                if level_name not in self.cpr_bands:
                    continue
                band_data = self.cpr_bands[level_name]
                bearish_zone = band_data['bearish_zone']
                bearish_zone_lower = bearish_zone[0]  # level - CPR_BAND_WIDTH
                
                # If calculated_price or low falls below bearish zone lower bound, maintain BEARISH
                if calculated_price < bearish_zone_lower or low < bearish_zone_lower:
                    # Explicitly maintain BEARISH (already BEARISH, but log for clarity)
                    if timestamp:
                        trigger_point = 'low' if low < bearish_zone_lower else 'calculated_price'
                        reason = f"BEARISH maintained: {trigger_point} below CPR {level_name} bearish zone lower bound"
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    # Don't return True - continue to check for reversals
            
            # Hybrid Price Check: Check BOTH low AND calculated_price against bullish zone
            # Check ALL CPR bands for bullish zones
            # This is important because low can touch bullish zone of any CPR band
            
            # Check ALL CPR bands (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4) for bullish zones
            # IMPORTANT: PIVOT must be included - it works exactly like any other CPR band
            all_cpr_levels = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            for level_name in all_cpr_levels:
                if level_name not in self.cpr_bands:
                    # Skip if band not initialized (shouldn't happen, but safety check)
                    continue
                band_data = self.cpr_bands[level_name]
                bullish_zone = band_data['bullish_zone']
                
                # Check if low OR calculated_price touches the bullish zone
                # Bullish zone: [level, level + CPR_BAND_WIDTH]
                # For PIVOT: [pivot, pivot + CPR_BAND_WIDTH]
                low_in_zone = bullish_zone[0] <= low <= bullish_zone[1]
                calc_in_zone = bullish_zone[0] <= calculated_price <= bullish_zone[1]
                
                if low_in_zone or calc_in_zone:
                    self.sentiment = 'BULLISH'
                    trigger_point = 'low' if low_in_zone else 'calculated_price'
                    reason = f"BEARISH -> BULLISH: {trigger_point} touched CPR {level_name} bullish zone [{bullish_zone[0]:.2f}, {bullish_zone[1]:.2f}]"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
        
        elif self.sentiment == 'NEUTRAL':
            # When NEUTRAL: Check for transitions to BULLISH or BEARISH based on CPR band interactions
            # Check ALL CPR bands for bullish zones (to flip to BULLISH)
            all_levels = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            for level_name in all_levels:
                if level_name not in self.cpr_bands:
                    continue
                band_data = self.cpr_bands[level_name]
                bullish_zone = band_data['bullish_zone']
                
                # Check if low OR calculated_price touches the bullish zone
                if (bullish_zone[0] <= low <= bullish_zone[1] or 
                    bullish_zone[0] <= calculated_price <= bullish_zone[1]):
                    self.sentiment = 'BULLISH'
                    trigger_point = 'low' if bullish_zone[0] <= low <= bullish_zone[1] else 'calculated_price'
                    reason = f"NEUTRAL -> BULLISH: {trigger_point} touched CPR {level_name} bullish zone"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
            
            # Check ALL CPR bands for bearish zones (to flip to BEARISH)
            for level_name in all_levels:
                if level_name not in self.cpr_bands:
                    continue
                band_data = self.cpr_bands[level_name]
                bearish_zone = band_data['bearish_zone']
                
                # Check if high OR calculated_price touches the bearish zone
                if (bearish_zone[0] <= high <= bearish_zone[1] or 
                    bearish_zone[0] <= calculated_price <= bearish_zone[1]):
                    self.sentiment = 'BEARISH'
                    trigger_point = 'high' if bearish_zone[0] <= high <= bearish_zone[1] else 'calculated_price'
                    reason = f"NEUTRAL -> BEARISH: {trigger_point} touched CPR {level_name} bearish zone"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
        
        # Check for price jumping over CPR bands
        if len(self.candle_history) >= 2:
            prev_ohlc = self.candle_history[-2]
            prev_calculated = self._calculate_price(prev_ohlc)
            curr_calculated = self._calculate_price(ohlc)
            
            # Check if price jumped over any CPR band
            for level_name, band_data in self.cpr_bands.items():
                level_value = band_data['level']
                bullish_zone = band_data['bullish_zone']
                bearish_zone = band_data['bearish_zone']
                
                # Check if jumped from below to above
                if prev_calculated < level_value and curr_calculated > bullish_zone[1]:
                    self.sentiment = 'BULLISH'
                    reason = f"Price jumped over CPR {level_name}, ended above bullish zone"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
                # Check if jumped from above to below
                elif prev_calculated > level_value and curr_calculated < bearish_zone[0]:
                    self.sentiment = 'BEARISH'
                    reason = f"Price jumped over CPR {level_name}, ended below bearish zone"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return True
        
        return False

    def _check_horizontal_band_interaction(self, ohlc: dict, timestamp: str = None):
        """Checks for price interaction with Horizontal bands and updates sentiment."""
        calculated_price = self._calculate_price(ohlc)
        open_price = ohlc['open']
        close = ohlc['close']
        
        # Determine which CPR pair the price is in
        current_pair = self._get_current_cpr_pair(calculated_price)
        if not current_pair:
            return
        
        # Debug: Print current horizontal bands for this pair
        if timestamp and (timestamp.endswith(':18:00') or timestamp.endswith(':19:00') or timestamp.endswith(':20:00') or timestamp.endswith(':21:00')):
            print(f"\n[DEBUG {timestamp}] Checking horizontal bands in {current_pair}:")
            print(f"  Calculated price: {calculated_price:.2f}")
            print(f"  Current sentiment: {self.sentiment}")
            print(f"  Support bands: {len(self.horizontal_bands[current_pair]['support'])}")
            for i, band in enumerate(self.horizontal_bands[current_pair]['support']):
                print(f"    Support {i+1}: [{band[0]:.2f}, {band[1]:.2f}]")
            print(f"  Resistance bands: {len(self.horizontal_bands[current_pair]['resistance'])}")
            for i, band in enumerate(self.horizontal_bands[current_pair]['resistance']):
                print(f"    Resistance {i+1}: [{band[0]:.2f}, {band[1]:.2f}]")
        
        if self.sentiment == 'BULLISH':
            # When BULLISH: Check resistance bands (for entering NEUTRAL or breaking below)
            # ALSO check support bands (for breaking below support → BEARISH)
            
            # First, check resistance bands
            for band in self.horizontal_bands[current_pair]['resistance']:
                band_lower = band[0]
                band_upper = band[1]
                
                # Check if price enters the band
                if band_lower <= calculated_price <= band_upper:
                    # Only set NEUTRAL if this is a swing high band, not a default band
                    # Check if band is in default_bands list
                    is_default = False
                    if current_pair in self.default_bands:
                        for default_band in self.default_bands[current_pair]['resistance']:
                            if abs(band_lower - default_band[0]) < 0.01 and abs(band_upper - default_band[1]) < 0.01:
                                is_default = True
                                break
                    
                    # Only set NEUTRAL for swing high bands, not default bands
                    if not is_default:
                        self.sentiment = 'NEUTRAL'
                        reason = f"BULLISH -> NEUTRAL: Entered horizontal resistance band in {current_pair}"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                
                # Check if price moves below the resistance band
                if calculated_price < band_lower:
                    # Check if this is a direct cross (open above, close below)
                    if open_price > band_upper and close < band_lower:
                        self.sentiment = 'BEARISH'
                        reason = f"BULLISH -> BEARISH: Crossed horizontal resistance band {current_pair} completely below"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                    # Or if price was previously above or inside the band and now below
                    elif len(self.candle_history) >= 2:
                        prev_ohlc = self.candle_history[-2]
                        prev_calculated = self._calculate_price(prev_ohlc)
                        # If previous price was above or inside the band, and now below
                        if prev_calculated >= band_lower:
                            self.sentiment = 'BEARISH'
                            reason = f"BULLISH -> BEARISH: Moved below horizontal resistance band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                            if timestamp:
                                self.sentiment_log.append((timestamp, self.sentiment, reason))
                            return
                    # If no previous candle, but we're BULLISH and price is below resistance, change to BEARISH
                    else:
                        self.sentiment = 'BEARISH'
                        reason = f"BULLISH -> BEARISH: Price below horizontal resistance band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                
                # Exception: Large candle crossing completely (open below, close above)
                if open_price < band_lower and close > band_upper:
                    # Stay BULLISH - broke above resistance
                    pass
            
            # Second, check support bands - if price moves below support, change to BEARISH
            # BUT: Don't override if price is inside a CPR bullish zone (CPR takes priority)
            for band in self.horizontal_bands[current_pair]['support']:
                band_lower = band[0]
                band_upper = band[1]
                
                # Check if price enters the support band
                if band_lower <= calculated_price <= band_upper:
                    # Only set NEUTRAL if this is a swing low band, not a default band
                    # Check if band is in default_bands list
                    is_default = False
                    if current_pair in self.default_bands:
                        for default_band in self.default_bands[current_pair]['support']:
                            if abs(band_lower - default_band[0]) < 0.01 and abs(band_upper - default_band[1]) < 0.01:
                                is_default = True
                                break
                    
                    # Only set NEUTRAL for swing low bands, not default bands
                    if not is_default:
                        self.sentiment = 'NEUTRAL'
                        reason = f"BULLISH -> NEUTRAL: Entered horizontal support band in {current_pair}"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                
                # Check if price moves below the support band
                if calculated_price < band_lower:
                    # IMPORTANT: Check if price is inside OR ABOVE a CPR bullish zone first
                    # If it is, don't override BULLISH to BEARISH (CPR takes priority)
                    inside_or_above_bullish_zone = False
                    for level_name in ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']:
                        if level_name in self.cpr_bands:
                            bullish_zone = self.cpr_bands[level_name]['bullish_zone']
                            # Check if price is inside OR above the bullish zone
                            if calculated_price >= bullish_zone[0]:
                                inside_or_above_bullish_zone = True
                                break
                    
                    # If price is inside or above a CPR bullish zone, don't override BULLISH
                    if inside_or_above_bullish_zone:
                        # Price is in or above bullish zone, keep BULLISH (CPR takes priority)
                        continue
                    
                    # Check if this is a direct cross (open above, close below)
                    if open_price > band_upper and close < band_lower:
                        self.sentiment = 'BEARISH'
                        reason = f"BULLISH -> BEARISH: Crossed horizontal support band {current_pair} completely below"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                    # Or if price was previously above or inside the band and now below
                    elif len(self.candle_history) >= 2:
                        prev_ohlc = self.candle_history[-2]
                        prev_calculated = self._calculate_price(prev_ohlc)
                        # If previous price was above or inside the band, and now below
                        if prev_calculated >= band_lower:
                            self.sentiment = 'BEARISH'
                            reason = f"BULLISH -> BEARISH: Moved below horizontal support band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                            if timestamp:
                                self.sentiment_log.append((timestamp, self.sentiment, reason))
                            return
                        # If previous price was also below, but we're BULLISH and price is below support, change to BEARISH
                        # This handles the case where sentiment was incorrectly set to BULLISH while price was below support
                        else:
                            self.sentiment = 'BEARISH'
                            reason = f"BULLISH -> BEARISH: Price below horizontal support band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                            if timestamp:
                                self.sentiment_log.append((timestamp, self.sentiment, reason))
                            return
                    # If no previous candle, but we're BULLISH and price is below support, change to BEARISH
                    else:
                        self.sentiment = 'BEARISH'
                        reason = f"BULLISH -> BEARISH: Price below horizontal support band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
        
        elif self.sentiment == 'BEARISH':
            # When BEARISH: Check support bands (for entering NEUTRAL or breaking above)
            # ALSO check resistance bands (for breaking above resistance → BULLISH)
            
            # First, check support bands
            for band in self.horizontal_bands[current_pair]['support']:
                band_lower = band[0]
                band_upper = band[1]
                
                # Check if price enters the band
                if band_lower <= calculated_price <= band_upper:
                    self.sentiment = 'NEUTRAL'
                    reason = f"BEARISH -> NEUTRAL: Entered horizontal support band in {current_pair}"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return
                
                # Exception: Large candle crossing completely
                if (open_price < band_lower and close > band_upper) or (open_price > band_upper and close < band_lower):
                    if close < band_lower:
                        # Stay BEARISH - broke below support
                        pass
                    elif close > band_upper:
                        self.sentiment = 'BULLISH'
                        reason = f"BEARISH -> BULLISH: Crossed horizontal support band {current_pair} completely above"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
            
            # Second, check resistance bands - if price moves above resistance, change to BULLISH
            for band in self.horizontal_bands[current_pair]['resistance']:
                band_lower = band[0]
                band_upper = band[1]
                
                # Check if price enters the resistance band
                if band_lower <= calculated_price <= band_upper:
                    # Only set NEUTRAL if this is a swing high band, not a default band
                    # Check if band is in default_bands list
                    is_default = False
                    if current_pair in self.default_bands:
                        for default_band in self.default_bands[current_pair]['resistance']:
                            if abs(band_lower - default_band[0]) < 0.01 and abs(band_upper - default_band[1]) < 0.01:
                                is_default = True
                                break
                    
                    # Only set NEUTRAL for swing high bands, not default bands
                    if not is_default:
                        self.sentiment = 'NEUTRAL'
                        reason = f"BEARISH -> NEUTRAL: Entered horizontal resistance band in {current_pair}"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                
                # Check if price moves above the resistance band
                if calculated_price > band_upper:
                    # Check if this is a direct cross (open below, close above)
                    if open_price < band_lower and close > band_upper:
                        self.sentiment = 'BULLISH'
                        reason = f"BEARISH -> BULLISH: Crossed horizontal resistance band {current_pair} completely above"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
                    # Or if price was previously below or inside the band and now above
                    elif len(self.candle_history) >= 2:
                        prev_ohlc = self.candle_history[-2]
                        prev_calculated = self._calculate_price(prev_ohlc)
                        # If previous price was below or inside the band, and now above
                        if prev_calculated <= band_upper:
                            self.sentiment = 'BULLISH'
                            reason = f"BEARISH -> BULLISH: Moved above horizontal resistance band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                            if timestamp:
                                self.sentiment_log.append((timestamp, self.sentiment, reason))
                            return
                    # If no previous candle or previous was already above, but we're BEARISH and price is above resistance, change to BULLISH
                    else:
                        self.sentiment = 'BULLISH'
                        reason = f"BEARISH -> BULLISH: Price above horizontal resistance band in {current_pair} (band: [{band_lower:.2f}, {band_upper:.2f}])"
                        if timestamp:
                            self.sentiment_log.append((timestamp, self.sentiment, reason))
                        return
        
        elif self.sentiment == 'NEUTRAL':
            # When NEUTRAL: Check if price moved above support bands or below resistance bands
            # This handles the bounce/rejection logic correctly
            
            # First, check if price is still inside any band
            all_bands = (self.horizontal_bands[current_pair]['resistance'] + 
                        self.horizontal_bands[current_pair]['support'])
            
            inside_any_band = False
            for band in all_bands:
                band_lower = band[0]
                band_upper = band[1]
                if band_lower <= calculated_price <= band_upper:
                    inside_any_band = True
                    break
            
            if inside_any_band:
                # Still in NEUTRAL - price is inside at least one band
                return
            
            # Price is not inside any band - check which type of band it moved from
            # IMPORTANT: Check BOTH support and resistance bands
            # If price is both above support AND below resistance, determine which band we were in
            # by checking which band the price is closer to
            
            # First, check if price is above any support band (bounce → BULLISH)
            above_support = False
            support_band_bounced = None
            closest_support_distance = float('inf')
            for band in self.horizontal_bands[current_pair]['support']:
                band_lower = band[0]
                band_upper = band[1]
                
                if calculated_price > band_upper:
                    # Calculate distance to this support band (distance to upper edge)
                    distance = calculated_price - band_upper
                    if distance < closest_support_distance:
                        above_support = True
                        support_band_bounced = band
                        closest_support_distance = distance
            
            # Second, check if price is below any resistance band (rejection → BEARISH)
            below_resistance = False
            resistance_band_rejected = None
            closest_resistance_distance = float('inf')
            for band in self.horizontal_bands[current_pair]['resistance']:
                band_lower = band[0]
                band_upper = band[1]
                
                if calculated_price < band_lower:
                    # Calculate distance to this resistance band (distance to lower edge)
                    distance = band_lower - calculated_price
                    if distance < closest_resistance_distance:
                        below_resistance = True
                        resistance_band_rejected = band
                        closest_resistance_distance = distance
            
            # If price is both above support AND below resistance, check which is closer
            # The closer band is the one we were most likely in
            if above_support and below_resistance:
                # Price is between bands - check which is closer
                if closest_support_distance < closest_resistance_distance:
                    # Closer to support band - we were in support, bounced → BULLISH
                    self.sentiment = 'BULLISH'
                    reason = f"NEUTRAL -> BULLISH: Bounced above support band in {current_pair} (band: [{support_band_bounced[0]:.2f}, {support_band_bounced[1]:.2f}])"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return
                else:
                    # Closer to resistance band - we were in resistance, rejected → BEARISH
                    self.sentiment = 'BEARISH'
                    reason = f"NEUTRAL -> BEARISH: Rejected below resistance band in {current_pair} (band: [{resistance_band_rejected[0]:.2f}, {resistance_band_rejected[1]:.2f}])"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return
            elif above_support:
                # Price bounced above support → BULLISH
                self.sentiment = 'BULLISH'
                reason = f"NEUTRAL -> BULLISH: Bounced above support band in {current_pair} (band: [{support_band_bounced[0]:.2f}, {support_band_bounced[1]:.2f}])"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            elif below_resistance:
                # Price rejected below resistance → BEARISH
                self.sentiment = 'BEARISH'
                reason = f"NEUTRAL -> BEARISH: Rejected below resistance band in {current_pair} (band: [{resistance_band_rejected[0]:.2f}, {resistance_band_rejected[1]:.2f}])"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            
            # If neither above support nor below resistance, check break scenarios
            # Check if price broke above highest resistance band
            highest_resistance_band = None
            for band in self.horizontal_bands[current_pair]['resistance']:
                if highest_resistance_band is None or band[1] > highest_resistance_band[1]:
                    highest_resistance_band = band
            
            if highest_resistance_band and calculated_price > highest_resistance_band[1]:
                # Price is above the highest resistance band → BULLISH
                self.sentiment = 'BULLISH'
                reason = f"NEUTRAL -> BULLISH: Broke above highest resistance band in {current_pair} (band: [{highest_resistance_band[0]:.2f}, {highest_resistance_band[1]:.2f}])"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            
            # Check if price broke below lowest support band
            lowest_support_band = None
            for band in self.horizontal_bands[current_pair]['support']:
                if lowest_support_band is None or band[0] < lowest_support_band[0]:
                    lowest_support_band = band
            
            if lowest_support_band and calculated_price < lowest_support_band[0]:
                # Price is below the lowest support band → BEARISH
                self.sentiment = 'BEARISH'
                reason = f"NEUTRAL -> BEARISH: Broke below lowest support band in {current_pair} (band: [{lowest_support_band[0]:.2f}, {lowest_support_band[1]:.2f}])"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            
            # Price is between bands - check if above all or below all
            above_all = True
            below_all = True
            
            for band in all_bands:
                band_lower = band[0]
                band_upper = band[1]
                
                if calculated_price <= band_upper:
                    above_all = False
                if calculated_price >= band_lower:
                    below_all = False
            
            # If price is above all bands, go BULLISH
            if above_all:
                self.sentiment = 'BULLISH'
                reason = f"NEUTRAL -> BULLISH: Moved above all horizontal bands in {current_pair}"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            
            # If price is below all bands, go BEARISH
            if below_all:
                self.sentiment = 'BEARISH'
                reason = f"NEUTRAL -> BEARISH: Moved below all horizontal bands in {current_pair}"
                if timestamp:
                    self.sentiment_log.append((timestamp, self.sentiment, reason))
                return
            
            # Price is between bands - check the closest band
            # Find the band that was previously active (if any)
            # For simplicity, check if price moved above or below the nearest band
            # This is a simplified approach - in practice, we'd track which band we were in
            for band in all_bands:
                band_lower = band[0]
                band_upper = band[1]
                
                # Check if price moved above this band's upper zone
                if calculated_price > band_upper:
                    self.sentiment = 'BULLISH'
                    reason = f"NEUTRAL -> BULLISH: Moved above horizontal band in {current_pair}"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return
                
                # Check if price moved below this band's lower zone
                if calculated_price < band_lower:
                    self.sentiment = 'BEARISH'
                    reason = f"NEUTRAL -> BEARISH: Moved below horizontal band in {current_pair}"
                    if timestamp:
                        self.sentiment_log.append((timestamp, self.sentiment, reason))
                    return

    def _get_current_cpr_pair(self, price: float) -> Optional[str]:
        """Determines which CPR pair the price falls into."""
        level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
        level_values = [(name, self.cpr_bands[name]['level']) for name in level_names if name in self.cpr_bands]
        level_values.sort(key=lambda x: x[1], reverse=True)  # Sort descending
        
        # Find the pair the price is between
        for i in range(len(level_values) - 1):
            upper_name, upper_value = level_values[i]
            lower_name, lower_value = level_values[i + 1]
            
            if lower_value <= price <= upper_value:
                pair_name = f"{upper_name}_{lower_name}"
                if pair_name in self.horizontal_bands:
                    return pair_name
        
        return None

    def _detect_and_manage_swings(self):
        """Detects new swing highs/lows and manages the horizontal band list."""
        n = self.swing_confirmation_candles
        
        # Need at least 2*n + 1 candles to confirm a swing
        if len(self.candle_history) < 2 * n + 1:
            return
        
        # Check for swing at index (len - n - 1) - this is the swing candidate
        swing_idx = len(self.candle_history) - n - 1
        candidate_candle = self.candle_history[swing_idx]
        # Get timestamp of the swing candle
        swing_timestamp = self.candle_timestamps[swing_idx] if swing_idx < len(self.candle_timestamps) else ""
        # Format timestamp for display (extract time part if available)
        if swing_timestamp:
            try:
                # Try to extract time part from timestamp string
                # Handle formats like: "2025-10-23 13:06:00+05:30" or "2025-10-23 13:06:00"
                if ' ' in swing_timestamp:
                    # Split by space and take the time part (before timezone offset)
                    time_part = swing_timestamp.split(' ')[1]
                    # Remove timezone offset if present (+05:30 or -05:30)
                    if '+' in time_part:
                        time_part = time_part.split('+')[0]
                    elif '-' in time_part and time_part.count('-') > 1:  # Has timezone offset
                        # Split by '-' and take first two parts (HH:MM:SS)
                        parts = time_part.split('-')
                        if len(parts) >= 2:
                            time_part = '-'.join(parts[:-1])  # Everything except timezone
                    # Extract just HH:MM (remove seconds for cleaner display)
                    if ':' in time_part:
                        time_parts = time_part.split(':')
                        if len(time_parts) >= 2:
                            time_part = f"{time_parts[0]}:{time_parts[1]}"  # HH:MM format
                else:
                    time_part = swing_timestamp
            except Exception as e:
                # Fallback to original timestamp if parsing fails
                time_part = swing_timestamp
        else:
            time_part = "N/A"
        
        # Check for swing high using calculated_price (not raw high)
        is_swing_high = True
        candidate_calculated = self._calculate_price(candidate_candle)
        for i in range(max(0, swing_idx - n), swing_idx):
            prev_calculated = self._calculate_price(self.candle_history[i])
            if prev_calculated >= candidate_calculated:
                is_swing_high = False
                break
        if is_swing_high:
            for i in range(swing_idx + 1, min(len(self.candle_history), swing_idx + n + 1)):
                next_calculated = self._calculate_price(self.candle_history[i])
                if next_calculated >= candidate_calculated:
                    is_swing_high = False
                    break
        
        # Check for swing low using calculated_price (not raw low)
        is_swing_low = True
        # candidate_calculated already calculated above (same for both high and low)
        for i in range(max(0, swing_idx - n), swing_idx):
            prev_calculated = self._calculate_price(self.candle_history[i])
            if prev_calculated <= candidate_calculated:
                is_swing_low = False
                break
        if is_swing_low:
            for i in range(swing_idx + 1, min(len(self.candle_history), swing_idx + n + 1)):
                next_calculated = self._calculate_price(self.candle_history[i])
                if next_calculated <= candidate_calculated:
                    is_swing_low = False
                    break
        
        # Process swing high
        if is_swing_high:
            swing_price = candidate_calculated  # Use calculated_price instead of high
            # Validate swing: ignore if inside or within CPR_IGNORE_BUFFER of CPR bands
            if not self._is_swing_valid(swing_price):
                print(f"  [SWING HIGH @{time_part}] Price {swing_price:.2f} - IGNORED (within CPR_IGNORE_BUFFER)")
                return
            
            # Determine which CPR pair it falls into
            current_pair = self._get_current_cpr_pair(swing_price)
            if not current_pair:
                print(f"  [SWING HIGH @{time_part}] Price {swing_price:.2f} - IGNORED (no CPR pair found)")
                return
            
            # Check swing high validation using percentage-based zones from CPR boundaries
            level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            pair_levels = current_pair.split('_')
            if len(pair_levels) == 2:
                upper_level = self.cpr_bands[pair_levels[0]]['level']
                lower_level = self.cpr_bands[pair_levels[1]]['level']
                pair_size = abs(upper_level - lower_level)
                
                # Apply percentage-based validation only if enabled
                if self.enable_swing_midpoint_validation:
                    # Determine percentage threshold based on pair size
                    if pair_size < self.cpr_pair_width_threshold:
                        # Small pairs: use 50% threshold
                        percentage = 0.50
                    else:
                        # Large pairs: use 25% threshold
                        percentage = 0.25
                    
                    # Calculate swing high threshold: upper_level - (percentage * pair_size)
                    swing_high_threshold = upper_level - (percentage * pair_size)
                    
                    if swing_price > swing_high_threshold:
                        # Valid swing high - add to resistance bands
                        print(f"  [SWING HIGH @{time_part}] Price {swing_price:.2f} in {current_pair} (above {percentage*100:.0f}% threshold {swing_high_threshold:.2f}, pair_size: {pair_size:.2f}) -> Adding to RESISTANCE")
                        self._add_or_merge_swing_band(current_pair, 'resistance', swing_price, time_part)
                    else:
                        print(f"  [SWING HIGH @{time_part}] Price {swing_price:.2f} in {current_pair} - IGNORED (below {percentage*100:.0f}% threshold {swing_high_threshold:.2f}, pair_size: {pair_size:.2f})")
                else:
                    # Validation disabled - accept all valid swings
                    print(f"  [SWING HIGH @{time_part}] Price {swing_price:.2f} in {current_pair} (validation disabled) -> Adding to RESISTANCE")
                    self._add_or_merge_swing_band(current_pair, 'resistance', swing_price, time_part)
        
        # Process swing low
        if is_swing_low:
            swing_price = candidate_calculated  # Use calculated_price instead of low
            # Validate swing: ignore if inside or within CPR_IGNORE_BUFFER of CPR bands
            if not self._is_swing_valid(swing_price):
                print(f"  [SWING LOW @{time_part}] Price {swing_price:.2f} - IGNORED (within CPR_IGNORE_BUFFER)")
                return
            
            # Determine which CPR pair it falls into
            current_pair = self._get_current_cpr_pair(swing_price)
            if not current_pair:
                print(f"  [SWING LOW @{time_part}] Price {swing_price:.2f} - IGNORED (no CPR pair found)")
                return
            
            # Check swing low validation using percentage-based zones from CPR boundaries
            level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
            pair_levels = current_pair.split('_')
            if len(pair_levels) == 2:
                upper_level = self.cpr_bands[pair_levels[0]]['level']
                lower_level = self.cpr_bands[pair_levels[1]]['level']
                pair_size = abs(upper_level - lower_level)
                
                # Apply percentage-based validation only if enabled
                if self.enable_swing_midpoint_validation:
                    # Determine percentage threshold based on pair size
                    if pair_size < self.cpr_pair_width_threshold:
                        # Small pairs: use 50% threshold
                        percentage = 0.50
                    else:
                        # Large pairs: use 25% threshold
                        percentage = 0.25
                    
                    # Calculate swing low threshold: lower_level + (percentage * pair_size)
                    swing_low_threshold = lower_level + (percentage * pair_size)
                    
                    if swing_price < swing_low_threshold:
                        # Valid swing low - add to support bands
                        print(f"  [SWING LOW @{time_part}] Price {swing_price:.2f} in {current_pair} (below {percentage*100:.0f}% threshold {swing_low_threshold:.2f}, pair_size: {pair_size:.2f}) -> Adding to SUPPORT")
                        self._add_or_merge_swing_band(current_pair, 'support', swing_price, time_part)
                    else:
                        print(f"  [SWING LOW @{time_part}] Price {swing_price:.2f} in {current_pair} - IGNORED (above {percentage*100:.0f}% threshold {swing_low_threshold:.2f}, pair_size: {pair_size:.2f})")
                else:
                    # Validation disabled - accept all valid swings
                    print(f"  [SWING LOW @{time_part}] Price {swing_price:.2f} in {current_pair} (validation disabled) -> Adding to SUPPORT")
                    self._add_or_merge_swing_band(current_pair, 'support', swing_price, time_part)

    def _is_swing_valid(self, swing_price: float) -> bool:
        """Checks if a swing point is valid (not within CPR_IGNORE_BUFFER of CPR bands)."""
        for level_name, band_data in self.cpr_bands.items():
            level_value = band_data['level']
            bullish_zone = band_data['bullish_zone']
            bearish_zone = band_data['bearish_zone']
            
            # Check if swing is inside any zone
            if (bullish_zone[0] <= swing_price <= bullish_zone[1] or
                bearish_zone[0] <= swing_price <= bearish_zone[1]):
                return False
            
            # Check if swing is within CPR_IGNORE_BUFFER of the level
            if abs(swing_price - level_value) <= self.cpr_ignore_buffer:
                return False
            
            # Check if swing is within CPR_IGNORE_BUFFER of zone boundaries
            # This catches swings that are very close to zone edges even if not inside
            distance_to_bullish_lower = abs(swing_price - bullish_zone[0])
            distance_to_bullish_upper = abs(swing_price - bullish_zone[1])
            distance_to_bearish_lower = abs(swing_price - bearish_zone[0])
            distance_to_bearish_upper = abs(swing_price - bearish_zone[1])
            
            if (distance_to_bullish_lower <= self.cpr_ignore_buffer or
                distance_to_bullish_upper <= self.cpr_ignore_buffer or
                distance_to_bearish_lower <= self.cpr_ignore_buffer or
                distance_to_bearish_upper <= self.cpr_ignore_buffer):
                return False
        
        return True

    def _add_or_merge_swing_band(self, pair_name: str, band_type: str, swing_price: float, timestamp: str = ""):
        """Adds a new swing band or merges with existing one if within tolerance."""
        bands_list = self.horizontal_bands[pair_name][band_type]
        
        # Check if there's an existing band within merge tolerance
        merged = False
        for i, band in enumerate(bands_list):
            band_center = (band[0] + band[1]) / 2.0
            if abs(swing_price - band_center) <= self.merge_tolerance:
                # Merge: create new band centered between swing_price and old_band_center
                old_band = band.copy()
                new_center = (swing_price + band_center) / 2.0
                new_band = [
                    new_center - self.horizontal_band_width,
                    new_center + self.horizontal_band_width
                ]
                bands_list[i] = new_band
                merged = True
                time_str = f"@{timestamp} " if timestamp else ""
                print(f"    -> MERGED with existing {band_type} band {i+1} {time_str}: [{old_band[0]:.2f}, {old_band[1]:.2f}] (center {band_center:.2f})")
                print(f"       New merged band: [{new_band[0]:.2f}, {new_band[1]:.2f}] (center {new_center:.2f})")
                # Log the update
                self._log_support_resistance_levels(f"Swing {band_type} merged @{timestamp}")
                break
        
        if not merged:
            # Create new band
            new_band = [
                swing_price - self.horizontal_band_width,
                swing_price + self.horizontal_band_width
            ]
            bands_list.append(new_band)
            time_str = f"@{timestamp} " if timestamp else ""
            print(f"    -> CREATED new {band_type} band {time_str}: [{new_band[0]:.2f}, {new_band[1]:.2f}] (center {swing_price:.2f})")
            # Log the update
            self._log_support_resistance_levels(f"Swing {band_type} created @{timestamp}")

    def _calculate_price(self, ohlc: dict) -> float:
        """Calculates the special price for horizontal band analysis."""
        return ((ohlc['low'] + ohlc['close']) / 2 + (ohlc['high'] + ohlc['open']) / 2) / 2
    
    def _log_support_resistance_levels(self, event_description: str = ""):
        """
        Logs all CPR levels, horizontal support/resistance bands, and swing highs/lows to a file.
        This helps debug why sentiment didn't change in areas of interest.
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.levels_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {timestamp}\n")
                if event_description:
                    f.write(f"Event: {event_description}\n")
                f.write(f"{'='*80}\n\n")
                
                # 1. CPR Levels
                f.write("CPR LEVELS:\n")
                f.write("-" * 80 + "\n")
                level_names = ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']
                for level_name in level_names:
                    if level_name in self.cpr_bands:
                        band_data = self.cpr_bands[level_name]
                        level_value = band_data['level']
                        bullish_zone = band_data['bullish_zone']
                        bearish_zone = band_data['bearish_zone']
                        f.write(f"  {level_name.upper():6s}: Level={level_value:8.2f} | "
                               f"Bullish Zone=[{bullish_zone[0]:8.2f}, {bullish_zone[1]:8.2f}] | "
                               f"Bearish Zone=[{bearish_zone[0]:8.2f}, {bearish_zone[1]:8.2f}]\n")
                f.write("\n")
                
                # 2. Horizontal Support/Resistance Bands (grouped by CPR pair)
                f.write("HORIZONTAL SUPPORT/RESISTANCE BANDS:\n")
                f.write("-" * 80 + "\n")
                
                # Sort pairs by their upper level (descending)
                pairs_with_levels = []
                for pair_name in self.horizontal_bands.keys():
                    pair_levels = pair_name.split('_')
                    if len(pair_levels) == 2 and pair_levels[0] in self.cpr_bands:
                        upper_level = self.cpr_bands[pair_levels[0]]['level']
                        pairs_with_levels.append((upper_level, pair_name))
                pairs_with_levels.sort(reverse=True)
                
                for upper_level, pair_name in pairs_with_levels:
                    f.write(f"\n  CPR Pair: {pair_name.upper()}\n")
                    
                    # Support bands (swing lows)
                    support_bands = self.horizontal_bands[pair_name]['support']
                    if support_bands:
                        f.write(f"    Support Bands ({len(support_bands)}):\n")
                        for i, band in enumerate(support_bands, 1):
                            center = (band[0] + band[1]) / 2.0
                            is_default = False
                            if pair_name in self.default_bands:
                                is_default = band in self.default_bands[pair_name]['support']
                            band_type = "DEFAULT" if is_default else "SWING LOW"
                            f.write(f"      {i}. [{band[0]:8.2f}, {band[1]:8.2f}] (center: {center:8.2f}) - {band_type}\n")
                    else:
                        f.write(f"    Support Bands: None\n")
                    
                    # Resistance bands (swing highs)
                    resistance_bands = self.horizontal_bands[pair_name]['resistance']
                    if resistance_bands:
                        f.write(f"    Resistance Bands ({len(resistance_bands)}):\n")
                        for i, band in enumerate(resistance_bands, 1):
                            center = (band[0] + band[1]) / 2.0
                            is_default = False
                            if pair_name in self.default_bands:
                                is_default = band in self.default_bands[pair_name]['resistance']
                            band_type = "DEFAULT" if is_default else "SWING HIGH"
                            f.write(f"      {i}. [{band[0]:8.2f}, {band[1]:8.2f}] (center: {center:8.2f}) - {band_type}\n")
                    else:
                        f.write(f"    Resistance Bands: None\n")
                
                f.write(f"\n{'='*80}\n\n")
                
        except Exception as e:
            # Don't fail if logging fails - just print error
            print(f"Error logging support/resistance levels: {e}")
    
    def print_horizontal_bands_summary(self):
        """Print a summary of all horizontal bands grouped by CPR pair."""
        print("\n" + "=" * 80)
        print("HORIZONTAL BANDS SUMMARY BY CPR PAIR")
        print("=" * 80)
        
        # Helper function to check if a band is in default bands
        def is_default_band(band, pair_name, band_type):
            """Check if a band is a default initialized band."""
            if pair_name not in self.default_bands:
                return False
            default_bands_list = self.default_bands[pair_name][band_type]
            for default_band in default_bands_list:
                if abs(band[0] - default_band[0]) < 0.01 and abs(band[1] - default_band[1]) < 0.01:
                    return True
            return False
        
        # Separate default and dynamic bands
        default_summary = {}
        dynamic_summary = {}
        
        for pair_name in sorted(self.horizontal_bands.keys()):
            support_bands = self.horizontal_bands[pair_name]['support']
            resistance_bands = self.horizontal_bands[pair_name]['resistance']
            
            default_support = []
            default_resistance = []
            dynamic_support = []
            dynamic_resistance = []
            
            # Separate support bands
            for band in support_bands:
                if is_default_band(band, pair_name, 'support'):
                    default_support.append(band)
                else:
                    dynamic_support.append(band)
            
            # Separate resistance bands
            for band in resistance_bands:
                if is_default_band(band, pair_name, 'resistance'):
                    default_resistance.append(band)
                else:
                    dynamic_resistance.append(band)
            
            if default_support or default_resistance:
                default_summary[pair_name] = {
                    'support': default_support,
                    'resistance': default_resistance
                }
            
            if dynamic_support or dynamic_resistance:
                dynamic_summary[pair_name] = {
                    'support': dynamic_support,
                    'resistance': dynamic_resistance
                }
        
        # Print Table 1: Default Initialized 50% CPR Pair Bands
        print("\n" + "-" * 80)
        print("TABLE 1: DEFAULT INITIALIZED 50% CPR PAIR HORIZONTAL BANDS")
        print("-" * 80)
        
        if default_summary:
            for pair_name in sorted(default_summary.keys()):
                support_bands = default_summary[pair_name]['support']
                resistance_bands = default_summary[pair_name]['resistance']
                
                print(f"\n{pair_name.upper()}:")
                
                if support_bands:
                    print(f"  Support bands ({len(support_bands)}):")
                    for i, band in enumerate(support_bands):
                        center = (band[0] + band[1]) / 2.0
                        print(f"    {i+1}. [{band[0]:.2f}, {band[1]:.2f}] (center: {center:.2f}) [DEFAULT]")
                else:
                    print(f"  Support bands: None")
                
                if resistance_bands:
                    print(f"  Resistance bands ({len(resistance_bands)}):")
                    for i, band in enumerate(resistance_bands):
                        center = (band[0] + band[1]) / 2.0
                        print(f"    {i+1}. [{band[0]:.2f}, {band[1]:.2f}] (center: {center:.2f}) [DEFAULT]")
                else:
                    print(f"  Resistance bands: None")
        else:
            print("\n  No default initialized bands found.")
        
        # Print Table 2: Dynamically Identified Swing High/Low Bands
        print("\n" + "-" * 80)
        print("TABLE 2: DYNAMICALLY IDENTIFIED SWING HIGH/LOW BANDS")
        print("-" * 80)
        
        if dynamic_summary:
            for pair_name in sorted(dynamic_summary.keys()):
                support_bands = dynamic_summary[pair_name]['support']
                resistance_bands = dynamic_summary[pair_name]['resistance']
                
                print(f"\n{pair_name.upper()}:")
                
                if support_bands:
                    print(f"  Support bands ({len(support_bands)}) [from Swing Lows]:")
                    for i, band in enumerate(support_bands):
                        center = (band[0] + band[1]) / 2.0
                        print(f"    {i+1}. [{band[0]:.2f}, {band[1]:.2f}] (center: {center:.2f}) [SWING LOW]")
                else:
                    print(f"  Support bands: None")
                
                if resistance_bands:
                    print(f"  Resistance bands ({len(resistance_bands)}) [from Swing Highs]:")
                    for i, band in enumerate(resistance_bands):
                        center = (band[0] + band[1]) / 2.0
                        print(f"    {i+1}. [{band[0]:.2f}, {band[1]:.2f}] (center: {center:.2f}) [SWING HIGH]")
                else:
                    print(f"  Resistance bands: None")
        else:
            print("\n  No dynamically identified swing bands found.")
        
        print("\n" + "=" * 80)

    def get_current_sentiment(self) -> str:
        """Returns the current sentiment."""
        return self.sentiment

    def get_calculated_price(self, ohlc: dict) -> float:
        """Returns the calculated price for a given OHLC."""
        return self._calculate_price(ohlc)

