"""
Refactored Trading Sentiment Analyzer - Proof of Concept

This is a proof-of-concept refactored version that demonstrates:
1. Separated concerns (spatial, temporal, state)
2. Explicit rules instead of implicit priority order
3. Composable, testable components
4. Clear dependencies

The goal is to show how the architecture can be improved while preserving
the same behavior as the original implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Sentiment(Enum):
    """Sentiment states"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class Band:
    """Represents a price band (CPR or horizontal)"""
    lower: float
    upper: float
    band_type: str  # 'cpr_bullish', 'cpr_bearish', 'cpr_neutralized', 'horizontal'
    level_name: Optional[str] = None  # For CPR bands: 'R1', 'PIVOT', etc.
    is_neutralized: bool = False


@dataclass
class SpatialAnalysis:
    """Spatial relationships: where is price relative to bands?"""
    inside_cpr_bullish: List[Band] = None
    inside_cpr_bearish: List[Band] = None
    inside_cpr_neutralized: List[Band] = None
    inside_horizontal: List[Band] = None
    above_bands: List[Band] = None
    below_bands: List[Band] = None
    touching_cpr_bullish: List[Band] = None  # Low touches bullish zone
    touching_cpr_bearish: List[Band] = None  # High touches bearish zone
    calc_price_touching_cpr_bullish: List[Band] = None  # Calculated_price touches bullish zone (for state-dependent reversal)
    
    def __post_init__(self):
        if self.inside_cpr_bullish is None:
            self.inside_cpr_bullish = []
        if self.inside_cpr_bearish is None:
            self.inside_cpr_bearish = []
        if self.inside_cpr_neutralized is None:
            self.inside_cpr_neutralized = []
        if self.inside_horizontal is None:
            self.inside_horizontal = []
        if self.above_bands is None:
            self.above_bands = []
        if self.below_bands is None:
            self.below_bands = []
        if self.touching_cpr_bullish is None:
            self.touching_cpr_bullish = []
        if self.touching_cpr_bearish is None:
            self.touching_cpr_bearish = []
        if self.calc_price_touching_cpr_bullish is None:
            self.calc_price_touching_cpr_bullish = []


@dataclass
class TemporalAnalysis:
    """Temporal relationships: how did price move relative to bands?"""
    crossed_above_cpr: List[Band] = None
    crossed_below_cpr: List[Band] = None
    crossed_above_horizontal: List[Band] = None
    crossed_below_horizontal: List[Band] = None
    implicit_cpr_pair_change: Optional[str] = None  # 'up', 'down', or None
    
    def __post_init__(self):
        if self.crossed_above_cpr is None:
            self.crossed_above_cpr = []
        if self.crossed_below_cpr is None:
            self.crossed_below_cpr = []
        if self.crossed_above_horizontal is None:
            self.crossed_above_horizontal = []
        if self.crossed_below_horizontal is None:
            self.crossed_below_horizontal = []


class SpatialAnalyzer:
    """Analyzes where price is relative to all bands"""
    
    def __init__(self, cpr_levels: dict, cpr_band_states: dict, horizontal_bands: dict, cpr_width: float, current_candle_index: int = -1):
        self.cpr_levels = cpr_levels
        self.cpr_band_states = cpr_band_states
        self.horizontal_bands = horizontal_bands
        self.cpr_width = cpr_width
        self.current_candle_index = current_candle_index
    
    def analyze(self, candle: dict) -> SpatialAnalysis:
        """Analyze spatial relationships for a candle"""
        calc_price = candle['calculated_price']
        high = candle['high']
        low = candle['low']
        
        analysis = SpatialAnalysis()
        
        # Analyze CPR bands
        for name, level in self.cpr_levels.items():
            state = self.cpr_band_states[name]
            bull_zone = Band(level, level + self.cpr_width, 'cpr_bullish', name)
            bear_zone = Band(level - self.cpr_width, level, 'cpr_bearish', name)
            
            # Check if neutralized
            is_bullish_neutralized = state.get('bullish_neutralized', False)
            is_bearish_neutralized = state.get('bearish_neutralized', False)
            
            # Bullish zone
            if bull_zone.lower <= calc_price <= bull_zone.upper:
                # Check if neutralized AND current candle is at or after neutralization point
                if is_bullish_neutralized and self.current_candle_index >= state.get('bullish_neutralized_at', -1):
                    bull_zone.band_type = 'cpr_neutralized'
                    bull_zone.is_neutralized = True
                    analysis.inside_cpr_neutralized.append(bull_zone)
                else:
                    analysis.inside_cpr_bullish.append(bull_zone)
            
            # Bearish zone
            if bear_zone.lower <= calc_price <= bear_zone.upper:
                # Check if neutralized AND current candle is at or after neutralization point
                if is_bearish_neutralized and self.current_candle_index >= state.get('bearish_neutralized_at', -1):
                    bear_zone.band_type = 'cpr_neutralized'
                    bear_zone.is_neutralized = True
                    analysis.inside_cpr_neutralized.append(bear_zone)
                else:
                    analysis.inside_cpr_bearish.append(bear_zone)
            
            # Touch detection (using raw high/low)
            if bull_zone.lower <= low <= bull_zone.upper:
                analysis.touching_cpr_bullish.append(bull_zone)
            if bear_zone.lower <= high <= bear_zone.upper:
                analysis.touching_cpr_bearish.append(bear_zone)
            
            # Calculated_price touch detection (for state-dependent reversal checks)
            # When BEARISH, check if calculated_price touches bullish zone for reversal
            if bull_zone.lower <= calc_price <= bull_zone.upper:
                # Store the band with neutralization info for decision engine
                calc_touch_band = Band(bull_zone.lower, bull_zone.upper, 'cpr_bullish', name)
                calc_touch_band.is_neutralized = (is_bullish_neutralized and 
                                                  self.current_candle_index >= state.get('bullish_neutralized_at', -1))
                analysis.calc_price_touching_cpr_bullish.append(calc_touch_band)
            
            # Above/below
            if calc_price > bull_zone.upper:
                analysis.above_bands.append(bull_zone)
            if calc_price < bear_zone.lower:
                analysis.below_bands.append(bear_zone)
        
        # Analyze horizontal bands
        for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
            if isinstance(band_entry, dict):
                band_values = band_entry['band']
            else:
                band_values = band_entry
            
            horizontal_band = Band(band_values[0], band_values[1], 'horizontal')
            
            if horizontal_band.lower <= calc_price <= horizontal_band.upper:
                analysis.inside_horizontal.append(horizontal_band)
            elif calc_price > horizontal_band.upper:
                analysis.above_bands.append(horizontal_band)
            elif calc_price < horizontal_band.lower:
                analysis.below_bands.append(horizontal_band)
        
        return analysis


class TemporalAnalyzer:
    """Analyzes how price moved relative to bands"""
    
    def __init__(self, cpr_levels: dict, horizontal_bands: dict, cpr_width: float):
        self.cpr_levels = cpr_levels
        self.horizontal_bands = horizontal_bands
        self.cpr_width = cpr_width
    
    def _get_cpr_pair(self, price):
        """Helper to find which CPR pair the price is in"""
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
            if lower <= price <= upper:
                return f"{levels[i][0]}_{levels[i+1][0]}"
        return None
    
    def analyze(self, candle: dict, previous_candle: Optional[dict]) -> TemporalAnalysis:
        """Analyze temporal relationships (crossings)"""
        analysis = TemporalAnalysis()
        
        if previous_candle is None:
            return analysis  # No previous candle, can't detect crossings
        
        calc_price = candle['calculated_price']
        high = candle['high']
        low = candle['low']
        prev_calc_price = previous_candle['calculated_price']
        
        # Analyze CPR crossings
        for name, level in self.cpr_levels.items():
            bull_zone_upper = level + self.cpr_width
            bear_zone_lower = level - self.cpr_width
            
            # Crossed above bullish zone
            if (calc_price > bull_zone_upper or high > bull_zone_upper) and prev_calc_price <= bull_zone_upper:
                analysis.crossed_above_cpr.append(Band(level, bull_zone_upper, 'cpr_bullish', name))
            
            # Crossed below bearish zone
            if (calc_price < bear_zone_lower or low < bear_zone_lower) and prev_calc_price >= bear_zone_lower:
                analysis.crossed_below_cpr.append(Band(bear_zone_lower, level, 'cpr_bearish', name))
        
        # Analyze horizontal band crossings
        for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
            if isinstance(band_entry, dict):
                band_values = band_entry['band']
            else:
                band_values = band_entry
            
            band_lower = band_values[0]
            band_upper = band_values[1]
            
            if prev_calc_price <= band_upper and calc_price > band_upper:
                analysis.crossed_above_horizontal.append(Band(band_lower, band_upper, 'horizontal'))
            if prev_calc_price >= band_lower and calc_price < band_lower:
                analysis.crossed_below_horizontal.append(Band(band_lower, band_upper, 'horizontal'))
        
        # Implicit Crossing (Gap/Jump Logic) - detect CPR pair changes
        curr_pair = self._get_cpr_pair(calc_price)
        prev_pair = self._get_cpr_pair(prev_calc_price)
        
        if curr_pair and prev_pair and curr_pair != prev_pair:
            # Determine direction of pairs (R4 is highest, S4 is lowest)
            order = ['R4_R3', 'R3_R2', 'R2_R1', 'R1_PIVOT', 'PIVOT_S1', 'S1_S2', 'S2_S3', 'S3_S4']
            try:
                curr_idx = order.index(curr_pair)
                prev_idx = order.index(prev_pair)
                
                # Lower index = Higher Price (R4 is top)
                if curr_idx < prev_idx:  # Moved Up
                    analysis.implicit_cpr_pair_change = 'up'
                elif curr_idx > prev_idx:  # Moved Down
                    analysis.implicit_cpr_pair_change = 'down'
            except ValueError:
                pass  # Pair not in standard list
        
        return analysis


@dataclass
class SentimentRule:
    """A rule for determining sentiment"""
    name: str
    condition: callable
    action: callable
    priority: int
    description: str = ""


class SentimentDecisionEngine:
    """Composes analyses into final sentiment using explicit rules"""
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> List[SentimentRule]:
        """Build the rule set for ongoing sentiment logic - explicit, testable, composable
        
        Priority order matches original _run_ongoing_sentiment_logic exactly:
        1. PRIORITY 1: State-dependent reversal checks (BEARISH → BULLISH, BULLISH → BEARISH)
        2. PRIORITY 1 (second pass): All CPR touch checks
        3. PRIORITY 2: CPR inside (bullish zones first, then bearish)
        4. PRIORITY 3: CPR breakout/breakdown
        5. PRIORITY 4: Horizontal band inside
        6. Horizontal band crosses
        7. Implicit crossing (gap/jump logic)
        """
        rules = []
        
        # PRIORITY 1: State-dependent reversal checks
        # When BEARISH, check for reversal to BULLISH: low OR calculated_price touches bullish zone
        rules.append(SentimentRule(
            name="Reversal BEARISH→BULLISH (Low Touch)",
            condition=lambda spatial, temporal, current: (
                current == Sentiment.BEARISH and len(spatial.touching_cpr_bullish) > 0
            ),
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=1,
            description="When BEARISH: Low touches bullish zone → BULLISH (reversal)"
        ))
        
        rules.append(SentimentRule(
            name="Reversal BEARISH→BULLISH (CalcPrice Touch)",
            condition=lambda spatial, temporal, current: (
                current == Sentiment.BEARISH and len(spatial.calc_price_touching_cpr_bullish) > 0 and
                not any(b.is_neutralized for b in spatial.calc_price_touching_cpr_bullish)
            ),
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=1,
            description="When BEARISH: Calculated_price touches bullish zone → BULLISH (reversal, unless neutralized)"
        ))
        
        rules.append(SentimentRule(
            name="Reversal BEARISH→NEUTRAL (CalcPrice Touch Neutralized)",
            condition=lambda spatial, temporal, current: (
                current == Sentiment.BEARISH and len(spatial.calc_price_touching_cpr_bullish) > 0 and
                any(b.is_neutralized for b in spatial.calc_price_touching_cpr_bullish)
            ),
            action=lambda s, t, c: Sentiment.NEUTRAL,
            priority=1,
            description="When BEARISH: Calculated_price touches neutralized bullish zone → NEUTRAL"
        ))
        
        # When BULLISH, check for reversal to BEARISH: high touches bearish zone
        rules.append(SentimentRule(
            name="Reversal BULLISH→BEARISH (High Touch)",
            condition=lambda spatial, temporal, current: (
                current == Sentiment.BULLISH and len(spatial.touching_cpr_bearish) > 0
            ),
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=1,
            description="When BULLISH: High touches bearish zone → BEARISH (reversal)"
        ))
        
        # PRIORITY 1 (second pass): All CPR touch checks (for NEUTRAL sentiment or if no reversal detected)
        rules.append(SentimentRule(
            name="CPR Touch Bearish (All)",
            condition=lambda spatial, temporal, current: len(spatial.touching_cpr_bearish) > 0,
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=1,
            description="High touches CPR bearish zone → BEARISH"
        ))
        
        rules.append(SentimentRule(
            name="CPR Touch Bullish (All)",
            condition=lambda spatial, temporal, current: len(spatial.touching_cpr_bullish) > 0,
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=1,
            description="Low touches CPR bullish zone → BULLISH"
        ))
        
        # PRIORITY 2: CPR inside (bullish zones first, then bearish)
        rules.append(SentimentRule(
            name="CPR Inside Bullish",
            condition=lambda spatial, temporal, current: len(spatial.inside_cpr_bullish) > 0,
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=2,
            description="Price inside CPR bullish zone → BULLISH"
        ))
        
        rules.append(SentimentRule(
            name="CPR Inside Bearish",
            condition=lambda spatial, temporal, current: len(spatial.inside_cpr_bearish) > 0,
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=2,
            description="Price inside CPR bearish zone → BEARISH"
        ))
        
        rules.append(SentimentRule(
            name="CPR Inside Neutralized",
            condition=lambda spatial, temporal, current: len(spatial.inside_cpr_neutralized) > 0,
            action=lambda s, t, c: Sentiment.NEUTRAL,
            priority=2,
            description="Price inside neutralized CPR zone → NEUTRAL"
        ))
        
        # PRIORITY 3: CPR breakout/breakdown
        rules.append(SentimentRule(
            name="CPR Breakout",
            condition=lambda spatial, temporal, current: len(temporal.crossed_above_cpr) > 0,
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=3,
            description="Price crossed above CPR zone → BULLISH"
        ))
        
        rules.append(SentimentRule(
            name="CPR Breakdown",
            condition=lambda spatial, temporal, current: len(temporal.crossed_below_cpr) > 0,
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=3,
            description="Price crossed below CPR zone → BEARISH"
        ))
        
        # PRIORITY 4: Horizontal band inside
        rules.append(SentimentRule(
            name="Horizontal Band Inside",
            condition=lambda spatial, temporal, current: len(spatial.inside_horizontal) > 0,
            action=lambda s, t, c: Sentiment.NEUTRAL,
            priority=4,
            description="Price inside horizontal band → NEUTRAL"
        ))
        
        # Horizontal band crosses (after CPR breakout/breakdown and inside checks)
        rules.append(SentimentRule(
            name="Horizontal Band Breakout",
            condition=lambda spatial, temporal, current: len(temporal.crossed_above_horizontal) > 0,
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=5,
            description="Price crossed above horizontal band → BULLISH"
        ))
        
        rules.append(SentimentRule(
            name="Horizontal Band Breakdown",
            condition=lambda spatial, temporal, current: len(temporal.crossed_below_horizontal) > 0,
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=5,
            description="Price crossed below horizontal band → BEARISH"
        ))
        
        # Price above horizontal band (not crossing, just above) - for support bands
        # This comes BEFORE "Price Below" so that when price is above MID band, it wins
        rules.append(SentimentRule(
            name="Price Above Horizontal Band",
            condition=lambda spatial, temporal, current: (
                len(spatial.above_bands) > 0 and 
                any(b.band_type == 'horizontal' for b in spatial.above_bands) and
                len(spatial.inside_horizontal) == 0  # Not inside any horizontal band
            ),
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=5,
            description="Price above horizontal band → BULLISH"
        ))
        
        # Price below horizontal band (not crossing, just below) - for resistance bands
        rules.append(SentimentRule(
            name="Price Below Horizontal Band",
            condition=lambda spatial, temporal, current: (
                len(spatial.below_bands) > 0 and 
                any(b.band_type == 'horizontal' for b in spatial.below_bands) and
                len(spatial.inside_horizontal) == 0  # Not inside any horizontal band
            ),
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=5,
            description="Price below horizontal band → BEARISH"
        ))
        
        # Implicit Crossing (Gap/Jump Logic)
        rules.append(SentimentRule(
            name="Implicit CPR Pair Change Up",
            condition=lambda spatial, temporal, current: temporal.implicit_cpr_pair_change == 'up',
            action=lambda s, t, c: Sentiment.BULLISH,
            priority=6,
            description="Price jumped to higher CPR pair → BULLISH"
        ))
        
        rules.append(SentimentRule(
            name="Implicit CPR Pair Change Down",
            condition=lambda spatial, temporal, current: temporal.implicit_cpr_pair_change == 'down',
            action=lambda s, t, c: Sentiment.BEARISH,
            priority=6,
            description="Price jumped to lower CPR pair → BEARISH"
        ))
        
        # Sort by priority
        rules.sort(key=lambda r: r.priority)
        return rules
    
    def determine_sentiment(
        self, 
        spatial: SpatialAnalysis, 
        temporal: TemporalAnalysis, 
        current_sentiment: Sentiment
    ) -> Sentiment:
        """Apply rules in priority order to determine sentiment"""
        for rule in self.rules:
            if rule.condition(spatial, temporal, current_sentiment):
                result = rule.action(spatial, temporal, current_sentiment)
                # Action should always return Sentiment
                return result
        
        # Fallback: maintain current sentiment
        return current_sentiment


class TradingSentimentAnalyzerRefactored:
    """
    Refactored version of TradingSentimentAnalyzer
    
    This version demonstrates:
    - Separated concerns (spatial, temporal, state)
    - Explicit rules instead of implicit priority order
    - Composable, testable components
    - Clear dependencies
    """
    
    def __init__(self, config: dict, cpr_levels: dict):
        self.config = config
        self.cpr_levels = cpr_levels
        
        # State Variables
        self.sentiment = Sentiment.NEUTRAL
        self.candles = []
        self.current_candle_index = -1
        
        # Band Storage
        self.horizontal_bands = {
            'resistance': [],
            'support': []
        }

        # Feature flags
        self.enable_dynamic_swing_bands = self.config.get('ENABLE_DYNAMIC_SWING_BANDS', True)
        self.enable_default_cpr_mid_bands = self.config.get('ENABLE_DEFAULT_CPR_MID_BANDS', True)

        # CPR Band State (Tracks Neutralization)
        self.cpr_band_states = self._init_cpr_states()
        
        # Initialize Default Horizontal Bands
        self._init_default_horizontal_bands()
        
        # Analyzers (will be updated with current_candle_index in process_new_candle)
        self.spatial_analyzer = SpatialAnalyzer(
            self.cpr_levels, 
            self.cpr_band_states, 
            self.horizontal_bands,
            self.config['CPR_BAND_WIDTH'],
            self.current_candle_index
        )
        self.temporal_analyzer = TemporalAnalyzer(
            self.cpr_levels,
            self.horizontal_bands,
            self.config['CPR_BAND_WIDTH']
        )
        self.decision_engine = SentimentDecisionEngine()
        
        # Swing Detection Logging
        self.verbose_swing_logging = self.config.get('VERBOSE_SWING_LOGGING', False)
        self.detected_swing_highs = []
        self.detected_swing_lows = []
        self.sentiment_results = []

    def _init_cpr_states(self):
        """Initialize the neutralization state for all CPR levels."""
        states = {}
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            states[level_name] = {
                'bullish_neutralized': False,
                'bearish_neutralized': False,
                'bullish_neutralized_at': -1,
                'bearish_neutralized_at': -1
            }
        return states
    
    def _init_default_horizontal_bands(self):
        """Create 50% bands if CPR pair width > Threshold (and feature enabled)."""
        if not self.enable_default_cpr_mid_bands:
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
                self.horizontal_bands['resistance'].append({'band': band, 'timestamp': None})
                self.horizontal_bands['support'].append({'band': band, 'timestamp': None})

    def process_new_candle(self, candle: dict) -> dict:
        """Main entry point. Processes a single new candle."""
        self.candles.append(candle)
        self.current_candle_index += 1
        
        # Calculate Price
        calc_price = ((candle['low'] + candle['close']) / 2 + 
                      (candle['high'] + candle['open']) / 2) / 2
        candle['calculated_price'] = calc_price

        # Detect Swings (Delayed by N candles)
        neutralization_occurred = self._process_delayed_swings()

        # Get previous candle for temporal analysis
        prev_candle = None
        if self.current_candle_index > 0:
            prev_candle = self.candles[self.current_candle_index - 1]
        
        # Update spatial analyzer with current index and latest bands
        self.spatial_analyzer.current_candle_index = self.current_candle_index
        self.spatial_analyzer.horizontal_bands = self.horizontal_bands  # Update bands reference
        
        # Analyze spatial relationships
        spatial = self.spatial_analyzer.analyze(candle)
        
        # Analyze temporal relationships
        temporal = self.temporal_analyzer.analyze(candle, prev_candle)
        
        # Determine sentiment
        if self.current_candle_index == 0:
            # First candle uses initial logic
            self._run_initial_sentiment_logic(candle)
        else:
            # Ongoing logic - matches original priority order
            self._run_ongoing_sentiment_logic(candle, spatial, temporal)
        
        # Store result
        result = {
            'date': candle['date'],
            'sentiment': self.sentiment.value,
            'calculated_price': calc_price,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close']
        }
        self.sentiment_results.append(result)
        
        # Reprocess after neutralization if needed
        if neutralization_occurred:
            self._reprocess_after_neutralization(neutralization_occurred)

        return result

    def _run_initial_sentiment_logic(self, candle):
        """Initial sentiment logic for first candle - matches original priority order"""
        high = candle['high']
        low = candle['low']
        calculated_price = candle['calculated_price']
        close = candle['close']
        open_price = candle['open']
        cpr_width = self.config['CPR_BAND_WIDTH']
        
        sentiment_determined = False
        
        # PRIORITY 1: Check calculated_price inside CPR Bearish Zones (all levels)
        all_cpr_levels = ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']
        for level_name in all_cpr_levels:
            if level_name not in self.cpr_levels:
                continue
            level = self.cpr_levels[level_name]
            state = self.cpr_band_states[level_name]
            bearish_zone = [level - cpr_width, level]
            
            if bearish_zone[0] <= calculated_price <= bearish_zone[1]:
                if state['bearish_neutralized'] and self.current_candle_index >= state.get('bearish_neutralized_at', -1):
                    self.sentiment = Sentiment.NEUTRAL
                else:
                    self.sentiment = Sentiment.BEARISH
                sentiment_determined = True
                break
        
        # PRIORITY 2: Check calculated_price inside CPR Bullish Zones
        if not sentiment_determined:
            all_cpr_levels_bullish = ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']
            for level_name in all_cpr_levels_bullish:
                if level_name not in self.cpr_levels:
                    continue
                level = self.cpr_levels[level_name]
                state = self.cpr_band_states[level_name]
                bullish_zone = [level, level + cpr_width]
                
                if bullish_zone[0] <= calculated_price <= bullish_zone[1]:
                    if state['bullish_neutralized'] and self.current_candle_index >= state.get('bullish_neutralized_at', -1):
                        self.sentiment = Sentiment.NEUTRAL
                    else:
                        self.sentiment = Sentiment.BULLISH
                    sentiment_determined = True
                    break
        
        # PRIORITY 3: Fallback - Check raw high/low inside CPR zones
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
                        self.sentiment = Sentiment.NEUTRAL
                    else:
                        self.sentiment = Sentiment.BEARISH
                    sentiment_determined = True
                    break
                # Check if high/low is inside bullish zone → BULLISH
                elif (bullish_zone[0] <= high <= bullish_zone[1] or 
                      bullish_zone[0] <= low <= bullish_zone[1]):
                    if state['bullish_neutralized']:
                        self.sentiment = Sentiment.NEUTRAL
                    else:
                        self.sentiment = Sentiment.BULLISH
                    sentiment_determined = True
                    break
        
        # PRIORITY 4: Check calculated_price inside Horizontal Bands
        last_interaction = None
        if not sentiment_determined:
            for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                band = self._get_band_values(band_entry)
                if band[0] <= calculated_price <= band[1]:
                    last_interaction = 'horizontal_band'
        
        # If inside horizontal band and sentiment not determined, set to NEUTRAL
        if last_interaction and not sentiment_determined:
            self.sentiment = Sentiment.NEUTRAL
            sentiment_determined = True
        
        # PRIORITY 5: Touch-and-Move Detection (CPR Bands)
        if not sentiment_determined:
            touched_and_moved = False
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                if level_name not in self.cpr_levels:
                    continue
                level = self.cpr_levels[level_name]
                bullish_zone = [level, level + cpr_width]
                bearish_zone = [level - cpr_width, level]
                
                # Check if high or low touched a band boundary
                if (low <= bullish_zone[1] <= high) or (low <= bearish_zone[0] <= high):
                    # Check where close is relative to the band
                    if close < bearish_zone[0]:
                        self.sentiment = Sentiment.BEARISH
                        touched_and_moved = True
                        break
                    elif close > bullish_zone[1]:
                        self.sentiment = Sentiment.BULLISH
                        touched_and_moved = True
                        break
            
            # PRIORITY 6: Horizontal Band Cross Detection
            if not touched_and_moved:
                for band_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                    band = self._get_band_values(band_entry)
                    band_lower = band[0]
                    band_upper = band[1]
                    
                    # Check if price crossed the band completely
                    if (open_price < band_lower and close > band_upper) or (open_price > band_upper and close < band_lower):
                        if close < band_lower:
                            self.sentiment = Sentiment.BEARISH
                            touched_and_moved = True
                            break
                        elif close > band_upper:
                            self.sentiment = Sentiment.BULLISH
                            touched_and_moved = True
                            break
            
            # PRIORITY 7: Pivot-Based Fallback
            if not touched_and_moved:
                pivot_value = self.cpr_levels['PIVOT']
                if low > pivot_value:
                    self.sentiment = Sentiment.BULLISH
                elif high < pivot_value:
                    self.sentiment = Sentiment.BEARISH
                else:
                    self.sentiment = Sentiment.NEUTRAL
    
    def _run_ongoing_sentiment_logic(self, candle, spatial, temporal):
        """
        Ongoing sentiment logic for subsequent candles.
        Uses decision engine with explicit rules - fully refactored!
        """
        # Use decision engine to determine sentiment based on spatial/temporal analysis
        self.sentiment = self.decision_engine.determine_sentiment(
            spatial, temporal, self.sentiment
        )
    
    def _get_cpr_pair(self, price):
        """Helper to find which CPR pair the price is in"""
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
            if lower <= price <= upper:
                return f"{levels[i][0]}_{levels[i+1][0]}"
        return None
    
    def _get_band_values(self, band_entry):
        """Helper to extract band [low, high] from either old format [low, high] or new format {'band': [low, high], 'timestamp': ...}"""
        if isinstance(band_entry, dict):
            return band_entry['band']
        else:
            return band_entry
    
    def _bands_overlap(self, band_a, band_b):
        """Return True if two bands overlap (inclusive)."""
        return not (band_a[1] < band_b[0] or band_b[1] < band_a[0])
    
    def _is_swing_high(self, idx, n):
        """Check if candle at index is a swing high."""
        val = self.candles[idx]['calculated_price']
        # Check N before
        for i in range(max(0, idx - n), idx):
            if self.candles[i]['calculated_price'] >= val:
                return False
        # Check N after
        for i in range(idx + 1, min(len(self.candles), idx + n + 1)):
            if self.candles[i]['calculated_price'] >= val:
                return False
        return True
    
    def _is_swing_low(self, idx, n):
        """Check if candle at index is a swing low."""
        val = self.candles[idx]['calculated_price']
        # Check N before
        for i in range(max(0, idx - n), idx):
            if self.candles[i]['calculated_price'] <= val:
                return False
        # Check N after
        for i in range(idx + 1, min(len(self.candles), idx + n + 1)):
            if self.candles[i]['calculated_price'] <= val:
                return False
        return True
    
    def _process_delayed_swings(self):
        """
        Checks if the candle at (Current - N) was a swing.
        If so, adds a horizontal band or neutralizes a CPR band.
        """
        n = self.config['SWING_CONFIRMATION_CANDLES']
        target_idx = self.current_candle_index - n
        
        if target_idx < n:
            return None  # Not enough history yet

        target_candle = self.candles[target_idx]
        swing_detection_timestamp = target_candle.get('date', '')
        timestamp = swing_detection_timestamp
        if timestamp and ' ' in str(timestamp):
            time_part = str(timestamp).split(' ')[1] if ' ' in str(timestamp) else str(timestamp)
        else:
            time_part = str(timestamp) if timestamp else ""
        
        neutralization_info = None
        
        # Check Swing High
        if self._is_swing_high(target_idx, n):
            neutralization_info = self._handle_new_swing(
                target_candle['calculated_price'], 'high', time_part, swing_detection_timestamp
            )
            
        # Check Swing Low
        if self._is_swing_low(target_idx, n):
            neutralization_info = self._handle_new_swing(
                target_candle['calculated_price'], 'low', time_part, swing_detection_timestamp
            )
        
        return neutralization_info

    def _handle_new_swing(self, price, swing_type, timestamp="", swing_detection_datetime=None):
        """Decides whether to create a horizontal band or neutralize a CPR zone."""
        cpr_width = self.config['CPR_BAND_WIDTH']
        ignore_buffer = self.config['CPR_IGNORE_BUFFER']
        
        # First pass: Check if swing is INSIDE any CPR zone (neutralize takes priority)
        for name, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            
            # Check if INSIDE Bullish Zone -> Neutralize
            if bullish_zone[0] <= price <= bullish_zone[1]:
                n = self.config['SWING_CONFIRMATION_CANDLES']
                swing_detection_idx = self.current_candle_index - n
                self.cpr_band_states[name]['bullish_neutralized'] = True
                self.cpr_band_states[name]['bullish_neutralized_at'] = swing_detection_idx
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - NEUTRALIZED CPR {name} bullish zone")
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price, 'timestamp': timestamp, 'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bullish zone"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price, 'timestamp': timestamp, 'status': 'NEUTRALIZED',
                        'reason': f"neutralized CPR {name} bullish zone"
                    })
                return {'neutralized_at': swing_detection_idx, 'level_name': name, 'zone_type': 'bullish'}
                
            # Check if INSIDE Bearish Zone -> Neutralize
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
                return  # Do not create horizontal band

        # Second pass: Check Ignore Buffer (ONLY if NOT inside any zone)
        for name, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            
            if abs(price - level) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED (within {ignore_buffer:.2f} of CPR {name})")
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price, 'timestamp': timestamp, 'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} level"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price, 'timestamp': timestamp, 'status': 'IGNORED',
                        'reason': f"within {ignore_buffer:.2f} of CPR {name} level"
                    })
                return None
            
            # Check distance to zone edges
            if price < bullish_zone[0] and abs(price - bullish_zone[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED")
                return None
            if price > bullish_zone[1] and abs(price - bullish_zone[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED")
                return None
            if price < bearish_zone[0] and abs(price - bearish_zone[0]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED")
                return None
            if price > bearish_zone[1] and abs(price - bearish_zone[1]) <= ignore_buffer:
                if self.verbose_swing_logging:
                    print(f"  [SWING {swing_type.upper()} @{timestamp}] Price {price:.2f} - IGNORED")
                return None
        
        # Create/merge horizontal band (only if dynamic swing bands are enabled)
        if self.enable_dynamic_swing_bands:
            self._add_horizontal_band(price, swing_type, timestamp, swing_detection_datetime)
        return None

    def _add_horizontal_band(self, price, swing_type, timestamp="", swing_detection_datetime=None):
        """Add or merge a horizontal band from a swing."""
        width = self.config['HORIZONTAL_BAND_WIDTH']
        tolerance = self.config['MERGE_TOLERANCE']
        
        target_list = self.horizontal_bands['resistance'] if swing_type == 'high' else self.horizontal_bands['support']
        band_type = 'RESISTANCE' if swing_type == 'high' else 'SUPPORT'
        
        detection_datetime = swing_detection_datetime
        if detection_datetime is None:
            if self.current_candle_index >= 0 and len(self.candles) > 0:
                detection_datetime = self.candles[self.current_candle_index].get('date')
        
        proposed_band = [price - width, price + width]

        # 1) Ignore if overlaps CPR zone
        cpr_width = self.config['CPR_BAND_WIDTH']
        for _, level in self.cpr_levels.items():
            bullish_zone = [level, level + cpr_width]
            bearish_zone = [level - cpr_width, level]
            if self._bands_overlap(proposed_band, bullish_zone) or self._bands_overlap(proposed_band, bearish_zone):
                if self.verbose_swing_logging:
                    print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: overlaps CPR zone")
                return

        # 2) Ignore if overlaps default horizontal band
        for default_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
            default_band = self._get_band_values(default_entry)
            default_ts = default_entry.get('timestamp') if isinstance(default_entry, dict) else None
            if default_ts is None and self._bands_overlap(proposed_band, default_band):
                if self.verbose_swing_logging:
                    print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: overlaps default midpoint band")
                return

        # 3) Try to merge
        for i, band_entry in enumerate(target_list):
            band = self._get_band_values(band_entry)
            center = (band[0] + band[1]) / 2
            if abs(price - center) <= tolerance:
                old_band = band.copy()
                min_bound = min(band[0], price - width)
                max_bound = max(band[1], price + width)
                new_center = (price + center) / 2
                expanded_min = min(min_bound, new_center - width)
                expanded_max = max(max_bound, new_center + width)
                merged_band = [expanded_min, expanded_max]
                
                existing_timestamp = band_entry.get('timestamp') if isinstance(band_entry, dict) else None
                if existing_timestamp is None:
                    if self.verbose_swing_logging:
                        print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: would merge with default band")
                    return
                
                # Check if merged band overlaps default bands
                for default_entry in self.horizontal_bands['resistance'] + self.horizontal_bands['support']:
                    default_band = self._get_band_values(default_entry)
                    default_ts = default_entry.get('timestamp') if isinstance(default_entry, dict) else None
                    if default_ts is None and self._bands_overlap(merged_band, default_band):
                        if self.verbose_swing_logging:
                            print(f"    -> IGNORED {band_type.lower()} band @{timestamp}: merged band would overlap default")
                        return
                
                target_list[i] = {'band': merged_band, 'timestamp': existing_timestamp if existing_timestamp else detection_datetime}
                if self.verbose_swing_logging:
                    print(f"    -> MERGED with existing {band_type.lower()} band @{timestamp}")
                if swing_type == 'high':
                    self.detected_swing_highs.append({
                        'price': price, 'timestamp': timestamp, 'status': 'VALID',
                        'reason': f"merged with existing {band_type.lower()} band"
                    })
                else:
                    self.detected_swing_lows.append({
                        'price': price, 'timestamp': timestamp, 'status': 'VALID',
                        'reason': f"merged with existing {band_type.lower()} band"
                    })
                return

        # Create new
        new_band = [price - width, price + width]
        target_list.append({'band': new_band, 'timestamp': detection_datetime})
        if self.verbose_swing_logging:
            print(f"    -> CREATED new {band_type.lower()} band @{timestamp}: [{new_band[0]:.2f}, {new_band[1]:.2f}]")
        if swing_type == 'high':
            self.detected_swing_highs.append({
                'price': price, 'timestamp': timestamp, 'status': 'VALID',
                'reason': f"created new {band_type.lower()} band"
            })
        else:
            self.detected_swing_lows.append({
                'price': price, 'timestamp': timestamp, 'status': 'VALID',
                'reason': f"created new {band_type.lower()} band"
            })

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
                    # Get spatial and temporal analysis for this candle
                    prev_candle = self.candles[idx - 1] if idx > 0 else None
                    # Update spatial analyzer with current index
                    self.spatial_analyzer.current_candle_index = idx
                    spatial = self.spatial_analyzer.analyze(candle)
                    temporal = self.temporal_analyzer.analyze(candle, prev_candle)
                    self._run_ongoing_sentiment_logic(candle, spatial, temporal)
                    self.current_candle_index = original_index
                
                # Update stored result
                self.sentiment_results[idx]['sentiment'] = self.sentiment.value


# Example usage and comparison
if __name__ == "__main__":
    print("=" * 80)
    print("Refactored Trading Sentiment Analyzer - Proof of Concept")
    print("=" * 80)
    print("\nKey Improvements:")
    print("1. Separated concerns: SpatialAnalyzer, TemporalAnalyzer, DecisionEngine")
    print("2. Explicit rules: Each rule is independent and testable")
    print("3. Clear dependencies: Rules are composable, not chained")
    print("4. Easy to modify: Change one rule without affecting others")
    print("\nBenefits:")
    print("- Can test each analyzer independently")
    print("- Can test each rule independently")
    print("- Can add/modify rules without breaking existing ones")
    print("- Clear documentation of what each rule does")
    print("=" * 80)
        
