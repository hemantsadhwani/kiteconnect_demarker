"""
Real-Time Position Manager
Efficiently monitors active positions and detects SL/TP triggers using periodic checks and gap detection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set
import pandas as pd

from event_system import Event, EventType, get_event_dispatcher

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an active position"""
    symbol: str
    entry_price: float
    fixed_sl_price: float
    tp_price: Optional[float]  # None if MA trailing active
    trade_type: str  # 'Entry2', 'Entry3', 'Manual'
    metadata: dict = field(default_factory=dict)
    
    # Trailing SL state
    supertrend_sl_active: bool = False
    ma_trailing_active: bool = False
    
    # Last update timestamps
    last_sl_update: Optional[datetime] = None
    last_tp_check: Optional[datetime] = None


class RealTimePositionManager:
    """
    Efficiently monitors active positions and detects SL/TP triggers.
    
    Uses hybrid approach:
    - Periodic checks: Every 1 second (when positions active)
    - Gap detection: Immediate check on significant price movements (>2%)
    - Zero overhead: No processing when no positions exist
    """
    
    def __init__(self, state_manager, ticker_handler, config):
        self.state_manager = state_manager
        self.ticker_handler = ticker_handler
        self.config = config
        self.event_dispatcher = get_event_dispatcher()
        
        # Position tracking
        self.active_positions: Dict[str, PositionInfo] = {}
        self.exit_locks: Dict[str, asyncio.Lock] = {}  # Prevent duplicate exits while a check is in-flight
        # Track which symbols have had an EXIT_SIGNAL dispatched recently.
        # IMPORTANT: This is now used as a *throttling* / de-duplication hint,
        # not as a permanent "do not manage this symbol" flag.
        self.exit_signals_dispatched: Set[str] = set()
        
        # Efficient tick batching
        self.latest_ltp: Dict[str, float] = {}  # Track latest LTP per symbol
        self.symbol_token_map: Dict[str, int] = {}  # Symbol to token mapping
        self.token_symbol_map: Dict[int, str] = {}  # Token to symbol mapping
        
        # Configuration
        pm_config = config.get('POSITION_MANAGEMENT', {})
        self.check_interval: float = pm_config.get('CHECK_INTERVAL_SEC', 1.0)
        gap_config = pm_config.get('GAP_DETECTION', {})
        self.gap_threshold_percent: float = gap_config.get('GAP_THRESHOLD_PERCENT', 2.0)
        self.gap_detection_enabled: bool = gap_config.get('ENABLED', True)
        
        # Background task for periodic checks
        self.periodic_check_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"RealTimePositionManager initialized: check_interval={self.check_interval}s, gap_threshold={self.gap_threshold_percent}%")
    
    async def start(self):
        """Start periodic position monitoring"""
        if not self.is_running:
            self.is_running = True
            self.periodic_check_task = asyncio.create_task(self._periodic_check_loop())
            # Register handler for CANDLE_FORMED events to check MA trailing exit
            self.event_dispatcher.register_handler(EventType.CANDLE_FORMED, self.handle_candle_formed)
            logger.info("[OK] RealTimePositionManager started (CANDLE_FORMED handler registered for MA trailing)")
    
    async def stop(self):
        """Stop periodic position monitoring"""
        self.is_running = False
        if self.periodic_check_task:
            self.periodic_check_task.cancel()
            try:
                await self.periodic_check_task
            except asyncio.CancelledError:
                pass
        logger.info("[STOP] RealTimePositionManager stopped")
    
    def update_symbol_token_mapping(self, symbol_token_map: Dict[str, int]):
        """Update symbol to token mapping (called when symbols change)"""
        self.symbol_token_map = symbol_token_map
        self.token_symbol_map = {token: symbol for symbol, token in symbol_token_map.items()}
        logger.debug(f"Updated symbol-token mapping: {len(self.symbol_token_map)} symbols")
    
    async def _periodic_check_loop(self):
        """Periodic check loop - runs every CHECK_INTERVAL_SEC"""
        while self.is_running:
            try:
                # CRITICAL: Re-register orphaned positions (exist in state_manager but not in active_positions)
                # This handles cases where position was unregistered but order failed
                await self._recover_orphaned_positions()
                
                # Only process if we have active positions
                if self.active_positions:
                    await self._check_all_positions()
                
                # Sleep for check interval (default 1 second)
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic check loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def handle_candle_formed(self, event: Event):
        """
        Handle CANDLE_FORMED events:
        1. Activate SuperTrend SL if SuperTrend turns bullish (for Entry2/Manual trades)
        2. Update SuperTrend SL value every candle (trailing)
        3. Check MA trailing exit only on completed candles
        This ensures indicators are updated and exits are detected on completed candle data.
        """
        try:
            candle_data = event.data or {}
            token = candle_data.get('token')
            symbol = self.token_symbol_map.get(token) if token else None
            
            if not symbol or symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            
            # CRITICAL: Check if SuperTrend SL should be activated (if not already active)
            # This handles cases where SuperTrend turns bullish after entry
            if not position.supertrend_sl_active:
                # Check if SuperTrend is bullish and should activate trailing SL
                # Only for Entry2 and Manual trades (Entry3 uses fixed SL)
                if position.trade_type in ['Entry2', 'Manual']:
                    supertrend_value = self._get_supertrend_value(symbol)
                    if supertrend_value:
                        token = self.symbol_token_map.get(symbol)
                        if token and self.ticker_handler:
                            df_indicators = self.ticker_handler.get_indicators(token)
                            if not df_indicators.empty:
                                latest = df_indicators.iloc[-1]
                                supertrend_dir = latest.get('supertrend_dir')
                                
                                if supertrend_dir == 1:  # Bullish - activate SuperTrend SL
                                    position.supertrend_sl_active = True
                                    position.metadata['supertrend_sl_active'] = True
                                    # Update state manager metadata
                                    self.state_manager.update_trade_metadata(symbol, {
                                        'supertrend_sl_active': True
                                    })
                                    logger.info(f"[SYNC] SuperTrend SL ACTIVATED for {symbol}: SuperTrend turned bullish, using {supertrend_value:.2f} as trailing SL")
            
            # CRITICAL: Update SuperTrend SL value every candle if SuperTrend SL is active
            # This ensures trailing SL uses the latest SuperTrend value
            if position.supertrend_sl_active:
                supertrend_value = self._get_supertrend_value(symbol)
                token = self.symbol_token_map.get(symbol)
                if supertrend_value and token and self.ticker_handler:
                    df_indicators = self.ticker_handler.get_indicators(token)
                    if not df_indicators.empty:
                        latest = df_indicators.iloc[-1]
                        supertrend_dir = latest.get('supertrend_dir')
                        
                        # CRITICAL: Handle SuperTrend flip-back to bearish
                        # When SuperTrend flips back to bearish, check if price crossed below last bullish SuperTrend value
                        if supertrend_dir == -1:  # SuperTrend flipped back to bearish
                            # Get previous candle's SuperTrend value (last bullish value)
                            if len(df_indicators) >= 2:
                                prev_row = df_indicators.iloc[-2]
                                prev_supertrend_dir = prev_row.get('supertrend_dir')
                                prev_supertrend_value = prev_row.get('supertrend')
                                
                                # If previous candle was bullish, check if price crossed below it
                                if prev_supertrend_dir == 1 and pd.notna(prev_supertrend_value):
                                    candle_low = candle_data.get('low')
                                    if candle_low and candle_low <= prev_supertrend_value:
                                        # Price crossed below last bullish SuperTrend - trigger exit
                                        logger.warning(f"[STOP] SuperTrend FLIP-BACK EXIT: {symbol} - Price {candle_low:.2f} crossed below last bullish SuperTrend {prev_supertrend_value:.2f}")
                                        # Mark exit signal as dispatched
                                        if symbol not in self.exit_signals_dispatched:
                                            self.exit_signals_dispatched.add(symbol)
                                            # CRITICAL: DO NOT unregister position here - wait for order execution confirmation
                                            # The position will be unregistered by ImmediateExitExecutor after order is confirmed
                                            # Dispatch exit signal
                                            self.event_dispatcher.dispatch_event(
                                                Event(EventType.EXIT_SIGNAL, {
                                                    'symbol': symbol,
                                                    'exit_reason': 'ST_SL_FLIPBACK',
                                                    'trigger_price': candle_low,
                                                    'sl_price': prev_supertrend_value,
                                                    'order_type': 'MARKET',
                                                    'priority': 'HIGH'
                                                }, source='position_manager')
                                            )
                                            return  # Exit early - position already handled
                        
                        # Normal case: SuperTrend is still bullish - update SL value
                        if supertrend_dir == 1:
                            old_sl = position.fixed_sl_price
                            position.fixed_sl_price = supertrend_value
                            position.metadata['last_sl_price'] = supertrend_value
                            # Update state manager metadata
                            self.state_manager.update_trade_metadata(symbol, {
                                'last_sl_price': supertrend_value,
                                'calculated_sl_price': supertrend_value
                            })
                            logger.info(f"[SYNC] SuperTrend SL updated for {symbol}: {old_sl:.2f} -> {supertrend_value:.2f}")
                        
                        # CRITICAL: Also check candle low for SuperTrend SL (not just LTP)
                        # This ensures we catch price movements that happen during candle formation
                        candle_low = candle_data.get('low')
                        if candle_low and supertrend_dir == 1:
                            current_sl = position.fixed_sl_price or supertrend_value
                            if candle_low <= current_sl:
                                # Price went below SuperTrend SL during candle - trigger exit
                                logger.warning(f"[STOP] SuperTrend SL TRIGGERED (candle low): {symbol} - Low={candle_low:.2f} <= SL={current_sl:.2f}")
                                # Throttle duplicate SuperTrend SL triggers, but allow future checks
                                if symbol not in self.exit_signals_dispatched:
                                    self.exit_signals_dispatched.add(symbol)
                                    # CRITICAL: DO NOT unregister position here - wait for order execution confirmation
                                    # The position will be unregistered by ImmediateExitExecutor after order is confirmed
                                    # Dispatch exit signal
                                    self.event_dispatcher.dispatch_event(
                                        Event(EventType.EXIT_SIGNAL, {
                                            'symbol': symbol,
                                            'exit_reason': 'ST_SL',
                                            'trigger_price': candle_low,
                                            'sl_price': current_sl,
                                            'order_type': 'MARKET',
                                            'priority': 'HIGH'
                                        }, source='position_manager')
                                    )
                                    return  # Exit early - position already handled
            
            # CRITICAL: Check MA trailing exit if it's active (AFTER SuperTrend SL updates)
            # Dynamic MA trailing exit is ONLY checked at END OF CANDLE (CANDLE_FORMED events)
            # This requires completed candle data (prev and current candle MA values for crossunder detection)
            # Unlike Fixed SL and Trailing SL which are checked on every tick (mid-candle)
            if position.ma_trailing_active:
                # Get LTP for exit price
                ltp = self.latest_ltp.get(symbol)
                if ltp:
                    # Check MA trailing exit - if it triggers, it will dispatch exit signal and return True
                    if await self._check_trailing_exit(symbol, ltp, position):
                        return  # MA trailing exit triggered - stop processing
        except Exception as e:
            logger.error(f"Error in handle_candle_formed: {e}", exc_info=True)
    
    async def handle_tick_update(self, event: Event):
        """
        Update latest LTP and check for immediate triggers (gaps).
        This is lightweight - just updates LTP cache and checks for significant movements.
        """
        try:
            tick_data = event.data or {}
            token = tick_data.get('token')
            ltp = tick_data.get('ltp') or tick_data.get('last_price')
            
            if not ltp or not token:
                return
            
            # Get symbol for this token
            symbol = self.token_symbol_map.get(token)
            if not symbol:
                # Token not in mapping - might be a new symbol, try to update mapping
                if self.ticker_handler and hasattr(self.ticker_handler, 'symbol_token_map'):
                    symbol_token_map = self.ticker_handler.symbol_token_map
                    self.update_symbol_token_mapping(symbol_token_map)
                    symbol = self.token_symbol_map.get(token)
                    if symbol:
                        logger.debug(f"[LINK] Updated mapping: token {token} -> {symbol}")
                    else:
                        return  # Still no mapping
                else:
                    return
            
            # Update latest LTP (lightweight operation)
            prev_ltp = self.latest_ltp.get(symbol)
            self.latest_ltp[symbol] = ltp
            
            # Only check if position exists (early exit - most ticks will exit here)
            if symbol not in self.active_positions:
                return
            
            # CRITICAL: Always check exit triggers for active positions (not just on gaps)
            # This ensures we catch SL/TP even if price moves slowly
            await self._check_exit_triggers(symbol, ltp)
            
            # Also check for immediate trigger (gap detection) for logging
            if self.gap_detection_enabled and prev_ltp:
                price_change_percent = abs((ltp - prev_ltp) / prev_ltp) * 100
                if price_change_percent >= self.gap_threshold_percent:
                    # Significant price movement - log it
                    logger.debug(f"[WARN] Gap detected for {symbol}: {price_change_percent:.2f}% change (prev={prev_ltp:.2f}, curr={ltp:.2f})")
        
        except Exception as e:
            logger.error(f"Error in handle_tick_update: {e}", exc_info=True)
    
    async def _check_all_positions(self):
        """Check all active positions (called periodically)"""
        if not self.active_positions:
            return
        
        logger.debug(f"[SEARCH] Checking {len(self.active_positions)} active positions...")
        for symbol in list(self.active_positions.keys()):
            ltp = self.latest_ltp.get(symbol)
            if ltp:
                await self._check_exit_triggers(symbol, ltp)
            else:
                # Try to get LTP from ticker_handler if not in cache
                token = self.symbol_token_map.get(symbol)
                if token and self.ticker_handler:
                    ltp = self.ticker_handler.get_ltp(token)
                    if ltp:
                        self.latest_ltp[symbol] = ltp
                        await self._check_exit_triggers(symbol, ltp)
                    else:
                        # Only log once per symbol to reduce noise (LTP might not be available immediately after startup/slab change)
                        if symbol not in getattr(self, '_ltp_warning_logged', set()):
                            if not hasattr(self, '_ltp_warning_logged'):
                                self._ltp_warning_logged = set()
                            self._ltp_warning_logged.add(symbol)
                            logger.debug(f"[WARN] No LTP available for {symbol} (token={token}) - WebSocket may not be subscribed yet")
                else:
                    # Only log once per symbol
                    if symbol not in getattr(self, '_token_warning_logged', set()):
                        if not hasattr(self, '_token_warning_logged'):
                            self._token_warning_logged = set()
                        self._token_warning_logged.add(symbol)
                        logger.warning(f"[WARN] No token mapping or ticker_handler for {symbol}")
    
    async def _check_exit_triggers(self, symbol: str, ltp: float):
        """Check all exit conditions for a position"""
        # CRITICAL: Early exit if position not registered
        if symbol not in self.active_positions:
            return
        
        # CRITICAL: Validate LTP is available before checking exit conditions
        # Exit signals require a valid trigger price - cannot dispatch with None
        if ltp is None or ltp <= 0:
            logger.debug(f"[SKIP] Cannot check exit triggers for {symbol}: LTP is None or invalid (ltp={ltp})")
            return
        
        # NOTE: We NO LONGER hard-skip all checks just because an exit signal
        # was dispatched earlier. An EXIT_SIGNAL is a *request*, not a guarantee
        # that the position is flat. We always confirm with state manager,
        # and optionally broker (via ImmediateExitExecutor), before giving up.
        
        # CRITICAL: Check state manager for trade metadata when available.
        # If trade is missing, we CONTINUE managing the position based on
        # active_positions to avoid silently dropping SL/TP protection.
        trade = self.state_manager.get_trade(symbol)
        position = self.active_positions[symbol]
        
        if trade:
            # CRITICAL: Sync state from metadata to ensure position manager has latest flags
            trade_metadata = trade.get('metadata', {})
            
            # Sync SuperTrend SL state
            metadata_supertrend_sl_active = trade_metadata.get('supertrend_sl_active', False)
            if metadata_supertrend_sl_active != position.supertrend_sl_active:
                position.supertrend_sl_active = metadata_supertrend_sl_active
                position.metadata['supertrend_sl_active'] = metadata_supertrend_sl_active
                if metadata_supertrend_sl_active:
                    logger.info(f"[SYNC] Synced SuperTrend SL state: Activated for {symbol}")
            
            # Sync MA trailing state from metadata (in case it was activated by threshold check)
            # This ensures we check for MA crossunder exit only after threshold is reached
            metadata_ma_trailing_active = trade_metadata.get('dynamic_trailing_ma_active', False)
            if metadata_ma_trailing_active != position.ma_trailing_active:
                # State mismatch - sync from metadata
                position.ma_trailing_active = metadata_ma_trailing_active
                position.metadata['dynamic_trailing_ma_active'] = metadata_ma_trailing_active
                if metadata_ma_trailing_active:
                    position.tp_price = None  # Remove TP when MA trailing activates
                    logger.info(f"[SYNC] Synced MA trailing state: Activated for {symbol} (threshold reached)")
        
        # Prevent overlapping checks, but do NOT permanently suppress future checks
        if symbol in self.exit_locks and self.exit_locks[symbol].locked():
            logger.debug(f"[WARN] Exit check already in progress for {symbol}, skipping this tick")
            return
        
        async with self.exit_locks.setdefault(symbol, asyncio.Lock()):
                # Double-check position still exists
                if symbol not in self.active_positions:
                    return
                
                # 1. Check Stop Loss (highest priority) - checked on EVERY TICK (mid-candle)
                # This includes both Fixed SL and Trailing SL (SuperTrend SL)
                # These are checked immediately on tick updates for fast exit execution
                if await self._check_stop_loss(symbol, ltp, position):
                    return
                
                # 2. Check Take Profit (if fixed TP exists) - checked on EVERY TICK (mid-candle)
                # TP is checked immediately on tick updates for fast exit execution
                if await self._check_take_profit(symbol, ltp, position):
                    return
            
                # 3. MA Trailing Exit is NOT checked here - it's ONLY checked on CANDLE_FORMED events (end of candle)
                # Dynamic MA trailing exit requires completed candle data (prev and current candle MA values)
                # This ensures we only check MA crossunder on completed candles, not mid-candle
                # See handle_candle_formed() method for MA trailing exit check
    
    async def _check_stop_loss(self, symbol: str, ltp: float, position: PositionInfo) -> bool:
        """Check if SL is triggered"""
        try:
            # CRITICAL: Validate LTP is available before checking SL
            # Exit signals require a valid trigger price
            if ltp is None or ltp <= 0:
                logger.debug(f"[SKIP] Cannot check SL for {symbol}: LTP is None or invalid (ltp={ltp})")
                return False
            
            current_sl = await self._get_current_sl_price(symbol, position)
            
            # CRITICAL: Log warning if SL price is invalid (None or 0)
            # This prevents silent failures where SL never triggers
            if not current_sl or current_sl <= 0:
                logger.error(
                    f"[CRITICAL] SL CHECK SKIPPED for {symbol}: Invalid SL price (current_sl={current_sl}, "
                    f"LTP={ltp:.2f}, entry={position.entry_price:.2f}, fixed_sl={position.fixed_sl_price}, "
                    f"supertrend_sl_active={position.supertrend_sl_active}). "
                    f"Position is UNPROTECTED - SL will never trigger!"
                )
                return False
            
            if ltp <= current_sl:
                # CRITICAL: Throttle duplicate SL triggers within the same
                # life-cycle, but do NOT permanently disable SL checks.
                if symbol in self.exit_signals_dispatched:
                    logger.warning(f"[WARN] Exit signal already dispatched for {symbol}, skipping duplicate SL trigger on this tick")
                    return False
                
                self.exit_signals_dispatched.add(symbol)
                
                # CRITICAL: DO NOT unregister position here - wait for order execution confirmation
                # The position will be unregistered by ImmediateExitExecutor after order is confirmed
                # This prevents orphaned positions if order execution fails
                
                gap_size = current_sl - ltp
                gap_percent = (gap_size / current_sl) * 100 if current_sl > 0 else 0
                
                logger.warning(f"[STOP] SL TRIGGERED: {symbol} - LTP={ltp:.2f} <= SL={current_sl:.2f} (gap={gap_size:.2f}, {gap_percent:.2f}%)")
                
                self.event_dispatcher.dispatch_event(
                    Event(EventType.EXIT_SIGNAL, {
                        'symbol': symbol,
                        'exit_reason': 'SL',
                        'trigger_price': ltp,
                        'sl_price': current_sl,
                        'order_type': 'MARKET',  # Immediate market order
                        'priority': 'HIGH',  # SL has highest priority
                        'gap_size': gap_size,
                        'gap_percent': gap_percent
                    }, source='position_manager')
                )
                return True
        except Exception as e:
            logger.error(f"Error checking stop loss for {symbol}: {e}", exc_info=True)
        return False
    
    async def _check_take_profit(self, symbol: str, ltp: float, position: PositionInfo) -> bool:
        """Check if TP is triggered (only if fixed TP exists)"""
        try:
            # CRITICAL: Validate LTP is available before checking TP
            # Exit signals require a valid trigger price
            if ltp is None or ltp <= 0:
                logger.debug(f"[SKIP] Cannot check TP for {symbol}: LTP is None or invalid (ltp={ltp})")
                return False
            
            # CRITICAL: Check if position is already closed in state manager
            trade = self.state_manager.get_trade(symbol)
            if not trade:
                # Position already closed - unregister and return
                if symbol in self.active_positions:
                    await self.unregister_position(symbol)
                return False
            
            # Skip TP check if MA trailing is active (no fixed TP)
            if position.ma_trailing_active or position.tp_price is None:
                return False
            
            if ltp >= position.tp_price:
                # CRITICAL: Mark exit signal as dispatched FIRST to prevent duplicates
                if symbol in self.exit_signals_dispatched:
                    logger.warning(f"[WARN] Exit signal already dispatched for {symbol}, skipping duplicate TP trigger")
                    return False
                
                self.exit_signals_dispatched.add(symbol)
                
                # CRITICAL: DO NOT unregister position here - wait for order execution confirmation
                # The position will be unregistered by ImmediateExitExecutor after order is confirmed
                # This prevents orphaned positions if order execution fails
                
                logger.info(f"[TARGET] TP TRIGGERED: {symbol} - LTP={ltp:.2f} >= TP={position.tp_price:.2f}")
                
                self.event_dispatcher.dispatch_event(
                    Event(EventType.EXIT_SIGNAL, {
                        'symbol': symbol,
                        'exit_reason': 'TP',
                        'trigger_price': ltp,
                        'tp_price': position.tp_price,
                        'order_type': 'MARKET',
                        'priority': 'MEDIUM'
                    }, source='position_manager')
                )
                return True
        except Exception as e:
            logger.error(f"Error checking take profit for {symbol}: {e}", exc_info=True)
        return False
    
    async def _check_trailing_exit(self, symbol: str, ltp: float, position: PositionInfo) -> bool:
        """
        Check for trailing exit conditions (MA crossunder).
        This is ONLY called on CANDLE_FORMED events, ensuring we check completed candle data.
        """
        try:
            # CRITICAL: Validate LTP is available before checking trailing exit
            # Exit signals require a valid trigger price
            if ltp is None or ltp <= 0:
                logger.debug(f"[SKIP] Cannot check trailing exit for {symbol}: LTP is None or invalid (ltp={ltp})")
                return False
            
            # Only check if MA trailing is active
            if not position.ma_trailing_active:
                return False
            
            # CRITICAL: Verify threshold was reached before checking MA crossunder
            trade = self.state_manager.get_trade(symbol)
            if not trade:
                return False
            
            trade_metadata = trade.get('metadata', {})
            if not trade_metadata.get('dynamic_trailing_ma_active', False):
                logger.debug(f"MA trailing not activated yet for {symbol} (threshold not reached), skipping crossunder check")
                return False
            
            # Get indicators for MA crossunder check
            token = self.symbol_token_map.get(symbol)
            if not token or not self.ticker_handler:
                return False
            
            df_indicators = self.ticker_handler.get_indicators(token)
            if df_indicators.empty or len(df_indicators) < 2:
                return False
            
            # Get column names (fast_ma/slow_ma or legacy ema3/sma7)
            fast_ma_col = 'fast_ma' if 'fast_ma' in df_indicators.columns else 'ema3'
            slow_ma_col = 'slow_ma' if 'slow_ma' in df_indicators.columns else 'sma7'
            
            if fast_ma_col not in df_indicators.columns or slow_ma_col not in df_indicators.columns:
                return False
            
            current_row = df_indicators.iloc[-1]
            prev_row = df_indicators.iloc[-2]
            
            current_fast_ma = current_row.get(fast_ma_col)
            prev_fast_ma = prev_row.get(fast_ma_col)
            current_slow_ma = current_row.get(slow_ma_col)
            prev_slow_ma = prev_row.get(slow_ma_col)
            
            # Check for crossunder: prev_fast_ma >= prev_slow_ma AND current_fast_ma < current_slow_ma
            if pd.isna(current_fast_ma) or pd.isna(prev_fast_ma) or pd.isna(current_slow_ma) or pd.isna(prev_slow_ma):
                return False
            
            if prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma:
                # CRITICAL: Mark exit signal as dispatched FIRST to prevent duplicates
                if symbol in self.exit_signals_dispatched:
                    logger.warning(f"[WARN] Exit signal already dispatched for {symbol}, skipping duplicate MA trailing exit")
                    return False
                
                self.exit_signals_dispatched.add(symbol)
                
                # CRITICAL: DO NOT unregister position here - wait for order execution confirmation
                # The position will be unregistered by ImmediateExitExecutor after order is confirmed
                # This prevents orphaned positions if order execution fails
                
                # MA crossunder detected - dispatch exit signal
                logger.info(f"[DOWN] MA TRAILING EXIT: {symbol} - {fast_ma_col} crossed under {slow_ma_col} (LTP={ltp:.2f})")
                
                self.event_dispatcher.dispatch_event(
                    Event(EventType.EXIT_SIGNAL, {
                        'symbol': symbol,
                        'exit_reason': 'MA_TRAILING_EXIT',
                        'trigger_price': ltp,
                        'order_type': 'MARKET',
                        'priority': 'MEDIUM',
                        'fast_ma': current_fast_ma,
                        'slow_ma': current_slow_ma
                    }, source='position_manager')
                )
                return True
        except Exception as e:
            logger.error(f"Error checking trailing exit for {symbol}: {e}", exc_info=True)
        return False
    
    async def _get_current_sl_price(self, symbol: str, position: PositionInfo) -> Optional[float]:
        """Get current SL price (fixed or trailing)"""
        try:
            # Check if SuperTrend SL is active
            if position.supertrend_sl_active:
                # CRITICAL: Use position.fixed_sl_price which is updated every candle in handle_candle_formed
                # This ensures we use the latest SuperTrend value that was updated on candle completion
                # Fallback to getting it dynamically if somehow not updated
                if position.fixed_sl_price and position.fixed_sl_price > 0:
                    return position.fixed_sl_price
                else:
                    # Fallback: Get latest SuperTrend value dynamically
                    supertrend_value = self._get_supertrend_value(symbol)
                    if supertrend_value and supertrend_value > 0:
                        return supertrend_value
                    else:
                        # CRITICAL: SuperTrend SL is active but value unavailable - this is a SYSTEM FAILURE
                        # SuperTrend is a core indicator that should ALWAYS be available
                        # DO NOT fall back to fixed SL - if price moved 50% up, fixed 6% SL from entry would trigger immediately
                        # This is a critical error that needs immediate attention
                        logger.critical(
                            f"[CRITICAL SYSTEM FAILURE] SuperTrend SL active for {symbol} but SuperTrend value unavailable! "
                            f"This is a CORE SYSTEM INDICATOR that should always be available. "
                            f"Position is UNPROTECTED. Entry={position.entry_price:.2f}, "
                            f"fixed_sl={position.fixed_sl_price}, supertrend_value={supertrend_value}. "
                            f"Returning None - will be logged in _check_stop_loss."
                        )
                        # Return None - this will trigger critical error logging in _check_stop_loss
                        # DO NOT fall back to fixed SL as it would be dangerously wrong if price has moved significantly
                        return None
            
            # Use fixed SL
            # CRITICAL: Validate fixed SL is valid (not None or 0)
            if position.fixed_sl_price and position.fixed_sl_price > 0:
                return position.fixed_sl_price
            else:
                # Fixed SL is invalid - try to calculate from entry price as last resort
                if position.entry_price and position.entry_price > 0:
                    # Calculate default 6% SL as emergency fallback
                    trade_settings = self.config.get('TRADE_SETTINGS', {})
                    default_sl_percent = trade_settings.get('STOP_LOSS_PERCENT', 6.0)
                    calculated_sl = position.entry_price * (1 - default_sl_percent / 100)
                    logger.warning(
                        f"[FALLBACK] Fixed SL is invalid for {symbol} (fixed_sl={position.fixed_sl_price}). "
                        f"Calculating emergency SL from entry: {calculated_sl:.2f} ({default_sl_percent}% from entry {position.entry_price:.2f})"
                    )
                    return calculated_sl
                else:
                    # Cannot calculate SL - return None (will be logged in _check_stop_loss)
                    return None
        except Exception as e:
            logger.error(f"Error getting current SL price for {symbol}: {e}", exc_info=True)
            # Try to return fixed SL as fallback, or calculate from entry if available
            if position.fixed_sl_price and position.fixed_sl_price > 0:
                return position.fixed_sl_price
            elif position.entry_price and position.entry_price > 0:
                trade_settings = self.config.get('TRADE_SETTINGS', {})
                default_sl_percent = trade_settings.get('STOP_LOSS_PERCENT', 6.0)
                return position.entry_price * (1 - default_sl_percent / 100)
            return None
    
    def _get_supertrend_value(self, symbol: str) -> Optional[float]:
        """Get latest SuperTrend value for a symbol"""
        try:
            token = self.symbol_token_map.get(symbol)
            if not token or not self.ticker_handler:
                return None
            
            df_indicators = self.ticker_handler.get_indicators(token)
            if df_indicators.empty:
                return None
            
            latest = df_indicators.iloc[-1]
            supertrend_value = latest.get('supertrend')
            
            if pd.notna(supertrend_value):
                return float(supertrend_value)
        except Exception as e:
            logger.debug(f"Error getting SuperTrend value for {symbol}: {e}")
        return None
    
    async def register_position(self, symbol: str, entry_price: float, 
                               sl_price: float, tp_price: Optional[float], 
                               trade_type: str, metadata: dict):
        """Register a new position for monitoring"""
        try:
            # CRITICAL: Clean up stale positions before registering new trade
            # Only one trade should be active at a time, so unregister any existing positions
            # Check if positions still exist in state manager - if not, they're stale
            stale_positions = []
            for existing_symbol in list(self.active_positions.keys()):
                # Check if position still exists in state manager
                trade = self.state_manager.get_trade(existing_symbol)
                if not trade:
                    # Position doesn't exist in state manager - it's stale
                    stale_positions.append(existing_symbol)
                    logger.warning(f"[CLEAN] Found stale position: {existing_symbol} (not found in state manager)")
            
            # Unregister stale positions
            for stale_symbol in stale_positions:
                logger.warning(f"[CLEAN] Cleaning up stale position: {stale_symbol}")
                await self.unregister_position(stale_symbol)
            
            # CRITICAL: Update symbol-token mapping if ticker_handler is available
            if self.ticker_handler and hasattr(self.ticker_handler, 'symbol_token_map'):
                symbol_token_map = self.ticker_handler.symbol_token_map
                self.update_symbol_token_mapping(symbol_token_map)
                logger.info(f"[LINK] Updated symbol-token mapping for {symbol}: {len(self.symbol_token_map)} symbols mapped")
            
            # Verify we have token mapping for this symbol
            token = self.symbol_token_map.get(symbol)
            if not token:
                logger.warning(f"[WARN] WARNING: No token mapping found for {symbol}. Position monitoring may not work correctly!")
                logger.warning(f"[WARN] Available symbols in mapping: {list(self.symbol_token_map.keys())}")
            else:
                logger.info(f"[OK] Token mapping verified: {symbol} -> {token}")
            
            # CRITICAL: Validate SL price before registering position
            # This prevents positions from being registered with invalid SL prices that will never trigger
            if not sl_price or sl_price <= 0:
                logger.error(
                    f"[CRITICAL] Cannot register position {symbol}: Invalid SL price (sl_price={sl_price}, "
                    f"entry_price={entry_price:.2f}). Position would be UNPROTECTED - rejecting registration!"
                )
                raise ValueError(f"Invalid SL price for {symbol}: {sl_price}. SL price must be a positive number.")
            
            # Validate entry price
            if not entry_price or entry_price <= 0:
                logger.error(
                    f"[CRITICAL] Cannot register position {symbol}: Invalid entry price (entry_price={entry_price}). "
                    f"Rejecting registration!"
                )
                raise ValueError(f"Invalid entry price for {symbol}: {entry_price}. Entry price must be a positive number.")
            
            self.active_positions[symbol] = PositionInfo(
                symbol=symbol,
                entry_price=entry_price,
                fixed_sl_price=sl_price,
                tp_price=tp_price,
                trade_type=trade_type,
                metadata=metadata,
                supertrend_sl_active=metadata.get('supertrend_sl_active', False),
                ma_trailing_active=metadata.get('dynamic_trailing_ma_active', False)
            )
            
            # Clear any old exit signal tracking for this symbol (in case of re-entry)
            self.exit_signals_dispatched.discard(symbol)
            
            tp_str = f"{tp_price:.2f}" if tp_price else "None"
            logger.critical(f"[NOTE][NOTE][NOTE] REGISTERED POSITION FOR MONITORING: {symbol} (Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_str}, SuperTrendSL={metadata.get('supertrend_sl_active', False)}) [NOTE][NOTE][NOTE]")
            logger.info(f"[NOTE] Active positions count: {len(self.active_positions)}")
        except Exception as e:
            logger.error(f"Error registering position {symbol}: {e}", exc_info=True)
    
    async def unregister_position(self, symbol: str):
        """Remove position from monitoring"""
        try:
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                logger.info(f"[DELETE] Unregistered position: {symbol}")
            
            if symbol in self.exit_locks:
                del self.exit_locks[symbol]
            
            # Clear exit signal tracking
            self.exit_signals_dispatched.discard(symbol)
            
            # Clean up LTP cache if no other positions need it
            # (Keep it for now as it's lightweight and may be used by other components)
        except Exception as e:
            logger.error(f"Error unregistering position {symbol}: {e}", exc_info=True)
    
    async def update_trailing_sl(self, symbol: str):
        """Update trailing SL state for a position (called when SuperTrend changes)"""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            
            # Update SuperTrend SL state
            supertrend_value = self._get_supertrend_value(symbol)
            if supertrend_value:
                # Check if SuperTrend is bullish (for Entry2/Manual trades)
                token = self.symbol_token_map.get(symbol)
                if token and self.ticker_handler:
                    df_indicators = self.ticker_handler.get_indicators(token)
                    if not df_indicators.empty:
                        latest = df_indicators.iloc[-1]
                        supertrend_dir = latest.get('supertrend_dir')
                        
                        if supertrend_dir == 1:  # Bullish
                            if not position.supertrend_sl_active:
                                position.supertrend_sl_active = True
                                position.metadata['supertrend_sl_active'] = True
                                logger.info(f"[SYNC] SuperTrend SL activated for {symbol}: {supertrend_value:.2f}")
        except Exception as e:
            logger.error(f"Error updating trailing SL for {symbol}: {e}", exc_info=True)
    
    async def _recover_orphaned_positions(self):
        """
        Re-register positions that exist in state_manager but not in active_positions.
        This handles cases where position was unregistered but order execution failed.
        """
        try:
            active_trades = self.state_manager.get_active_trades()
            if not active_trades:
                return
            
            for symbol, trade in active_trades.items():
                # Check if position exists in state_manager but not in active_positions
                if symbol not in self.active_positions:
                    # Skip if exit signal was already dispatched (order might be in progress)
                    if symbol in self.exit_signals_dispatched:
                        continue
                    
                    # Position is orphaned - re-register it
                    entry_price = trade.get('entry_price')
                    metadata = trade.get('metadata', {})
                    
                    # Get SL price from metadata or calculate from entry
                    sl_price = metadata.get('calculated_sl_price') or metadata.get('last_sl_price')
                    if not sl_price and entry_price:
                        # Fallback: Calculate 6% SL
                        sl_price = entry_price * 0.94
                    
                    tp_price = trade.get('tp_price')
                    trade_type = trade.get('entry_type', 'Manual')
                    
                    if entry_price and sl_price:
                        logger.warning(f"[SYNC] RECOVERING ORPHANED POSITION: {symbol} (exists in state_manager but not in active_positions)")
                        await self.register_position(
                            symbol=symbol,
                            entry_price=entry_price,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            trade_type=trade_type,
                            metadata=metadata
                        )
                        logger.info(f"[OK] Re-registered orphaned position: {symbol}")
        except Exception as e:
            logger.error(f"Error recovering orphaned positions: {e}", exc_info=True)
    
    async def update_ma_trailing_state(self, symbol: str, is_active: bool):
        """Update MA trailing state for a position"""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            position.ma_trailing_active = is_active
            position.metadata['dynamic_trailing_ma_active'] = is_active
            
            if is_active:
                # Remove TP when MA trailing activates
                position.tp_price = None
                logger.info(f"[SYNC] MA trailing activated for {symbol} - TP removed")
        except Exception as e:
            logger.error(f"Error updating MA trailing state for {symbol}: {e}", exc_info=True)
    
    def get_active_positions_count(self) -> int:
        """Get count of active positions being monitored"""
        return len(self.active_positions)
    
    def is_position_active(self, symbol: str) -> bool:
        """Check if a position is being monitored"""
        return symbol in self.active_positions

