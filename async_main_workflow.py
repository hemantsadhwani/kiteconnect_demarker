"""
Async Main Workflow for Trading System
Replaces the polling-based main_workflow2.py with event-driven architecture
"""

import asyncio
import logging
import yaml
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from trading_bot_utils import (
    get_kite_api_instance, 
    is_market_open_time, 
    calculate_strikes, 
    get_weekly_expiry_date,
    get_nifty_opening_price_historical,
    get_nifty_latest_calculated_price_historical,
    generate_option_tokens_and_update_file
)
from dynamic_atm_strike_manager import DynamicATMStrikeManager
from trade_state_manager import TradeStateManager
from indicators import IndicatorManager
from strategy_executor import StrategyExecutor
from event_system import Event, EventType, get_event_dispatcher
from async_event_handlers import AsyncEventHandlers, get_async_event_handlers
from async_live_ticker_handler import AsyncLiveTickerHandler
from async_api_server import AsyncAPIServer, get_async_api_server
from entry_conditions import EntryConditionManager

# Setup logging with Unicode encoding support for Windows
import sys
import io

# Custom StreamHandler that handles Unicode encoding errors gracefully
class SafeUnicodeStreamHandler(logging.StreamHandler):
    """StreamHandler that safely handles Unicode characters on Windows"""
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        # Wrap the stream with UTF-8 encoding and error replacement
        if sys.platform == 'win32' and hasattr(stream, 'buffer'):
            try:
                # Create a TextIOWrapper that handles encoding errors
                wrapped_stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                super().__init__(wrapped_stream)
            except Exception:
                # Fallback to original stream
                super().__init__(stream)
        else:
            super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace emoji characters with ASCII-safe alternatives
            emoji_replacements = {
                '🛑': '[STOP]', '✅': '[OK]', '❌': '[X]', '⚠️': '[WARN]',
                '🗑️': '[DELETE]', '🚀': '[START]', '📊': '[CHART]', '💰': '[MONEY]',
                '🔴': '[RED]', '🟢': '[GREEN]', '🟡': '[YELLOW]', '⚡': '[FAST]',
                '🕯️': '[CANDLE]', '📌': '[INFO]', '🎯': '[TARGET]', '⏰': '[TIME]',
                '🕘': '[CLOCK]', '🚫': '[BLOCK]', '🚨': '[ALERT]', '🔄': '[SYNC]',
                '🔗': '[LINK]', '🔍': '[SEARCH]', '📉': '[DOWN]', '🧹': '[CLEAN]',
                '📝': '[NOTE]', '✓': 'OK', '✗': 'X', '→': '->', '⏳': '[WAIT]', '📨': '[MAIL]'
            }
            safe_msg = msg
            for emoji, replacement in emoji_replacements.items():
                safe_msg = safe_msg.replace(emoji, replacement)
            stream.write(safe_msg + self.terminator)
            self.flush()
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # If encoding still fails, replace all problematic characters
            try:
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                # Apply emoji replacements
                emoji_replacements = {
                    '🛑': '[STOP]', '✅': '[OK]', '❌': '[X]', '⚠️': '[WARN]',
                    '🗑️': '[DELETE]', '🚀': '[START]', '📊': '[CHART]', '💰': '[MONEY]',
                    '🔴': '[RED]', '🟢': '[GREEN]', '🟡': '[YELLOW]', '⚡': '[FAST]',
                    '🕯️': '[CANDLE]', '📌': '[INFO]', '🎯': '[TARGET]', '⏰': '[TIME]',
                    '🕘': '[CLOCK]', '🚫': '[BLOCK]', '🚨': '[ALERT]', '🔄': '[SYNC]',
                    '🔗': '[LINK]', '🔍': '[SEARCH]', '📉': '[DOWN]', '🧹': '[CLEAN]'
                }
                for emoji, replacement in emoji_replacements.items():
                    safe_msg = safe_msg.replace(emoji, replacement)
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                # Last resort: write ASCII-only message
                try:
                    ascii_msg = record.getMessage().encode('ascii', errors='replace').decode('ascii')
                    stream.write(f"{record.levelname}: {ascii_msg}\n")
                    self.flush()
                except Exception:
                    self.handleError(record)
        except Exception:
            self.handleError(record)

# Setup file logging:
# - `output/production_logs.txt`: historical log (multi-day, append)
# - `logs/dynamic_atm_strike.log`: daily "terminal mirror" (keeps ONLY current day)
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
production_log_path = output_dir / 'production_logs.txt'

logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)
# Add trading day's date to log filename to prevent overwriting
trading_date_suffix = datetime.now().strftime('%b%d').lower()  # e.g., "dec30"
daily_terminal_log_path = logs_dir / f'dynamic_atm_strike_{trading_date_suffix}.log'

def _extract_date_from_log_prefix(line: str):
    """
    Try to extract YYYY-MM-DD from the start of a log line or session header line.
    Returns datetime.date or None.
    """
    try:
        s = (line or "").strip()
        if not s:
            return None
        # Common log format: "YYYY-MM-DD HH:MM:SS,mmm - ..."
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        # Session header format: "NEW SESSION STARTED: YYYY-MM-DD HH:MM:SS"
        marker = "NEW SESSION STARTED:"
        if marker in s:
            idx = s.find(marker) + len(marker)
            rest = s[idx:].strip()
            if len(rest) >= 10 and rest[4] == "-" and rest[7] == "-":
                return datetime.strptime(rest[:10], "%Y-%m-%d").date()
    except Exception:
        return None
    return None

def _truncate_daily_log_if_not_today(path: Path):
    """
    Ensure the daily terminal log only contains the current day.
    We can't rely on mtime because an old multi-day file may have been appended today.
    """
    try:
        if not path.exists():
            return
        today = datetime.now().date()
        first_date = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # scan a few lines to find the first dated line
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                d = _extract_date_from_log_prefix(line)
                if d:
                    first_date = d
                    break
        if first_date and first_date != today:
            path.write_text("", encoding="utf-8")
    except Exception:
        # Never fail startup due to log housekeeping
        pass

_truncate_daily_log_if_not_today(daily_terminal_log_path)

# Root logger (handlers are attached below)
root_logger = logging.getLogger()

def _file_handler_exists_for(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
        return any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved for h in root_logger.handlers)
    except Exception:
        return False

def _make_file_handler(path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    return handler

production_file_handler = None if _file_handler_exists_for(production_log_path) else _make_file_handler(production_log_path)
daily_file_handler = None if _file_handler_exists_for(daily_terminal_log_path) else _make_file_handler(daily_terminal_log_path)

# Add session separator to BOTH files (append-mode). This makes it easy to compare sessions later.
try:
    for p in (production_log_path, daily_terminal_log_path):
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"NEW SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
except Exception:
    pass

# Configure logging to handle Unicode on Windows
if sys.platform == 'win32':
    # Try to reconfigure stdout/stderr for UTF-8
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # If reconfigure fails, continue with default
    
    # Create a safe Unicode handler for console
    console_handler = SafeUnicodeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Configure root logger with both console and file handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    if production_file_handler:
        root_logger.addHandler(production_file_handler)
    if daily_file_handler:
        root_logger.addHandler(daily_file_handler)
    
    # Don't suppress handlers - we want both console and file
else:
    # Linux/Unix: Use basicConfig with both handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    if production_file_handler:
        root_logger.addHandler(production_file_handler)
    if daily_file_handler:
        root_logger.addHandler(daily_file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)

# Log startup message to both console and file
if production_file_handler or daily_file_handler:
    logger.info(f"{'='*80}")
    logger.info(f"PRODUCTION LOGGING STARTED - Logs will be appended to: {production_log_path}")
    logger.info(f"DAILY TERMINAL LOG STARTED - Logs will be appended to: {daily_terminal_log_path} (auto-truncated per day)")
    logger.info(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")


class AsyncTradingBot:
    """
    Main async trading bot that orchestrates all components
    using event-driven architecture
    """
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.trade_symbols: Dict[str, Any] = {}

        # Core components
        self.kite = None
        self.state_manager: Optional[TradeStateManager] = None
        self.indicator_manager: Optional[IndicatorManager] = None
        self.strategy_executor: Optional[StrategyExecutor] = None
        self.ticker_handler: Optional[AsyncLiveTickerHandler] = None
        self.api_server: Optional[AsyncAPIServer] = None
        self.event_handlers: Optional[AsyncEventHandlers] = None
        self.event_dispatcher = get_event_dispatcher()
        self.entry_condition_manager: Optional[EntryConditionManager] = None
        self.dynamic_atm_manager: Optional[DynamicATMStrikeManager] = None
        
        # Market sentiment manager (for automated sentiment detection)
        self.market_sentiment_manager = None
        self.use_automated_sentiment = False  # Will be loaded from config
        
        # Trailing max drawdown components
        self.trade_ledger = None
        self.trailing_drawdown_manager = None

        # Control flags
        self.is_running = False
        self.is_initialized = False
        self.use_dynamic_atm = False  # Will be loaded from config
        self.trading_block_active = False
        self.trading_block_reason = None
        self.trading_block_source = None
        self.trading_block_date = None
        self.cpr_width_value = None
        
        # PE & CE initialization flags
        self.strikes_derived = False
        self.nifty_opening_price = None
        
        # Task references
        self.gtt_status_check_task = None
        self.entry_condition_check_task = None
        self.risk_watchdog_task = None
        self.drawdown_watchdog_task = None

    # Modify the initialize method to call the new method
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("[START] Initializing Async Trading Bot...")

            # Load configuration + symbols
            await self._load_configuration()

            # Initialize Kite API
            await self._initialize_kite_api()

            # Initialize trailing max drawdown components (before state manager)
            await self._initialize_trailing_drawdown()

            # Initialize state manager
            await self._initialize_state_manager()

            # Initialize indicator manager
            await self._initialize_indicator_manager()

            # Initialize API server
            await self._initialize_api_server()
            
            # Start API server immediately (before waiting for market open)
            # This allows control panel to connect even when bot is waiting for market
            await self.api_server.start()
            logger.info("[API] API server started early - control panel can connect now")

            # Initialize event handlers
            await self._initialize_event_handlers()
            
            # CRITICAL: Start event dispatcher BEFORE waiting for market open
            # This allows commands to be processed even while waiting for market
            await self.event_dispatcher.start()
            logger.info("[EVENT] Event dispatcher started early - commands can be processed now")

            # Initialize WebSocket handler (this may wait for market open)
            await self._initialize_websocket_handler()

            # Initialize strategy executor
            await self._initialize_strategy_executor()
            
            # Initialize dynamic ATM manager if enabled
            if self.use_dynamic_atm:
                await self._initialize_dynamic_atm_manager()
            
            # Initialize market sentiment manager if enabled
            if self.use_automated_sentiment:
                await self._initialize_market_sentiment_manager()
            
            # Initialize entry condition manager if strikes are already derived
            if self.strikes_derived:
                await self._initialize_entry_condition_manager()
                
                # CRITICAL: Wire entry_condition_manager to strategy_executor so it can clear SKIP_FIRST flags
                if self.entry_condition_manager and self.strategy_executor:
                    self.strategy_executor.entry_condition_manager = self.entry_condition_manager
                
                # Wire ticker_handler to entry_condition_manager for SKIP_FIRST feature
                if self.entry_condition_manager and self.ticker_handler:
                    self.entry_condition_manager.ticker_handler = self.ticker_handler
                    
                    # Initialize SKIP_FIRST daily values at market open (if enabled)
                    if self.entry_condition_manager.skip_first:
                        try:
                            await self.entry_condition_manager._initialize_daily_skip_first_values()
                        except Exception as e:
                            logger.warning(f"SKIP_FIRST: Could not initialize daily values at startup: {e}")

            # Wire context objects to event handlers for user commands
            self.event_handlers.strategy_executor = self.strategy_executor
            self.event_handlers.ticker_handler = self.ticker_handler
            self.event_handlers.trade_symbols = self.trade_symbols
            self.event_handlers.trading_bot = self  # Wire bot reference for API server access
            
            # Wire entry_condition_manager to event handlers if available
            if self.entry_condition_manager:
                self.event_handlers.entry_condition_manager = self.entry_condition_manager
                # CRITICAL: Wire entry_condition_manager to strategy_executor so it can clear SKIP_FIRST flags
                if self.strategy_executor:
                    self.strategy_executor.entry_condition_manager = self.entry_condition_manager
            
            # Wire position manager components if enabled
            if self.event_handlers.position_manager:
                self.event_handlers.position_manager.ticker_handler = self.ticker_handler
                # Update symbol-token mapping
                if self.ticker_handler:
                    symbol_token_map = self.ticker_handler.symbol_token_map
                    self.event_handlers.position_manager.update_symbol_token_mapping(symbol_token_map)
                # Start position manager
                await self.event_handlers.position_manager.start()
                logger.info("[OK] Real-time position manager started")

            # Force a full reconciliation on startup to ensure clean state
            if self.state_manager and self.kite:
                try:
                    logger.info("Performing full reconciliation during initialization...")
                    positions = self.kite.positions()
                    self.state_manager.reconcile_trades_with_broker(positions)
                    logger.info("Full reconciliation completed successfully")
                    
                    # Reset crossover indices to ensure clean state
                    if self.entry_condition_manager:
                        logger.info("Resetting crossover indices during initialization...")
                        self.entry_condition_manager._reset_crossover_indices()
                        logger.info("Crossover indices reset successfully")
                except Exception as e:
                    logger.warning(f"Could not perform full reconciliation during initialization: {e}")

            # Mark bot as running for background tasks and start watchdog loops
            self.is_initialized = True
            self.is_running = True

            # Start risk watchdog (stuck position protection)
            try:
                if not self.risk_watchdog_task:
                    self.risk_watchdog_task = asyncio.create_task(self._risk_watchdog_loop())
                    logger.info("[RISK] Risk watchdog loop started")
            except Exception as e:
                logger.warning(f"Could not start risk watchdog loop: {e}")

            # Start MARK2MARKET live drawdown watchdog if enabled
            try:
                if self.trailing_drawdown_manager and self.trailing_drawdown_manager.enabled and not self.drawdown_watchdog_task:
                    self.drawdown_watchdog_task = asyncio.create_task(self._drawdown_watchdog_loop())
                    logger.info("[RISK] Live MARK2MARKET drawdown watchdog loop started")
            except Exception as e:
                logger.warning(f"Could not start drawdown watchdog loop: {e}")

            logger.info("[OK] Async Trading Bot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def _risk_watchdog_loop(self):
        """
        Periodically reconcile broker positions, state manager trades, and
        real-time position manager to detect and flatten any unmanaged or
        inconsistent positions.
        """
        logger.info("[RISK] Risk watchdog loop running")
        # Run as long as the bot is considered running
        while self.is_running:
            try:
                # Require both broker API and state manager
                if not (self.kite and self.state_manager):
                    await asyncio.sleep(5.0)
                    continue

                # Fetch broker positions (off the event loop thread)
                positions = await asyncio.to_thread(self.kite.positions)
                net_positions = positions.get('net', []) if isinstance(positions, dict) else []

                # Collect open NFO positions by tradingsymbol
                open_symbols = []
                for p in net_positions:
                    try:
                        if p.get('exchange') != 'NFO':
                            continue
                        qty = p.get('quantity', 0) or 0
                        if qty == 0:
                            continue
                        tsym = p.get('tradingsymbol')
                        if tsym:
                            open_symbols.append(tsym)
                    except Exception:
                        continue

                if not open_symbols:
                    await asyncio.sleep(5.0)
                    continue

                # Get active trades from state manager
                try:
                    active_trades = self.state_manager.get_active_trades() or {}
                except Exception as e:
                    logger.warning(f"[RISK] Could not read active trades for watchdog: {e}")
                    await asyncio.sleep(5.0)
                    continue

                # Position manager reference (may be None if disabled)
                position_manager = getattr(self.event_handlers, "position_manager", None) if self.event_handlers else None

                for symbol in open_symbols:
                    trade = active_trades.get(symbol)
                    pm_active = position_manager.is_position_active(symbol) if position_manager else False

                    # CRITICAL: Verify broker position actually exists and has quantity > 0
                    # Sometimes broker positions list includes stale entries
                    broker_pos = next(
                        (p for p in net_positions 
                         if p.get('tradingsymbol') == symbol and p.get('exchange') == 'NFO'),
                        None
                    )
                    broker_qty = broker_pos.get('quantity', 0) if broker_pos else 0
                    
                    # If broker position doesn't exist or quantity is 0, skip (position is already closed)
                    if not broker_pos or broker_qty == 0:
                        logger.debug(
                            f"[RISK] Watchdog: {symbol} broker position is closed (qty={broker_qty}). "
                            f"Skipping - position already closed."
                        )
                        continue

                    # CRITICAL: Skip if trade is already closed AND position manager says inactive
                    # This handles cases where exit was already executed but broker position hasn't updated yet
                    # BUT: Only skip if we're confident broker position will update soon
                    # If broker position persists, it's a real inconsistency that needs fixing
                    if not trade and not pm_active:
                        # Both are False - our system says closed, but broker still shows position
                        # This could be:
                        # 1. Broker position is stale (order executed but not reflected yet) - wait
                        # 2. Real inconsistency (order failed, position still open) - must exit
                        # For now, we'll trigger exit to be safe - executor will verify broker position
                        logger.warning(
                            f"[RISK] Watchdog: {symbol} system says closed (trade_present=False, pm_active=False) "
                            f"but broker shows position (qty={broker_qty}). This may be stale or real inconsistency."
                        )
                        # Continue to trigger exit - executor will verify and handle appropriately

                    # CRITICAL: Grace period for newly entered trades
                    # Newly entered trades need time to:
                    # 1. Get entry price from broker (200-600ms)
                    # 2. Update entry price in state
                    # 3. Register with position manager
                    # Give them 3 seconds grace period before watchdog checks
                    if trade:
                        created_at = trade.get('created_at')
                        if created_at:
                            age_seconds = time.time() - created_at
                            if age_seconds < 3.0:  # Grace period: 3 seconds
                                logger.debug(
                                    f"[RISK] Watchdog: {symbol} is newly entered (age={age_seconds:.2f}s). "
                                    f"Skipping check during grace period. trade_present={bool(trade)}, pm_active={pm_active}"
                                )
                                continue  # Skip this check - trade is still setting up
                    
                    # Stuck/unmanaged if: broker shows open position but either
                    # 1) no trade in state, or 2) no active monitoring in position manager.
                    if not trade or not pm_active:
                        logger.critical(
                            f"[RISK] Watchdog detected unmanaged or inconsistent position for {symbol} "
                            f"(trade_present={bool(trade)}, pm_active={pm_active}). Triggering safety exit."
                        )
                        
                        # CRITICAL: Get current LTP for trigger_price to prevent None errors
                        trigger_price = None
                        try:
                            if self.ticker_handler:
                                # Try to get LTP from ticker handler
                                token = self.ticker_handler.get_token_by_symbol(symbol)
                                if token:
                                    trigger_price = self.ticker_handler.get_ltp(token)
                        except Exception as e:
                            logger.debug(f"[RISK] Could not get LTP for {symbol} in watchdog: {e}")
                        
                        # If LTP unavailable, try to get from broker position or use a fallback
                        if trigger_price is None or trigger_price <= 0:
                            # Try to get average price from broker position as fallback
                            try:
                                broker_pos = next(
                                    (p for p in net_positions 
                                     if p.get('tradingsymbol') == symbol and p.get('exchange') == 'NFO'),
                                    None
                                )
                                if broker_pos:
                                    avg_price = broker_pos.get('average_price')
                                    if avg_price and avg_price > 0:
                                        trigger_price = avg_price
                                        logger.debug(f"[RISK] Using broker avg_price as trigger_price for {symbol}: {trigger_price}")
                            except Exception:
                                pass
                        
                        # Dispatch a high-priority EXIT_SIGNAL for this symbol
                        try:
                            from event_system import Event, EventType
                            self.event_dispatcher.dispatch_event(
                                Event(
                                    EventType.EXIT_SIGNAL,
                                    {
                                        "symbol": symbol,
                                        "exit_reason": "RISK_WATCHDOG",
                                        "trigger_price": trigger_price,  # CRITICAL: Include trigger_price (may be None, but handled gracefully)
                                        "order_type": "MARKET",
                                        "priority": "HIGH",
                                    },
                                    source="risk_watchdog",
                                )
                            )
                        except Exception as e:
                            logger.warning(f"[RISK] Failed to dispatch watchdog EXIT_SIGNAL for {symbol}: {e}")

            except asyncio.CancelledError:
                logger.info("[RISK] Risk watchdog loop cancelled")
                break
            except Exception as e:
                logger.warning(f"[RISK] Risk watchdog loop error: {e}", exc_info=True)

            await asyncio.sleep(5.0)

    async def _drawdown_watchdog_loop(self):
        """
        Periodically evaluate live mark-to-market equity against MARK2MARKET
        drawdown limits. On breach, trigger a global FORCE_EXIT and block new
        trades for the rest of the session.
        """
        logger.info("[RISK] Live MARK2MARKET drawdown watchdog loop running")
        while self.is_running:
            try:
                if not (self.trailing_drawdown_manager and self.trailing_drawdown_manager.enabled):
                    await asyncio.sleep(10.0)
                    continue

                if not (self.state_manager and self.kite):
                    await asyncio.sleep(10.0)
                    continue

                # Get open trades from state manager
                try:
                    open_trades = self.state_manager.get_active_trades() or {}
                except Exception as e:
                    logger.warning(f"[RISK] Could not read active trades for drawdown watchdog: {e}")
                    await asyncio.sleep(10.0)
                    continue

                if not open_trades:
                    await asyncio.sleep(10.0)
                    continue

                # Get current prices from broker positions snapshot
                positions = await asyncio.to_thread(self.kite.positions)
                net_positions = positions.get('net', []) if isinstance(positions, dict) else []
                current_prices: Dict[str, float] = {}
                for p in net_positions:
                    try:
                        if p.get('exchange') != 'NFO':
                            continue
                        qty = p.get('quantity', 0) or 0
                        if qty == 0:
                            continue
                        tsym = p.get('tradingsymbol')
                        # Prefer last_price; fall back to average_price if needed
                        ltp = p.get('last_price') or p.get('average_price')
                        if tsym and ltp:
                            current_prices[tsym] = float(ltp)
                    except Exception:
                        continue

                if not current_prices:
                    await asyncio.sleep(10.0)
                    continue

                is_allowed, reason, state = self.trailing_drawdown_manager.check_realtime_drawdown(
                    open_trades, current_prices
                )

                if is_allowed:
                    await asyncio.sleep(10.0)
                    continue

                # Drawdown breached – only act once per session
                if self.trading_block_active and self.trading_block_source == "MARK2MARKET":
                    await asyncio.sleep(10.0)
                    continue

                logger.critical(f"[RISK] Live MARK2MARKET drawdown breach detected. State: {state}")
                # Block further trading for the rest of the session
                self.trading_block_active = True
                self.trading_block_reason = reason
                self.trading_block_source = "MARK2MARKET"
                self.trading_block_date = datetime.now().date()

                # Trigger a global FORCE_EXIT via the existing user command path
                try:
                    from event_system import Event, EventType
                    self.event_dispatcher.dispatch_event(
                        Event(
                            EventType.USER_COMMAND,
                            {"command": "FORCE_EXIT"},
                            source="mark2market_drawdown",
                        )
                    )
                    logger.critical("[RISK] FORCE_EXIT command dispatched due to MARK2MARKET breach")
                except Exception as e:
                    logger.error(f"[RISK] Failed to dispatch FORCE_EXIT on drawdown breach: {e}", exc_info=True)

            except asyncio.CancelledError:
                logger.info("[RISK] Drawdown watchdog loop cancelled")
                break
            except Exception as e:
                logger.warning(f"[RISK] Drawdown watchdog loop error: {e}", exc_info=True)

            await asyncio.sleep(10.0)

    def _migrate_sentiment_config(self):
        """Migrate old MARKET_SENTIMENT config format to new simplified format"""
        market_sentiment_config = self.config.get('MARKET_SENTIMENT', {})
        
        # Check if migration is needed (old format has ENABLED and MANUAL_MARKET_SENTIMENT)
        if 'ENABLED' in market_sentiment_config or 'MANUAL_MARKET_SENTIMENT' in market_sentiment_config:
            logger.info("[MIGRATION] Detected old MARKET_SENTIMENT config format - migrating to new format...")
            
            old_enabled = market_sentiment_config.get('ENABLED', False)
            old_manual = market_sentiment_config.get('MANUAL_MARKET_SENTIMENT', False)
            
            # Determine new MODE
            if not old_enabled:
                new_mode = 'DISABLE'
                new_manual_sentiment = None
            elif old_manual:
                new_mode = 'MANUAL'
                new_manual_sentiment = market_sentiment_config.get('MANUAL_SENTIMENT', 'NEUTRAL')
            else:
                new_mode = 'AUTO'
                new_manual_sentiment = None
            
            # Update config with new format
            if 'MODE' not in market_sentiment_config:
                market_sentiment_config['MODE'] = new_mode
                logger.info(f"[MIGRATION] Set MODE to {new_mode}")
            
            if new_mode == 'MANUAL' and 'MANUAL_SENTIMENT' not in market_sentiment_config:
                market_sentiment_config['MANUAL_SENTIMENT'] = new_manual_sentiment or 'NEUTRAL'
                logger.info(f"[MIGRATION] Set MANUAL_SENTIMENT to {market_sentiment_config['MANUAL_SENTIMENT']}")
            
            # Keep VERSION and CONFIG_PATH if they exist
            if 'VERSION' in market_sentiment_config and 'SENTIMENT_VERSION' not in market_sentiment_config:
                market_sentiment_config['SENTIMENT_VERSION'] = market_sentiment_config['VERSION']
            
            # Log migration summary
            logger.info(f"[MIGRATION] Migration complete:")
            logger.info(f"  - Old: ENABLED={old_enabled}, MANUAL_MARKET_SENTIMENT={old_manual}")
            logger.info(f"  - New: MODE={new_mode}" + (f", MANUAL_SENTIMENT={new_manual_sentiment}" if new_manual_sentiment else ""))
            logger.info(f"[MIGRATION] Old fields (ENABLED, MANUAL_MARKET_SENTIMENT) are deprecated but kept for backward compatibility")

    async def _load_configuration(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")

            # Load dynamic ATM configuration
            self.use_dynamic_atm = self.config.get('DYNAMIC_ATM', {}).get('ENABLED', False)
            logger.info(f"Dynamic ATM enabled: {self.use_dynamic_atm}")
            
            # Migrate old MARKET_SENTIMENT config format to new format
            self._migrate_sentiment_config()
            
            # Load sentiment mode configuration (new simplified architecture)
            market_sentiment_config = self.config.get('MARKET_SENTIMENT', {})
            sentiment_mode = market_sentiment_config.get('MODE', 'MANUAL').upper()
            
            # Determine if automated sentiment should be used
            self.use_automated_sentiment = (sentiment_mode == 'AUTO')
            logger.info(f"Sentiment mode: {sentiment_mode}")
            logger.info(f"Automated market sentiment enabled: {self.use_automated_sentiment}")

            # Load trading symbols (if file exists, otherwise will be created during initialization)
            try:
                with open(self.config['SUBSCRIBE_TOKENS_FILE_PATH'], 'r') as f:
                    self.trade_symbols = json.load(f)
                logger.info(f"Trading symbols loaded: {list(self.trade_symbols.keys())}")
            except FileNotFoundError:
                logger.info("Trading symbols file not found - will be created during initialization")
                self.trade_symbols = {}

        except FileNotFoundError as e:
            logger.critical(f"Configuration file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    async def _initialize_kite_api(self):
        """Initialize Kite API connection"""
        try:
            self.kite, _, _ = get_kite_api_instance()
            logger.info("Kite API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kite API: {e}")
            raise

    async def _initialize_trailing_drawdown(self):
        """Initialize trailing max drawdown components"""
        try:
            from trade_ledger import TradeLedger
            from trailing_max_drawdown_manager import TrailingMaxDrawdownManager
            
            # Initialize trade ledger
            # Use default ledger path with date suffix (e.g., ledger_dec30.txt)
            self.trade_ledger = TradeLedger()
            logger.info("Trade ledger initialized")
            
            # Initialize trailing drawdown manager
            self.trailing_drawdown_manager = TrailingMaxDrawdownManager(
                self.config, 
                self.trade_ledger
            )
            logger.info("Trailing max drawdown manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize trailing drawdown components: {e}")
            # Don't raise - allow bot to continue without trailing stop
            logger.warning("Bot will continue without trailing max drawdown protection")
            self.trade_ledger = None
            self.trailing_drawdown_manager = None

    async def _initialize_state_manager(self):
        """Initialize trade state manager"""
        try:
            import os
            state_file_path = self.config['TRADE_STATE_FILE_PATH']
            state_file_exists = os.path.exists(state_file_path)
            
            self.state_manager = TradeStateManager(
                state_file_path,
                trade_ledger=self.trade_ledger
            )

            # Load existing state
            self.state_manager.load_state()
            
            # Get config values
            market_sentiment_config = self.config.get('MARKET_SENTIMENT', {})
            config_sentiment_mode = market_sentiment_config.get('MODE', 'MANUAL').upper()
            config_manual_sentiment = market_sentiment_config.get('MANUAL_SENTIMENT', 'NEUTRAL').upper()
            
            # Get current state values
            current_mode = self.state_manager.get_sentiment_mode()
            current_sentiment = self.state_manager.get_sentiment()
            
            # Always sync with config.yaml on startup - config.yaml is the source of truth
            # 1. If state file didn't exist (first run) - initialize from config
            # 2. If mode is invalid - initialize from config
            # 3. If config mode is MANUAL - ALWAYS sync sentiment from config (config.yaml is source of truth for MANUAL mode)
            # 4. If config mode is DISABLE - sync to DISABLE
            # 5. If config mode is AUTO - sync to AUTO (sentiment will be set by algorithm later)
            should_initialize_from_config = (
                not state_file_exists or
                not current_mode or 
                current_mode not in ['AUTO', 'MANUAL', 'DISABLE']
            )
            
            # For MANUAL mode: Always sync sentiment from config.yaml on startup (config is source of truth)
            # This ensures that config.yaml settings are always respected on restart
            if config_sentiment_mode == 'MANUAL':
                if should_initialize_from_config or current_mode != 'MANUAL' or current_sentiment != config_manual_sentiment:
                    self.state_manager.set_sentiment_mode('MANUAL', config_manual_sentiment)
                    logger.info(f"Synced sentiment from config.yaml: MANUAL mode with sentiment {config_manual_sentiment} "
                              f"(previous: mode={current_mode}, sentiment={current_sentiment})")
            elif config_sentiment_mode == 'DISABLE':
                if should_initialize_from_config or current_mode != 'MANUAL' or current_sentiment != 'DISABLE':
                    self.state_manager.set_sentiment_mode('MANUAL', 'DISABLE')
                    logger.info(f"Synced sentiment from config.yaml: DISABLE mode "
                              f"(previous: mode={current_mode}, sentiment={current_sentiment})")
            elif config_sentiment_mode == 'AUTO':
                if should_initialize_from_config or current_mode != 'AUTO':
                    self.state_manager.set_sentiment_mode('AUTO')
                    self.state_manager.set_sentiment('NEUTRAL')  # Default until algorithm calculates
                    logger.info(f"Synced sentiment from config.yaml: AUTO mode "
                              f"(previous: mode={current_mode}, sentiment={current_sentiment})")

            # Reconcile with broker
            try:
                current_positions = self.kite.positions()
                self.state_manager.reconcile_trades_with_broker(current_positions)
            except Exception as e:
                logger.warning(f"Could not reconcile trades with broker: {e}")

            logger.info("State manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {e}")
            raise

    async def _initialize_indicator_manager(self):
        """Initialize indicator manager"""
        try:
            self.indicator_manager = IndicatorManager(self.config)
            logger.info("Indicator manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize indicator manager: {e}")
            raise

    async def _initialize_api_server(self):
        """Initialize async API server"""
        try:
            self.api_server = get_async_api_server()

            # Initialize with trade settings
            await self.api_server.initialize_config(self.config['TRADE_SETTINGS'])
            
            # Initialize MARKET_SENTIMENT settings for dynamic updates (legacy support)
            market_sentiment_config = self.config.get('MARKET_SENTIMENT', {})
            # Keep MANUAL_MARKET_SENTIMENT for backward compatibility (deprecated)
            await self.api_server.initialize_config({
                'MANUAL_MARKET_SENTIMENT': market_sentiment_config.get('MANUAL_MARKET_SENTIMENT', False)
            })

            logger.info("API server initialized")
        except Exception as e:
            logger.error(f"Failed to initialize API server: {e}")
            raise

    async def _initialize_event_handlers(self):
        """Initialize event handlers"""
        try:
            self.event_handlers = get_async_event_handlers()

            # Initialize handlers with dependencies
            self.event_handlers.kite = self.kite
            self.event_handlers.state_manager = self.state_manager
            self.event_handlers.config = self.config
            self.event_handlers.trade_symbols = self.trade_symbols  # ensure present
            self.event_handlers.trading_bot = self  # CRITICAL: Set trading_bot reference for handlers

            # Register all handlers
            self.event_handlers.initialize()

            logger.info("Event handlers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize event handlers: {e}")
            raise

    async def _initialize_websocket_handler(self):
        """Initialize WebSocket handler with simplified flow"""
        try:
            # Check market timing to determine initialization strategy
            market_is_open = is_market_open_time()
            
            if market_is_open:
                # Market is open - use historical API to get opening price
                logger.info("[CLOCK] Market is already open. Using historical API to get opening price...")
                await self._initialize_strikes_from_historical()
            else:
                # Before market open - show countdown and wait
                logger.info("[CLOCK] Market not yet open. Waiting for market to open...")
                await self._wait_for_market_open()
                # After market opens, get opening price from historical data
                await self._initialize_strikes_from_historical()

            # Determine initial subscription based on whether strikes are derived
            initial_symbol_token_map = {}
            
            if self.strikes_derived:
                # Strikes are ready - subscribe to CE and PE tokens
                initial_symbol_token_map = {
                    self.trade_symbols['ce_symbol']: self.trade_symbols['ce_token'],
                    self.trade_symbols['pe_symbol']: self.trade_symbols['pe_token']
                }
                
                # Check if SKIP_FIRST is enabled (needs NIFTY for sentiment calculation)
                skip_first_enabled = False
                if self.entry_condition_manager:
                    skip_first_enabled = self.entry_condition_manager.skip_first
                elif self.config.get('TRADE_SETTINGS', {}).get('SKIP_FIRST', False):
                    skip_first_enabled = True
                
                # Add NIFTY 50 token if dynamic ATM, automated sentiment, or SKIP_FIRST is enabled
                if self.use_dynamic_atm or self.use_automated_sentiment or skip_first_enabled:
                    nifty_token = 256265  # NIFTY 50 token
                    initial_symbol_token_map['NIFTY 50'] = nifty_token
                    reasons = []
                    if self.use_dynamic_atm:
                        reasons.append("Dynamic ATM")
                    if self.use_automated_sentiment:
                        reasons.append("Automated Sentiment")
                    if skip_first_enabled:
                        reasons.append("SKIP_FIRST")
                    logger.info(f"[OK] Strikes derived. Subscribing to CE, PE, and NIFTY 50 tokens ({', '.join(reasons)} enabled).")
                else:
                    logger.info("[OK] Strikes derived. Subscribing to CE and PE tokens only (Static ATM, Manual Sentiment, SKIP_FIRST disabled).")
            else:
                # Strikes not derived yet - subscribe only to NIFTY to get opening price from first candle
                nifty_token = 256265  # NIFTY 50 token
                initial_symbol_token_map['NIFTY 50'] = nifty_token
                logger.warning("[WARN] Strikes not derived from historical API. Will derive from first NIFTY candle.")
                logger.info("[CHART] Subscribing to NIFTY 50 only. CE/PE tokens will be added after first candle.")

            self.ticker_handler = AsyncLiveTickerHandler(
                self.kite,
                initial_symbol_token_map,
                self.indicator_manager,
                ce_symbol=self.trade_symbols.get('ce_symbol') if self.strikes_derived else None,
                pe_symbol=self.trade_symbols.get('pe_symbol') if self.strikes_derived else None
            )

            # Set reference to bot for callbacks
            self.ticker_handler.trading_bot = self

            logger.info("WebSocket handler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket handler: {e}")
            raise

    async def _initialize_strategy_executor(self):
        """Initialize strategy executor"""
        try:
            self.strategy_executor = StrategyExecutor(
                self.kite,
                self.state_manager,
                self.config,
                self.ticker_handler,
                trade_ledger=self.trade_ledger,
                trailing_drawdown_manager=self.trailing_drawdown_manager
            )

            # Set in event handlers (again, after creation)
            self.event_handlers.strategy_executor = self.strategy_executor

            logger.info("Strategy executor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize strategy executor: {e}")
            raise

    async def _initialize_dynamic_atm_manager(self):
        """Initialize dynamic ATM strike manager"""
        try:
            logger.info("Initializing Dynamic ATM Strike Manager...")
            
            # Create instance
            self.dynamic_atm_manager = DynamicATMStrikeManager(self.config_path)
            self.dynamic_atm_manager.kite = self.kite
            
            # Get initial NIFTY price: mid-day boot use latest completed candle; otherwise opening price
            opening_price = None
            if is_market_open_time():
                opening_price = get_nifty_latest_calculated_price_historical(self.kite)
            if opening_price is None:
                opening_price = get_nifty_opening_price_historical(self.kite)
            if opening_price is None:
                logger.warning("Could not get NIFTY opening price - will use first candle price")
                # Will be set when first NIFTY candle is received
                opening_price = 0.0
            
            # Calculate initial strikes based on STRIKE_TYPE configuration
            ce_strike, pe_strike = calculate_strikes(opening_price)
            
            # Initialize state
            self.dynamic_atm_manager.current_active_ce = ce_strike
            self.dynamic_atm_manager.current_active_pe = pe_strike
            self.dynamic_atm_manager.current_nifty_price = opening_price
            
            # Set minimum slab change interval from config
            min_interval = self.config.get('DYNAMIC_ATM', {}).get('MIN_SLAB_CHANGE_INTERVAL', 60)
            self.dynamic_atm_manager.min_slab_change_interval = min_interval
            self.dynamic_atm_manager.last_slab_change_time = None
            self.dynamic_atm_manager.last_slab_change_price = None  # Initialize price tolerance tracking
            
            logger.info(f"[OK] Dynamic ATM Manager initialized: CE={ce_strike}, PE={pe_strike} (NIFTY={opening_price})")
            
        except Exception as e:
            logger.error(f"Failed to initialize dynamic ATM manager: {e}")
            logger.warning("Falling back to static ATM mode")
            self.use_dynamic_atm = False
            self.dynamic_atm_manager = None
    
    async def _initialize_market_sentiment_manager(self):
        """Initialize automated market sentiment manager with cold start processing"""
        try:
            logger.info("Initializing Automated Market Sentiment Manager...")
            
            # Get version from config (default to v1 for backward compatibility)
            sentiment_config = self.config.get('MARKET_SENTIMENT', {})
            version = sentiment_config.get('VERSION', 'v1')
            
            # Import the correct version
            if version == 'v2':
                from market_sentiment_v2.realtime_sentiment_manager import RealTimeMarketSentimentManager
                default_config_path = 'market_sentiment_v2/config.yaml'
                logger.info("Using Market Sentiment v2 (improved implementation)")
            else:
                from market_sentiment_v1.realtime_sentiment_manager import RealTimeMarketSentimentManager
                default_config_path = 'market_sentiment_v1/config.yaml'
                logger.info("Using Market Sentiment v1 (legacy implementation)")
            
            # Get config path for market sentiment (auto-update if VERSION is set but CONFIG_PATH doesn't match)
            sentiment_config_path = sentiment_config.get('CONFIG_PATH', default_config_path)
            # Ensure config path matches version
            if version == 'v2' and 'v1' in sentiment_config_path:
                sentiment_config_path = 'market_sentiment_v2/config.yaml'
                logger.warning(f"Auto-updated CONFIG_PATH to match VERSION={version}: {sentiment_config_path}")
            elif version == 'v1' and 'v2' in sentiment_config_path:
                sentiment_config_path = 'market_sentiment_v1/config.yaml'
                logger.warning(f"Auto-updated CONFIG_PATH to match VERSION={version}: {sentiment_config_path}")
            
            self.market_sentiment_manager = RealTimeMarketSentimentManager(sentiment_config_path, self.kite)
            
            # For v2: No cold start - analyzer initializes when first candle is processed (matches backtesting)
            # For v1: Perform cold start if method exists
            if version == 'v2':
                logger.info(f"[OK] Automated Market Sentiment Manager (v2) initialized. "
                          f"Analyzer will initialize when first candle is processed (matches backtesting behavior). "
                          f"(config: {sentiment_config_path})")
            else:
                # v1 still uses cold start
                current_time = datetime.now()
                logger.info(f"Performing cold start for market sentiment (current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")
                
                # Check if cold start method exists (for v1 compatibility)
                if hasattr(self.market_sentiment_manager, 'initialize_with_cold_start'):
                    # Run cold start in thread to avoid blocking
                    cold_start_success = await asyncio.to_thread(
                        self.market_sentiment_manager.initialize_with_cold_start,
                        current_time
                    )
                    
                    if cold_start_success:
                        initial_sentiment = self.market_sentiment_manager.get_current_sentiment()
                        logger.info(f"[OK] Automated Market Sentiment Manager initialized with cold start. "
                                  f"Initial sentiment: {initial_sentiment} (config: {sentiment_config_path})")
                        
                        # Sync initial sentiment to state manager so entry conditions use correct sentiment
                        if initial_sentiment and self.state_manager:
                            # Check if manual override is enabled before syncing
                            manual_override = False
                            if self.api_server:
                                try:
                                    # Check if we're in AUTO mode (new architecture)
                                    current_mode = self.state_manager.get_sentiment_mode()
                                    manual_override = (current_mode != 'AUTO')
                                except Exception as e:
                                    logger.debug(f"Could not check sentiment mode during initialization: {e}")
                                    manual_override = False
                            
                            # Only update sentiment if in AUTO mode
                            if current_mode == 'AUTO':
                                current_state_sentiment = self.state_manager.get_sentiment()
                                if current_state_sentiment != initial_sentiment:
                                    self.state_manager.set_sentiment(initial_sentiment)
                                    logger.info(f"Synced initial sentiment to state manager: {current_state_sentiment} -> {initial_sentiment}")
                                else:
                                    logger.debug(f"State manager already has correct sentiment: {initial_sentiment}")
                            else:
                                logger.info(f"Not in AUTO mode (current: {current_mode}) - not syncing automated sentiment (calculated: {initial_sentiment})")
                    else:
                        logger.error("Cold start failed - sentiment manager may not work correctly")
                        # Don't raise - allow bot to continue with manual sentiment
                        logger.warning("Bot will continue with manual sentiment control only")
                        self.use_automated_sentiment = False
                else:
                    logger.warning("Cold start method not available - sentiment manager will initialize on first candle")
            
            await self._evaluate_cpr_width_filter()

        except Exception as e:
            logger.error(f"Failed to initialize Market Sentiment Manager: {e}", exc_info=True)
            # Don't raise - allow bot to continue with manual sentiment
            logger.warning("Bot will continue with manual sentiment control only")
            self.use_automated_sentiment = False

    async def _evaluate_cpr_width_filter(self):
        """Evaluate CPR width and disable autonomous trading if it exceeds configured threshold."""
        filter_config = self.config.get('CPR_WIDTH_FILTER', {})
        if not filter_config.get('ENABLED', False):
            self._clear_trading_block()
            return

        if not self.market_sentiment_manager:
            logger.warning("CPR width filter enabled but market sentiment manager is not initialized.")
            return

        today = datetime.now().date()
        try:
            cpr_width = await asyncio.to_thread(
                self.market_sentiment_manager.get_cpr_width_for_date,
                today
            )
        except Exception as e:
            logger.warning(f"Failed to evaluate CPR width filter: {e}")
            return

        self.cpr_width_value = cpr_width
        threshold = filter_config.get('THRESHOLD', 60)

        if cpr_width is None:
            logger.warning("CPR width filter enabled but width could not be calculated. Trading remains enabled.")
            self._clear_trading_block()
            return

        if cpr_width > threshold:
            self._apply_trading_block(
                reason=f"CPR width {cpr_width:.2f} > {threshold}",
                source="CPR_WIDTH"
            )
        else:
            logger.info(f"CPR width {cpr_width:.2f} <= {threshold}. Autonomous trading enabled for today.")
            self._clear_trading_block()

    def _apply_trading_block(self, reason: str, source: str = "CPR_WIDTH"):
        """Disable autonomous trading and force sentiment to DISABLE."""
        if self.trading_block_active and self.trading_block_reason == reason:
            return

        self.trading_block_active = True
        self.trading_block_reason = reason
        self.trading_block_source = source
        self.trading_block_date = datetime.now().date()

        logger.warning(f"[TRADING BLOCK] {reason}. Autonomous trading disabled for today.")
        if self.state_manager:
            current_sentiment = self.state_manager.get_sentiment()
            if current_sentiment != "DISABLE":
                self.state_manager.set_sentiment("DISABLE")
                logger.info("Sentiment forced to DISABLE due to CPR width filter.")

    def _clear_trading_block(self):
        """Clear any active trading block."""
        if self.trading_block_active:
            logger.info("Trading block cleared.")
        self.trading_block_active = False
        self.trading_block_reason = None
        self.trading_block_source = None
        self.trading_block_date = None

    def is_trading_blocked(self) -> bool:
        return self.trading_block_active

    def get_trading_block_reason(self) -> Optional[str]:
        return self.trading_block_reason

    # Add this method after _initialize_strategy_executor
    async def _initialize_entry_condition_manager(self):
        """Initialize entry condition manager"""
        try:
            if self.strikes_derived:
                self.entry_condition_manager = EntryConditionManager(
                    self.kite,
                    self.state_manager,
                    self.strategy_executor,
                    self.indicator_manager,
                    self.config,
                    self.trade_symbols.get('ce_symbol'),
                    self.trade_symbols.get('pe_symbol'),
                    self.trade_symbols.get('underlying_symbol')
                )
                logger.info("Entry condition manager initialized")
            else:
                logger.warning("Cannot initialize entry condition manager yet - strikes not derived")
        except Exception as e:
            logger.error(f"Failed to initialize entry condition manager: {e}")
            raise

    async def start(self):
        """Start the trading bot"""
        # Delete support/resistance levels log file at startup (if exists)
        levels_log_file = 'output/support_resistance_levels.txt'
        try:
            import os
            if os.path.exists(levels_log_file):
                os.remove(levels_log_file)
                logger.info(f"Deleted existing support/resistance levels log file: {levels_log_file}")
        except Exception as e:
            logger.warning(f"Could not delete support/resistance levels log file: {e}")
        
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info("[START] Starting Async Trading Bot...")

            # Event dispatcher is already started during initialization (before market wait)
            # No need to start it again here
            logger.info(f"[VERIFY] Event dispatcher status. Running: {self.event_dispatcher.is_running}, Task: {self.event_dispatcher.event_processing_task is not None}")

            # Start WebSocket handler with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            connection_timeout = 30  # seconds
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Attempting WebSocket connection (attempt {attempt}/{max_retries})...")
                    await self.ticker_handler.start_ticker()
                    
                    # Wait for WebSocket connection with timeout
                    if await self.ticker_handler.wait_for_connection(timeout_seconds=connection_timeout):
                        logger.info("WebSocket connection established successfully")
                        break
                    else:
                        error_msg = f"Could not establish WebSocket connection (attempt {attempt}/{max_retries})"
                        if hasattr(self.ticker_handler, '_connection_error') and self.ticker_handler._connection_error:
                            error_msg += f": {self.ticker_handler._connection_error}"
                        
                        if attempt < max_retries:
                            logger.warning(f"{error_msg}. Retrying in {retry_delay} seconds...")
                            # Clean up failed connection attempt
                            try:
                                await self.ticker_handler.stop_ticker()
                            except:
                                pass
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise Exception(error_msg)
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"WebSocket connection attempt {attempt} failed: {e}. Retrying in {retry_delay} seconds...")
                        try:
                            await self.ticker_handler.stop_ticker()
                        except:
                            pass
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise

            # Update subscriptions based on whether strikes are already derived
            if self.strikes_derived:
                # Include CE and PE tokens, and NIFTY if automated sentiment or SKIP_FIRST is enabled
                strike_symbol_token_map = {
                    self.trade_symbols['ce_symbol']: self.trade_symbols['ce_token'],
                    self.trade_symbols['pe_symbol']: self.trade_symbols['pe_token']
                }
                
                # Check if SKIP_FIRST is enabled (needs NIFTY for sentiment calculation)
                skip_first_enabled = False
                if self.entry_condition_manager:
                    skip_first_enabled = self.entry_condition_manager.skip_first
                elif self.config.get('TRADE_SETTINGS', {}).get('SKIP_FIRST', False):
                    skip_first_enabled = True
                
                # Add NIFTY if automated sentiment or SKIP_FIRST is enabled
                if self.use_automated_sentiment or skip_first_enabled:
                    nifty_token = 256265
                    strike_symbol_token_map['NIFTY 50'] = nifty_token
                    reasons = []
                    if self.use_automated_sentiment:
                        reasons.append("automated sentiment")
                    if skip_first_enabled:
                        reasons.append("SKIP_FIRST")
                    logger.info(f"[CHART] Subscribed to CE, PE, and NIFTY 50 tokens ({', '.join(reasons)} enabled)")
                else:
                    logger.info("[CHART] Subscribed to CE and PE tokens only (NIFTY excluded - automated sentiment and SKIP_FIRST disabled)")
                
                await self.ticker_handler.update_subscriptions(strike_symbol_token_map)
                
                # Ensure entry condition manager is initialized
                if not self.entry_condition_manager:
                    await self._initialize_entry_condition_manager()
                
                # Wire entry_condition_manager to event handlers and ticker_handler
                if self.entry_condition_manager:
                    self.event_handlers.entry_condition_manager = self.entry_condition_manager
                    # CRITICAL: Wire entry_condition_manager to strategy_executor so it can clear SKIP_FIRST flags
                    if self.strategy_executor:
                        self.strategy_executor.entry_condition_manager = self.entry_condition_manager
                    # Wire ticker_handler for SKIP_FIRST feature
                    if self.ticker_handler:
                        self.entry_condition_manager.ticker_handler = self.ticker_handler
                        # Initialize SKIP_FIRST daily values at market open
                        if self.entry_condition_manager.skip_first:
                            await self.entry_condition_manager._initialize_daily_skip_first_values()
            else:
                # Only NIFTY 50 is subscribed initially - CE/PE will be added after opening price
                logger.info("[CHART] Subscribed to NIFTY 50 only. Waiting for opening price to derive strikes.")

            # Prefill historical data
            await self.ticker_handler.prefill_historical_data()

            # Start the GTT status check task
            self.gtt_status_check_task = asyncio.create_task(self._gtt_status_check_loop())
            logger.info("GTT status check task started")
            
            # Entry condition checks are now triggered by new candle events only
            # No need for a separate scheduled task
            logger.info("[OK] Entry conditions will be checked on new candle formation only")
            
            self.is_running = True
            logger.info("[OK] Async Trading Bot started successfully")

            # Main event loop
            await self._main_event_loop()

        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("[STOP] Stopping Async Trading Bot...")

        self.is_running = False

        try:
            # Cancel the GTT status check task if it exists
            if self.gtt_status_check_task and not self.gtt_status_check_task.done():
                self.gtt_status_check_task.cancel()
                try:
                    await asyncio.wait_for(self.gtt_status_check_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                logger.info("GTT status check task cancelled")

            # Stop components in reverse order
            # Stop position manager if enabled
            if self.event_handlers and self.event_handlers.position_manager:
                try:
                    await self.event_handlers.position_manager.stop()
                    logger.info("Position manager stopped")
                except Exception as e:
                    logger.warning(f"Error stopping position manager: {e}")
            
            if self.ticker_handler:
                await self.ticker_handler.stop_ticker()

            if self.api_server:
                await self.api_server.stop()

            if self.event_dispatcher:
                await self.event_dispatcher.stop()

            logger.info("[OK] Async Trading Bot stopped")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Ignore "Event loop is closed" errors during shutdown
                logger.info("Event loop already closed during shutdown")
            else:
                logger.error(f"Error during shutdown: {e}")

    async def _gtt_status_check_loop(self):
        """Task to periodically check GTT order status"""
        logger.info("Starting GTT status check task...")
        
        # Use a shorter interval for more responsive GTT status updates
        check_interval = 2  # seconds
        last_log_time = {}  # Track last log time per symbol count to reduce verbosity
        
        while self.is_running:
            try:
                if self.strategy_executor and self.state_manager:
                    # Only check GTT order status if there are active trades
                    active_trades = self.state_manager.get_active_trades()
                    if active_trades:
                        trade_count = len(active_trades)
                        # Only log every 30 seconds to reduce verbosity
                        current_time = time.time()
                        last_time = last_log_time.get(trade_count, 0)
                        
                        if current_time - last_time >= 30:
                            logger.debug(f"Checking GTT order status for {trade_count} active trades...")
                            last_log_time[trade_count] = current_time
                        
                        # Run the check in a thread to avoid blocking the event loop
                        await asyncio.to_thread(self.strategy_executor.check_gtt_order_status, verbose=False)
                        
                        # After checking, verify if any trades were removed
                        updated_active_trades = self.state_manager.get_active_trades()
                        if len(updated_active_trades) < len(active_trades):
                            removed_symbols = set(active_trades.keys()) - set(updated_active_trades.keys())
                            logger.info(f"Trades removed during GTT check: {removed_symbols}")
                    else:
                        # Skip GTT check when no active trades
                        logger.debug("No active trades - skipping GTT order status check")
            except Exception as e:
                logger.error(f"Error in GTT status check task: {e}")
            
            # Use shorter interval for more responsive GTT status updates
            await asyncio.sleep(check_interval)
        logger.info("GTT status check task stopped")
    

    async def _main_event_loop(self):
        """Main event loop - replaces the polling loop"""
        logger.info("Entering main event loop...")

        while self.is_running:
            try:
                # Check for unmanaged trades every 5 seconds
                if self.state_manager:
                    try:
                        unmanaged_trades = self.state_manager.get_unmanaged_trades()
                        if unmanaged_trades:
                            logger.info(f"Found {len(unmanaged_trades)} unmanaged trades. Setting up exit orders.")
                            
                            for symbol in unmanaged_trades:
                                if self.strategy_executor:
                                    try:
                                        await asyncio.to_thread(
                                            self.strategy_executor.place_exit_orders,
                                            symbol
                                        )
                                    except RuntimeError as re:
                                        logger.error(f"RuntimeError placing exit orders for {symbol}: {re}")
                                        # Fall back to direct method call
                                        self.strategy_executor.place_exit_orders(symbol)
                                    except Exception as e:
                                        logger.error(f"Error placing exit orders for {symbol}: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error checking unmanaged trades: {e}", exc_info=True)
                
                # Check WebSocket connection every 5 seconds
                if self.ticker_handler:
                    try:
                        if not self.ticker_handler.is_connected() and self.ticker_handler.is_running_state():
                            logger.warning("WebSocket connection lost. Attempting to reconnect...")
                            await self.ticker_handler.check_and_reconnect()
                    except Exception as e:
                        logger.error(f"Error checking WebSocket connection: {e}", exc_info=True)
                
                # Check for active trades with missing GTT orders
                if self.state_manager and self.strategy_executor:
                    try:
                        active_trades = self.state_manager.get_active_trades()
                        for symbol, trade in active_trades.items():
                            # If trade has entry price but no GTT ID and exit orders not placed
                            if (trade.get('entry_price') and 
                                not trade.get('gtt_id') and 
                                not trade.get('exit_orders_placed', False)):
                                logger.warning(f"Found active trade {symbol} without exit orders. Attempting to place exit orders.")
                                try:
                                    await asyncio.to_thread(
                                        self.strategy_executor.place_exit_orders,
                                        symbol
                                    )
                                except Exception as e:
                                    logger.error(f"Error placing missing exit orders for {symbol}: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error checking for trades with missing GTT orders: {e}", exc_info=True)
                
                # Check if market has closed and exit all positions if needed
                if self.state_manager and self.strategy_executor:
                    try:
                        from trading_bot_utils import is_market_open_time
                        from datetime import time as dt_time
                        
                        # Check if we're past market close time
                        now = datetime.now().time()
                        start_time = dt_time(
                            self.config['TRADING_HOURS']['START_HOUR'],
                            self.config['TRADING_HOURS']['START_MINUTE']
                        )
                        end_time = dt_time(
                            self.config['TRADING_HOURS']['END_HOUR'],
                            self.config['TRADING_HOURS']['END_MINUTE']
                        )
                        market_close_config = self.config.get('MARKET_CLOSE', {})
                        market_close_time = dt_time(
                            market_close_config.get('HOUR', 15),
                            market_close_config.get('MINUTE', 30)
                        )
                        
                        # Reset market close exit flag if we're before market open (new trading day)
                        if now < start_time:
                            if hasattr(self, '_market_close_exit_attempted'):
                                self._market_close_exit_attempted = False
                            if hasattr(self, '_market_shutdown_triggered'):
                                self._market_shutdown_triggered = False
                        
                        # If market has closed (past END_TIME) and we have active positions, force exit them
                        if now >= end_time:
                            active_trades = self.state_manager.get_active_trades()
                            if active_trades:
                                # Check if we've already logged the market close exit attempt
                                if not hasattr(self, '_market_close_exit_attempted'):
                                    self._market_close_exit_attempted = False
                                
                                if not self._market_close_exit_attempted:
                                    logger.warning(f"[WARN] Market has closed (current time: {now.strftime('%H:%M:%S')}, end time: {end_time.strftime('%H:%M:%S')})")
                                    logger.warning(f"[WARN] Found {len(active_trades)} active position(s): {list(active_trades.keys())}")
                                    logger.warning("[WARN] Initiating force exit of all positions at market close...")
                                    
                                    # Force exit all positions
                                    if self.event_handlers:
                                        try:
                                            await self.event_handlers._force_exit_all_positions_async()
                                            self._market_close_exit_attempted = True
                                            logger.info("[OK] Market close exit completed")
                                        except Exception as e:
                                            logger.error(f"Error during market close exit: {e}", exc_info=True)
                                    else:
                                        # Fallback: use strategy executor directly
                                        try:
                                            await asyncio.to_thread(
                                                self.strategy_executor.force_exit_all_positions
                                            )
                                            self._market_close_exit_attempted = True
                                            logger.info("[OK] Market close exit completed (via strategy executor)")
                                        except Exception as e:
                                            logger.error(f"Error during market close exit: {e}", exc_info=True)

                        # After market close, stop the bot entirely once shutdown time is reached
                        if now >= market_close_time:
                            if not hasattr(self, '_market_shutdown_triggered'):
                                self._market_shutdown_triggered = False
                            
                            if not self._market_shutdown_triggered:
                                self._market_shutdown_triggered = True
                                logger.info(f"[TIME] Market close time reached (current time: {now.strftime('%H:%M:%S')}, shutdown time: {market_close_time.strftime('%H:%M:%S')})")
                                logger.info("[TIME] Stopping bot automatically for end-of-day shutdown")
                                
                                if self.is_running:
                                    await self.stop()
                                    logger.info("[OK] Async Trading Bot stopped for market close. Restart required for next session.")
                                
                                return
                    except Exception as e:
                        logger.error(f"Error checking market close: {e}", exc_info=True)
                
                # Dispatch a heartbeat event every 5 seconds to keep the system alive
                try:
                    # Directly dispatch the event - our improved event_system can handle non-asyncio threads
                    self.event_dispatcher.dispatch_event(
                        Event(
                            EventType.SYSTEM_STARTUP,  # Using SYSTEM_STARTUP as a heartbeat
                            {
                                'message': 'Main loop heartbeat',
                                'timestamp': datetime.now().timestamp()
                            },
                            source='async_main_workflow'
                        )
                    )
                    # Log at DEBUG level to reduce noise
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Main loop heartbeat dispatched")
                except Exception as e:
                    logger.error(f"Failed to dispatch heartbeat: {e}", exc_info=True)
                
                await asyncio.sleep(5)  # Reduced interval from 30 seconds to 5 seconds for more responsive system
                
            except asyncio.CancelledError:
                logger.info("Main event loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main event loop: {e}", exc_info=True)
                await self._handle_error(e)
                # Continue running the loop despite errors
                await asyncio.sleep(5)  # Wait before retrying to avoid tight error loops

    async def _wait_for_market_open(self):
        """Wait for market to open with countdown display"""
        from datetime import datetime, time
        
        market_open_time = time(9, 15)  # Market opens at 9:15 AM
        
        while True:
            now = datetime.now().time()
            
            if now >= market_open_time:
                logger.info("[CLOCK] Market is now open! Proceeding to get opening price...")
                break
            
            # Calculate time remaining
            time_remaining = datetime.combine(datetime.today(), market_open_time) - datetime.now()
            # Handle case where time_remaining might be negative (shouldn't happen, but safety check)
            if time_remaining.total_seconds() <= 0:
                logger.info("[CLOCK] Market is now open! Proceeding to get opening price...")
                break
            
            hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(f"[TIME] Waiting for market to open... Time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Wait for 30 seconds before checking again
            await asyncio.sleep(30)

    async def _initialize_strikes_from_historical(self):
        """Initialize PE & CE strikes using historical API"""
        try:
            # Mid-day boot: use latest completed candle's calculated price so we don't init at 9:15 and slab-change immediately
            opening_price = None
            if is_market_open_time():
                opening_price = get_nifty_latest_calculated_price_historical(self.kite)
            if opening_price is None:
                # Fallback: opening price (9:15) with retries - for boot at market open or when minute data unavailable
                opening_price = get_nifty_opening_price_historical(self.kite, max_retries=3, retry_delay=5)
            
            if opening_price is None:
                logger.error("[X] Could not get NIFTY opening price from historical data")
                logger.error("[X] This may happen if:")
                logger.error("   1. Market is not open yet")
                logger.error("   2. First candle hasn't completed (wait until 9:16 AM)")
                logger.error("   3. API is unavailable")
                logger.error("[X] Bot will wait for first NIFTY candle from WebSocket to initialize strikes")
                # Don't raise exception - let the bot continue and use first NIFTY tick/candle
                # The dynamic ATM manager will handle initialization when first NIFTY candle arrives
                return
            
            # Process the opening price to derive strikes
            await self._process_nifty_opening_price(opening_price)
            
            # Initialize entry condition manager now that strikes are derived
            await self._initialize_entry_condition_manager()
            
            # Wire entry_condition_manager to event handlers
            if self.entry_condition_manager:
                self.event_handlers.entry_condition_manager = self.entry_condition_manager
                # CRITICAL: Wire entry_condition_manager to strategy_executor so it can clear SKIP_FIRST flags
                if self.strategy_executor:
                    self.strategy_executor.entry_condition_manager = self.entry_condition_manager
            
        except Exception as e:
            logger.error(f"Error initializing strikes from historical data: {e}")
            logger.warning("[WARN] Will continue initialization and use first NIFTY candle/tick for strike calculation")
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - allow bot to continue and use first tick/candle

    async def _process_nifty_opening_price(self, opening_price):
        """Process NIFTY opening price and derive PE & CE strikes"""
        try:
            logger.info(f"[CHART] Processing NIFTY opening price: {opening_price}")
            
            # Store the opening price
            self.nifty_opening_price = opening_price
            
            # Calculate strikes based on STRIKE_TYPE configuration (ATM or OTM)
            ce_strike, pe_strike = calculate_strikes(opening_price)
            
            # Get weekly expiry date and monthly flag
            expiry_date, is_monthly = get_weekly_expiry_date()
            
            # Generate option symbols and update subscribe_tokens.json
            updated_symbols = generate_option_tokens_and_update_file(
                self.kite, 
                ce_strike, 
                pe_strike, 
                expiry_date,
                is_monthly,
                self.config['SUBSCRIBE_TOKENS_FILE_PATH']
            )
            
            if updated_symbols is None:
                raise Exception("Failed to generate option symbols and tokens")
            
            # Update trade_symbols with new data
            self.trade_symbols.update(updated_symbols)
            
            # Mark strikes as derived
            self.strikes_derived = True
            
            logger.info("[OK] Successfully derived and updated PE & CE strikes")
            logger.info(f"CE: {updated_symbols['ce_symbol']} (Strike: {ce_strike})")
            logger.info(f"PE: {updated_symbols['pe_symbol']} (Strike: {pe_strike})")
            
            # Check if SKIP_FIRST is enabled (needs NIFTY for sentiment calculation)
            skip_first_enabled = False
            if self.entry_condition_manager:
                skip_first_enabled = self.entry_condition_manager.skip_first
            elif self.config.get('TRADE_SETTINGS', {}).get('SKIP_FIRST', False):
                skip_first_enabled = True
            
            # Only unsubscribe from NIFTY if automated sentiment AND SKIP_FIRST are NOT enabled
            # If either is enabled, we need NIFTY
            if self.ticker_handler and not self.use_automated_sentiment and not skip_first_enabled:
                if hasattr(self.ticker_handler, 'unsubscribe_nifty_token'):
                    await self.ticker_handler.unsubscribe_nifty_token()
                    logger.info("[BLOCK] Unsubscribed from NIFTY 50 token after processing opening price (automated sentiment and SKIP_FIRST disabled)")
                else:
                    logger.warning("[WARN] unsubscribe_nifty_token method not available - NIFTY will remain subscribed")
            elif self.use_automated_sentiment or skip_first_enabled:
                reasons = []
                if self.use_automated_sentiment:
                    reasons.append("automated sentiment")
                if skip_first_enabled:
                    reasons.append("SKIP_FIRST")
                logger.info(f"[OK] Keeping NIFTY 50 subscribed for {', '.join(reasons)} processing")
            
        except Exception as e:
            logger.error(f"[X] Error processing NIFTY opening price: {e}")
            logger.error(f"Opening price was: {opening_price}")
            logger.error(f"Calculated strikes - CE: {ce_strike if 'ce_strike' in locals() else 'Not calculated'}, PE: {pe_strike if 'pe_strike' in locals() else 'Not calculated'}")
            logger.error(f"Expiry date: {expiry_date if 'expiry_date' in locals() else 'Not calculated'}, Is Monthly: {is_monthly if 'is_monthly' in locals() else 'Not calculated'}")
            raise Exception(f"Failed to generate option symbols and tokens")


    async def _handle_error(self, error: Exception):
        """Handle errors in the trading bot"""
        logger.error(f"Trading bot error: {error}")

        # Dispatch error event
        self.event_dispatcher.dispatch_event(
            Event(
                EventType.ERROR_OCCURRED,
                {
                    'message': f"trading_bot_error: {str(error)}",
                    'error_type': type(error).__name__
                },
                source='trading_bot'
            )
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the trading bot"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'websocket_connected': self.ticker_handler.is_connected() if self.ticker_handler else False,
            'api_server_running': self.api_server.is_running() if self.api_server else False,
            'event_queue_size': self.event_dispatcher.get_queue_size() if self.event_dispatcher else 0,
            'active_trades': len(self.state_manager.get_active_trades()) if self.state_manager else 0
        }


async def main():
    """Main entry point for the async trading bot"""
    # Create the bot instance
    bot = AsyncTradingBot()
    
    # Setup platform-independent signal handling
    try:
        import platform
        if platform.system() != 'Windows':
            # Unix-like systems can use loop.add_signal_handler
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(shutdown(sig, bot, loop))
                )
        else:
            # Windows - we'll rely on KeyboardInterrupt exception
            logger.info("Running on Windows - using KeyboardInterrupt for signal handling")
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")
    
    try:
        # Start the bot
        await bot.start()

        # Keep running until interrupted
        while bot.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting gracefully")
        await shutdown(signal.SIGINT, bot, asyncio.get_running_loop())
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
    finally:
        # If we get here without going through the signal handler,
        # ensure the bot is properly stopped
        if bot.is_running:
            await bot.stop()

async def shutdown(sig, bot, loop):
    """Gracefully shutdown the bot when a signal is received"""
    logger.info(f"Received exit signal {sig.name}...")
    
    # Set a flag to indicate shutdown is in progress
    # This helps prevent new tasks from being created during shutdown
    shutdown_in_progress = True
    
    # Stop the bot first - this should stop most active components
    if bot.is_running:
        try:
            logger.info("Stopping bot components...")
            await asyncio.wait_for(bot.stop(), timeout=3.0)
            logger.info("Bot components stopped successfully")
        except asyncio.TimeoutError:
            logger.warning("Timeout while stopping bot components, continuing shutdown anyway")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.info("Event loop already closed during bot shutdown")
            else:
                logger.error(f"Error stopping bot: {e}")
    
    # Cancel all running tasks except our shutdown task
    try:
        # Get all tasks before cancellation
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        task_count = len(tasks)
        
        if tasks:
            logger.info(f"Cancelling {task_count} outstanding tasks...")
            
            # First attempt: Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to be cancelled with a short timeout
            try:
                # Use a shorter timeout for the first attempt
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
                logger.info("All tasks cancelled successfully")
            except asyncio.TimeoutError:
                # Some tasks didn't cancel in time, try a more aggressive approach
                logger.warning("Some tasks did not cancel in time, trying more aggressive cancellation")
                
                # Get remaining tasks
                remaining_tasks = [t for t in tasks if not t.done()]
                logger.info(f"{len(remaining_tasks)} tasks still running after first cancellation attempt")
                
                # Try to cancel them again with a shorter timeout
                for task in remaining_tasks:
                    task.cancel()
                
                try:
                    # Use an even shorter timeout for the second attempt
                    await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=0.5)
                    logger.info("Remaining tasks cancelled successfully")
                except asyncio.TimeoutError:
                    # If tasks still don't cancel, log them but continue shutdown
                    still_running = [t for t in remaining_tasks if not t.done()]
                    if still_running:
                        logger.warning(f"{len(still_running)} tasks could not be cancelled. Ignoring and continuing shutdown.")
                        for i, task in enumerate(still_running):
                            logger.warning(f"Uncancellable task {i+1}: {task}")
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.info("Event loop already closed during task cancellation")
        else:
            logger.error(f"Error cancelling tasks: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during task cancellation: {e}")
    
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    # Run the async trading bot with improved signal handling
    try:
        # Set a custom exception handler for the event loop
        def custom_exception_handler(loop, context):
            # Don't log KeyboardInterrupt during shutdown
            exception = context.get('exception')
            if isinstance(exception, KeyboardInterrupt):
                return
            
            # Don't log "Task was destroyed but it is pending" errors
            message = context.get('message', '')
            if 'Task was destroyed but it is pending' in message:
                return
                
            # Use default handler for other exceptions
            loop.default_exception_handler(context)
        
        # Create a new event loop with custom settings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(custom_exception_handler)
        
        # Run the main function with the custom loop
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in main loop, initiating shutdown...")
            # Create a bot instance just for shutdown if needed
            bot = AsyncTradingBot()
            loop.run_until_complete(shutdown(signal.SIGINT, bot, loop))
    finally:
        # Clean up the event loop
        try:
            logger.info("Cleaning up event loop...")
            
            # Cancel any remaining tasks
            remaining_tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if remaining_tasks:
                logger.info(f"Cancelling {len(remaining_tasks)} remaining tasks during final cleanup...")
                for task in remaining_tasks:
                    task.cancel()
                
                # Wait briefly for tasks to cancel
                try:
                    loop.run_until_complete(asyncio.wait_for(
                        asyncio.gather(*remaining_tasks, return_exceptions=True),
                        timeout=0.5
                    ))
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("Some tasks could not be cancelled during final cleanup")
            
            # Shutdown async generators with a timeout
            if not loop.is_closed():
                logger.info("Shutting down async generators...")
                try:
                    # Use a direct approach with a timeout
                    shutdown_agen_task = loop.create_task(loop.shutdown_asyncgens())
                    loop.run_until_complete(asyncio.wait_for(shutdown_agen_task, timeout=1.0))
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("Async generator shutdown timed out")
                except Exception as e:
                    logger.warning(f"Error during async generator shutdown: {e}")
                
                # Close the event loop
                logger.info("Closing event loop...")
                loop.close()
                logger.info("Event loop closed")
        except Exception as e:
            # Ignore all errors during cleanup
            logger.warning(f"Ignored error during event loop cleanup: {e}")
        
        # Force exit if we're still hanging
        import os, sys
        logger.info("Shutdown complete, exiting process")
        # Use os._exit as a last resort if we're still hanging
        os._exit(0)
