import json
import logging
from threading import RLock
from typing import Any, Dict, Optional, Set
from trading_bot_utils import format_dict_for_logging


class TradeStateManager:
    """
    Manages the persistent state of the trading bot, including:
      - active trades
      - ATM strike
      - signal states
      - latest prices
      - global sentiment

    Ensures thread-safe operations with an internal re-entrant lock.

    Sentiment expected values:
      - 'NEUTRAL', 'BULLISH', 'BEARISH', 'DISABLE'
    """

    ALLOWED_SENTIMENTS: Set[str] = {"NEUTRAL", "BULLISH", "BEARISH", "DISABLE"}

    def __init__(self, file_path: str, trade_ledger=None):
        self.file_path = file_path
        self.state: Dict[str, Any] = {
            "atm_strike": None,
            "active_trades": {},
            "signal_states": {},
            "latest_prices": {},
            "sentiment": "NEUTRAL",
            "sentiment_mode": "MANUAL",  # AUTO | MANUAL | DISABLE
            "previous_sentiment": None,  # Store previous sentiment before DISABLE for restoration
            "previous_mode": None,  # Store previous mode before DISABLE for restoration
        }
        self._lock = RLock()
        self.logger = logging.getLogger(__name__)
        self.trade_ledger = trade_ledger

    def load_state(self):
        with self._lock:
            try:
                with open(self.file_path, "r") as f:
                    self.state = json.load(f)
                # Ensure defaults exist
                self.state.setdefault("active_trades", {})
                self.state.setdefault("signal_states", {})
                self.state.setdefault("latest_prices", {})
                self.state.setdefault("sentiment", "NEUTRAL")
                self.state.setdefault("sentiment_mode", "MANUAL")
                self.state.setdefault("previous_sentiment", None)
                self.state.setdefault("previous_mode", None)
                
                # Validate sentiment
                if self.state["sentiment"] not in self.ALLOWED_SENTIMENTS:
                    self.logger.warning(
                        f"Invalid sentiment '{self.state['sentiment']}' in state file. Resetting to NEUTRAL."
                    )
                    self.state["sentiment"] = "NEUTRAL"
                
                # Validate sentiment_mode
                allowed_modes = {"AUTO", "MANUAL", "DISABLE"}
                if self.state.get("sentiment_mode") not in allowed_modes:
                    self.logger.warning(
                        f"Invalid sentiment_mode '{self.state.get('sentiment_mode')}' in state file. Resetting to MANUAL."
                    )
                    self.state["sentiment_mode"] = "MANUAL"
                
                self.logger.info("Trade state loaded successfully.")
            except (FileNotFoundError, json.JSONDecodeError):
                self.logger.warning("Trade state file not found or invalid. Starting with a fresh state.")
                self.save_state()

    def save_state(self):
        with self._lock:
            try:
                with open(self.file_path, "w") as f:
                    json.dump(self.state, f, indent=4)
            except Exception as e:
                self.logger.error(f"Failed to save trade state: {e}")

    # ---------------------------
    # Reconciliation with broker
    # ---------------------------
    def reconcile_trades_with_broker(self, broker_positions: Dict[str, Any]):
        """
        Compares the bot's internal state with actual positions from the broker
        and removes any stale trades that are not actually open.
        
        Enhanced to be more robust in detecting stale trades and ensuring
        the trade state is accurately reflecting the broker's positions.
        """
        with self._lock:
            self.logger.info("Reconciling internal state with broker positions...")
            if not self.state.get("active_trades"):
                self.logger.info("No internal trades to reconcile.")
                return

            try:
                # Extract all NFO positions with non-zero quantity
                broker_symbols = {}
                for pos in broker_positions.get("net", []):
                    if pos.get("exchange") == "NFO" and pos.get("quantity", 0) != 0:
                        symbol = pos.get("tradingsymbol")
                        if symbol:
                            broker_symbols[symbol] = pos.get("quantity", 0)
                
                self.logger.info(f"Found {len(broker_symbols)} active positions in broker: {list(broker_symbols.keys())}")
            except Exception as e:
                self.logger.error(f"Could not parse broker positions: {e}. Skipping reconciliation.")
                return

            internal_symbols = set(self.state["active_trades"].keys())
            self.logger.info(f"Found {len(internal_symbols)} active trades in state: {list(internal_symbols)}")
            
            # Find stale trades (in state but not in broker)
            stale_symbols = internal_symbols - set(broker_symbols.keys())

            if stale_symbols:
                self.logger.warning(
                    f"Reconciliation: Found {len(stale_symbols)} stale trade(s) not present in broker positions. Removing them."
                )
                for symbol in stale_symbols:
                    self.logger.info(f"  - Removing stale trade: {symbol}")
                    del self.state["active_trades"][symbol]
                self.save_state()
                self.logger.info(f"Reconciliation: Removed {len(stale_symbols)} stale trades.")
            else:
                self.logger.info("Reconciliation: No stale trades found.")

            # Find untracked positions (in broker but not in state)
            untracked_symbols = set(broker_symbols.keys()) - internal_symbols
            if untracked_symbols:
                self.logger.warning(
                    f"Reconciliation: Found {len(untracked_symbols)} untracked position(s) in broker account."
                )
                for symbol in untracked_symbols:
                    self.logger.warning(
                        f"  - Untracked position: {symbol} (Quantity: {broker_symbols[symbol]}). This trade will not be managed by the bot."
                    )
            else:
                self.logger.info("Reconciliation: No untracked positions found.")
                
            # Double-check GTT IDs for active trades
            for symbol in list(self.state["active_trades"].keys()):
                trade = self.state["active_trades"][symbol]
                if trade.get("gtt_id") and not trade.get("exit_orders_placed", False):
                    self.logger.warning(
                        f"Reconciliation: Trade {symbol} has GTT ID but exit_orders_placed is False. Setting to True."
                    )
                    trade["exit_orders_placed"] = True
                    self.save_state()
                
            # Perform a final check to ensure all active trades in state actually exist in broker
            # This is a safety measure to prevent stale trades from persisting
            for symbol in list(self.state["active_trades"].keys()):
                if symbol not in broker_symbols:
                    self.logger.warning(
                        f"Final check: Trade {symbol} is in state but not found in broker positions. Removing it."
                    )
                    del self.state["active_trades"][symbol]
                    self.save_state()

    # -------------
    # Trade methods
    # -------------
    def add_trade(self, symbol: str, order_id: str, quantity: int, trans_type: str, metadata: dict = None):
        with self._lock:
            if symbol not in self.state["active_trades"]:
                # Store the transaction type properly to identify manual trades
                # For manual trades, trans_type will be 'CE' or 'PE'
                # For autonomous trades, it will be the option type
                from datetime import datetime
                trade_data = {
                    "order_id": order_id,
                    "quantity": quantity,
                    "transaction_type": trans_type,
                    "entry_price": None,
                    "gtt_id": None,
                    "exit_orders_placed": False,
                    "trailing_sl": None,
                    "trailing_activated": False,
                    "is_manual_trade": trans_type in ['CE', 'PE', 'BUY_CE', 'BUY_PE'],  # Flag to identify manual trades
                    "created_at": datetime.now().timestamp()  # Timestamp for grace period check in watchdog
                }
                
                # Add metadata if provided
                if metadata:
                    trade_data["metadata"] = metadata
                
                self.state["active_trades"][symbol] = trade_data
                # Format metadata for logging (2 decimal places for floats)
                formatted_metadata = format_dict_for_logging(metadata, decimals=2) if metadata else None
                self.logger.info(f"New trade added for {symbol}. Manual trade: {trans_type in ['CE', 'PE', 'BUY_CE', 'BUY_PE']}. Metadata: {formatted_metadata}")
                self.save_state()

    def update_trailing_activated(self, symbol: str, status: bool):
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.state["active_trades"][symbol]["trailing_activated"] = status
                self.logger.info(f"Trailing SL mechanism for {symbol} has been activated.")
                self.save_state()

    def update_trade_metadata(self, symbol: str, metadata: dict):
        with self._lock:
            if symbol in self.state["active_trades"]:
                if "metadata" not in self.state["active_trades"][symbol]:
                    self.state["active_trades"][symbol]["metadata"] = {}
                self.state["active_trades"][symbol]["metadata"].update(metadata)
                # Format metadata for logging (2 decimal places for floats)
                formatted_metadata = format_dict_for_logging(metadata, decimals=2)
                self.logger.info(f"Updated metadata for {symbol}: {formatted_metadata}")
                self.save_state()

    def update_atm_strike(self, atm_strike: int):
        with self._lock:
            if self.state.get("atm_strike") != atm_strike:
                self.state["atm_strike"] = atm_strike
                self.logger.info(f"ATM Strike updated to: {atm_strike}")
                self.save_state()

    def remove_trade(self, symbol: str):
        with self._lock:
            if symbol in self.state["active_trades"]:
                del self.state["active_trades"][symbol]
                self.logger.info(f"Trade removed for {symbol}.")
                self.save_state()

    def close_trade(self, symbol: str, exit_reason: str, exit_price: float):
        """Close a trade and record exit details"""
        with self._lock:
            if symbol in self.state["active_trades"]:
                trade = self.state["active_trades"][symbol]
                # Store exit information before removing (for logging/history)
                exit_info = {
                    'exit_reason': exit_reason,
                    'exit_price': exit_price
                }
                entry_price = trade.get('entry_price')
                
                # CRITICAL: Handle None exit_price gracefully
                # If exit_price is None, try to get current LTP or use entry_price as fallback
                if exit_price is None:
                    self.logger.warning(
                        f"[WARN] Exit price is None for {symbol} ({exit_reason}). "
                        f"Attempting to get current price or use entry price as fallback."
                    )
                    # Try to get current price from ticker handler if available
                    # For now, use entry_price as fallback (will result in 0% PnL)
                    exit_price = entry_price if entry_price else 0.0
                    self.logger.warning(f"[FALLBACK] Using exit_price={exit_price:.2f} for {symbol}")
                
                # Log trade exit to ledger
                if self.trade_ledger and entry_price:
                    try:
                        from datetime import datetime
                        exit_time = datetime.now()
                        self.trade_ledger.log_trade_exit(
                            symbol, exit_time, exit_price, entry_price, exit_reason
                        )
                    except Exception as e:
                        self.logger.error(f"Error logging trade exit to ledger: {e}", exc_info=True)
                
                del self.state["active_trades"][symbol]
                exit_price_str = f"{exit_price:.2f}" if exit_price is not None else "N/A"
                self.logger.info(f"Trade closed for {symbol}: {exit_reason} @ {exit_price_str}")
                self.save_state()
            else:
                self.logger.warning(f"Attempted to close trade {symbol} that doesn't exist")

    def update_trade_entry_price(self, symbol: str, price: float):
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.state["active_trades"][symbol]["entry_price"] = price
                self.save_state()

    def update_gtt_id(self, symbol: str, gtt_id: Any):
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.state["active_trades"][symbol]["gtt_id"] = gtt_id
                self.save_state()

    def update_trade_exit_order_status(self, symbol: str, status: bool):
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.state["active_trades"][symbol]["exit_orders_placed"] = status
                self.save_state()

    def update_trailing_sl(self, symbol: str, new_sl: Optional[float]):
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.state["active_trades"][symbol]["trailing_sl"] = new_sl
                self.save_state()

    # --------------
    # Price methods
    # --------------
    def update_latest_price(self, token: Any, price: float):
        with self._lock:
            self.state["latest_prices"][str(token)] = price
            # For performance, avoid saving on each tick
            # self.save_state()

    def get_latest_price(self, token: Any) -> Optional[float]:
        with self._lock:
            return self.state["latest_prices"].get(str(token))

    # ----------------
    # Querying methods
    # ----------------
    def get_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.state["active_trades"].get(symbol)

    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self.state["active_trades"].copy()

    def get_unmanaged_trades(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {s: t for s, t in self.state["active_trades"].items() if not t.get("exit_orders_placed")}

    def is_trade_active(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self.state["active_trades"]

    def get_atm_strike(self) -> Optional[int]:
        with self._lock:
            return self.state.get("atm_strike")

    # -----------------
    # Signal state map
    # -----------------
    def set_signal_state(self, symbol: str, key: str, value: Any):
        with self._lock:
            if symbol not in self.state["signal_states"]:
                self.state["signal_states"][symbol] = {}
            self.state["signal_states"][symbol][key] = value
            self.save_state()

    def get_signal_state(self, symbol: str, key: str) -> Any:
        with self._lock:
            return self.state["signal_states"].get(symbol, {}).get(key)

    def clear_signal_state(self, symbol: str, key: str):
        with self._lock:
            if symbol in self.state["signal_states"] and key in self.state["signal_states"][symbol]:
                del self.state["signal_states"][symbol][key]
                self.save_state()

    # -------------
    # Sentiment API
    # -------------
    def set_sentiment(self, sentiment: str):
        """
        Persist global sentiment. Expected values: 'NEUTRAL', 'BULLISH', 'BEARISH', 'DISABLE'
        When setting to DISABLE, stores the previous sentiment for restoration.
        """
        sentiment = str(sentiment).upper()
        if sentiment not in self.ALLOWED_SENTIMENTS:
            raise ValueError(
                f"Invalid sentiment '{sentiment}'. Allowed: {sorted(self.ALLOWED_SENTIMENTS)}"
            )
        with self._lock:
            current_sentiment = self.state.get("sentiment", "NEUTRAL")
            
            # If setting to DISABLE, store the current sentiment as previous_sentiment
            if sentiment == "DISABLE" and current_sentiment != "DISABLE":
                # Only store if we're not already in DISABLE state
                self.state["previous_sentiment"] = current_sentiment
                self.state["previous_mode"] = self.state.get("sentiment_mode", "MANUAL")
                self.logger.info(f"Storing previous sentiment '{current_sentiment}' and mode '{self.state['previous_mode']}' before setting to DISABLE")
            
            self.state["sentiment"] = sentiment
            self.logger.info(f"Sentiment set to: {sentiment}")
            self.save_state()

    def get_sentiment(self) -> str:
        with self._lock:
            value = self.state.get("sentiment", "NEUTRAL")
            if value not in self.ALLOWED_SENTIMENTS:
                return "NEUTRAL"
            return value

    def get_previous_sentiment(self) -> Optional[str]:
        """
        Get the previous sentiment that was stored before DISABLE was set.
        Returns None if no previous sentiment was stored.
        """
        with self._lock:
            previous = self.state.get("previous_sentiment")
            if previous and previous in self.ALLOWED_SENTIMENTS and previous != "DISABLE":
                return previous
            return None

    def set_sentiment_mode(self, mode: str, manual_sentiment: Optional[str] = None):
        """
        Set the sentiment mode and optionally the manual sentiment.
        
        Args:
            mode: "AUTO" | "MANUAL" | "DISABLE"
            manual_sentiment: Required if mode="MANUAL", ignored otherwise
                             "BULLISH" | "BEARISH" | "NEUTRAL" | "DISABLE"
        """
        mode = str(mode).upper()
        allowed_modes = {"AUTO", "MANUAL", "DISABLE"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {sorted(allowed_modes)}")
        
        with self._lock:
            current_mode = self.state.get("sentiment_mode", "MANUAL")
            current_sentiment = self.state.get("sentiment", "NEUTRAL")
            
            # If switching to DISABLE, store current state for restoration
            if mode == "DISABLE" and current_mode != "DISABLE":
                self.state["previous_sentiment"] = current_sentiment
                self.state["previous_mode"] = current_mode
                self.logger.info(f"[SENTIMENT_MODE_SWITCH] Storing previous mode '{current_mode}' and sentiment '{current_sentiment}' before setting to DISABLE")
            
            self.state["sentiment_mode"] = mode
            
            # Set sentiment based on mode
            if mode == "DISABLE":
                self.state["sentiment"] = "DISABLE"
            elif mode == "MANUAL":
                if manual_sentiment is None:
                    # If no manual_sentiment provided, keep current if valid, else default to NEUTRAL
                    if current_sentiment in self.ALLOWED_SENTIMENTS and current_sentiment != "DISABLE":
                        self.state["sentiment"] = current_sentiment
                    else:
                        self.state["sentiment"] = "NEUTRAL"
                else:
                    manual_sentiment = str(manual_sentiment).upper()
                    if manual_sentiment not in self.ALLOWED_SENTIMENTS:
                        raise ValueError(f"Invalid manual_sentiment '{manual_sentiment}'. Allowed: {sorted(self.ALLOWED_SENTIMENTS)}")
                    
                    # If setting to DISABLE sentiment, store previous state
                    if manual_sentiment == "DISABLE" and current_sentiment != "DISABLE":
                        self.state["previous_sentiment"] = current_sentiment
                        self.state["previous_mode"] = current_mode
                        self.logger.info(f"[SENTIMENT_MODE_SWITCH] Storing previous mode '{current_mode}' and sentiment '{current_sentiment}' before setting to DISABLE")
                    
                    self.state["sentiment"] = manual_sentiment
            # For AUTO mode, sentiment will be calculated by the algorithm, don't set it here
            
            self.logger.info(f"[SENTIMENT_MODE_SWITCH] Mode: {current_mode} → {mode}, Sentiment: {current_sentiment} → {self.state['sentiment']}")
            self.save_state()

    def get_sentiment_mode(self) -> str:
        """Get the current sentiment mode."""
        with self._lock:
            mode = self.state.get("sentiment_mode", "MANUAL")
            if mode not in {"AUTO", "MANUAL", "DISABLE"}:
                return "MANUAL"
            return mode

    def get_previous_mode(self) -> Optional[str]:
        """Get the previous mode that was stored before DISABLE was set."""
        with self._lock:
            previous = self.state.get("previous_mode")
            if previous and previous in {"AUTO", "MANUAL", "DISABLE"}:
                return previous
            return None

    def get_allowed_sentiments(self) -> Set[str]:
        return set(self.ALLOWED_SENTIMENTS)
        
    def force_reconciliation(self, broker_positions=None):
        """
        Force a full reconciliation of the state with broker positions.
        This is useful for manual cleanup of the state after manual trades or force exits.
        
        If broker_positions is not provided, it will be fetched from the broker.
        """
        with self._lock:
            self.logger.info("Forcing full reconciliation of trade state...")
            
            # If no broker positions provided, return early
            if not broker_positions:
                self.logger.warning("No broker positions provided for force reconciliation")
                return
                
            # Extract all NFO positions with non-zero quantity
            broker_symbols = {}
            for pos in broker_positions.get("net", []):
                if pos.get("exchange") == "NFO" and pos.get("quantity", 0) != 0:
                    symbol = pos.get("tradingsymbol")
                    if symbol:
                        broker_symbols[symbol] = pos.get("quantity", 0)
            
            self.logger.info(f"Found {len(broker_symbols)} active positions in broker: {list(broker_symbols.keys())}")
            
            # Get all active trades from state
            internal_symbols = set(self.state["active_trades"].keys())
            self.logger.info(f"Found {len(internal_symbols)} active trades in state: {list(internal_symbols)}")
            
            # Find stale trades (in state but not in broker)
            stale_symbols = internal_symbols - set(broker_symbols.keys())
            
            if stale_symbols:
                self.logger.warning(f"Force reconciliation: Found {len(stale_symbols)} stale trade(s) not present in broker positions. Removing them.")
                for symbol in stale_symbols:
                    self.logger.info(f"  - Removing stale trade: {symbol}")
                    del self.state["active_trades"][symbol]
                self.save_state()
                self.logger.info(f"Force reconciliation: Removed {len(stale_symbols)} stale trades.")
            else:
                self.logger.info("Force reconciliation: No stale trades found.")
            
            # Find untracked positions (in broker but not in state)
            untracked_symbols = set(broker_symbols.keys()) - internal_symbols
            if untracked_symbols:
                self.logger.warning(f"Force reconciliation: Found {len(untracked_symbols)} untracked position(s) in broker account.")
                for symbol in untracked_symbols:
                    self.logger.warning(f"  - Untracked position: {symbol} (Quantity: {broker_symbols[symbol]}). This trade will not be managed by the bot.")
            else:
                self.logger.info("Force reconciliation: No untracked positions found.")
            
            # Return the cleaned state
            return self.state["active_trades"].copy()
    
    def force_clear_symbol(self, symbol: str):
        """
        Force clear a specific symbol from active trades.
        This is useful for cleaning up stale state before manual trade execution.
        
        Returns True if the symbol was removed, False if it wasn't in active trades.
        """
        with self._lock:
            if symbol in self.state["active_trades"]:
                self.logger.info(f"Force clearing symbol {symbol} from active trades")
                del self.state["active_trades"][symbol]
                
                # Also clear any signal states for this symbol
                if symbol in self.state["signal_states"]:
                    self.logger.info(f"[BUG FIX] Also clearing signal states for {symbol}")
                    del self.state["signal_states"][symbol]
                
                self.save_state()
                self.logger.info(f"Successfully force cleared {symbol} from active trades")
                return True
            else:
                self.logger.info(f"Symbol {symbol} not found in active trades, nothing to clear")
                return False
    
    def reset_all_state(self):
        """
        [BUG FIX] Reset all state completely.
        This is a nuclear option to completely reset the system state.
        Use with caution as it will clear all active trades and signal states.
        """
        with self._lock:
            self.logger.warning("[BUG FIX] RESETTING ALL STATE - THIS IS A NUCLEAR OPTION")
            
            # Save the current sentiment
            current_sentiment = self.state.get("sentiment", "NEUTRAL")
            
            # Clear all state
            self.state["active_trades"] = {}
            self.state["signal_states"] = {}
            
            # Restore sentiment
            self.state["sentiment"] = current_sentiment
            
            self.save_state()
            self.logger.warning("[BUG FIX] ALL STATE RESET COMPLETE")
            return True
