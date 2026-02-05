# indicators.py

import pandas as pd
import numpy as np
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Any, Tuple

class RollingDataBuffer:
    """
    Memory-efficient rolling buffer for OHLC data.
    Maintains a fixed-size window of recent candles to avoid full DB reads.
    Updated for StochRSI session continuity: 35 candles minimum
    (StochRSI needs 32 + Williams %R needs 28 + 5 buffer for session gaps).
    """
    def __init__(self, max_size: int = 35):
        self.data = pd.DataFrame()
        self.max_size = max_size
        self.last_timestamp = None
    
    def update(self, new_candle: pd.DataFrame) -> bool:
        """
        Add new candle to buffer and trim if needed.
        Returns True if new data was added, False if duplicate.
        """
        if new_candle.empty:
            return False
        
        # Check if this is actually new data
        latest_timestamp = new_candle.iloc[-1]['timestamp'] if 'timestamp' in new_candle.columns else None
        if latest_timestamp and latest_timestamp == self.last_timestamp:
            return False
        
        # Add new data and trim
        self.data = pd.concat([self.data, new_candle], ignore_index=True).tail(self.max_size)
        self.last_timestamp = latest_timestamp
        return True
    
    def get_data(self) -> pd.DataFrame:
        """Return current buffer data."""
        return self.data.copy()
    
    def has_sufficient_data(self, min_periods: int) -> bool:
        """Check if buffer has enough data for calculations."""
        return len(self.data) >= min_periods

class IndicatorManager: # <-- RENAMED FROM OptimizedIndicators
    """
    High-performance indicator calculator with concurrent processing and incremental updates.
    """
    def __init__(self, config: dict):
        # Adjusted to expect the full config and select 'INDICATORS' itself
        self.config = config.get('INDICATORS', {})
        self.indicator_cache = {}  # Cache for incremental calculations
    
    def _calculate_atr_incremental(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with incremental update capability."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(alpha=1.0/period, adjust=False).mean()
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate SuperTrend indicator using pandas_ta to match backtesting exactly.
        
        This ensures consistency between real-time production code and backtesting.
        pandas_ta SuperTrend matches TradingView's SuperTrend implementation.
        
        Direction convention (pandas_ta):
        - 1 = bullish (price above SuperTrend line)
        - -1 = bearish (price below SuperTrend line)
        """
        period = self.config['SUPERTREND']['ATR_LENGTH']
        multiplier = self.config['SUPERTREND']['FACTOR']
        
        # Use pandas_ta SuperTrend (same as backtesting)
        supertrend_data = ta.supertrend(df['high'], df['low'], df['close'], 
                                       length=period, multiplier=multiplier)
        
        # Extract components - use the actual column names from pandas_ta
        supertrend_col = [col for col in supertrend_data.columns if 'SUPERT_' in col and 'd' not in col][0]
        supertrend_dir_col = [col for col in supertrend_data.columns if 'SUPERTd_' in col][0]
        
        # Return in the same format as before
        return {
            'supertrend': supertrend_data[supertrend_col].round(2),
            'supertrend_dir': supertrend_data[supertrend_dir_col]
        }
    
    def _calculate_wpr(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Williams %R for both fast and slow periods.
        Matches Pine Script formula exactly: 100 * (close - high_max) / (high_max - low_min)
        Same implementation as backtesting for consistency.
        """
        results = {}
        for period in [self.config['WPR_FAST_LENGTH'], self.config['WPR_SLOW_LENGTH']]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            denom = high_max - low_min

            # Pine Script formula: 100 * (close - high_max) / (high_max - low_min)
            # Avoid division by zero: when range is zero (flat period), W%R is undefined -> NaN
            williams_r = np.where(denom > 0, 100 * (df['close'] - high_max) / denom, np.nan)
            results[f'wpr_{period}'] = williams_r.round(2)
        return results
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI using Wilder's smoothing method (exactly like TradingView)."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's RSI calculation (exponential moving average with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stoch_rsi(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate StochRSI K and D values using pandas_ta to match backtesting exactly.
        
        This ensures consistency between real-time production code and backtesting.
        pandas_ta StochRSI matches TradingView's StochRSI implementation.
        
        Same implementation as backtesting/indicators_backtesting.py for consistency.
        """
        cfg = self.config['SToch_RSI']
        smoothK = cfg.get('K', 3)
        smoothD = cfg.get('D', 3)
        lengthRSI = cfg.get('RSI_LENGTH', 14)
        lengthStoch = cfg.get('STOCH_PERIOD', 14)

        # Use pandas_ta StochRSI (same as backtesting)
        stochrsi_data = ta.stochrsi(df['close'], 
                                   length=lengthStoch, 
                                   rsi_length=lengthRSI,
                                   k=smoothK, 
                                   d=smoothD)
        
        # Extract K and D values - use the actual column names from pandas_ta
        k_col = [col for col in stochrsi_data.columns if 'STOCHRSIk' in col][0]
        d_col = [col for col in stochrsi_data.columns if 'STOCHRSId' in col][0]
        
        # Return in the same format as before
        return {
            'stoch_k': stochrsi_data[k_col].round(2),
            'stoch_d': stochrsi_data[d_col].round(2)
        }
    
    def _calculate_fast_slow_ma(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Fast MA and Slow MA based on config.
        Supports new FAST_MA/SLOW_MA structure and legacy EMA_PERIODS/SMA_LENGTH.
        """
        results = {}
        # self.config is already the INDICATORS section from __init__
        indicators_config = self.config
        
        # Try new FAST_MA/SLOW_MA structure first
        fast_ma_config = indicators_config.get('FAST_MA', {})
        slow_ma_config = indicators_config.get('SLOW_MA', {})
        
        if fast_ma_config and slow_ma_config:
            # New structure: Use FAST_MA and SLOW_MA config
            fast_ma_type = fast_ma_config.get('MA', 'sma').lower()
            fast_ma_length = fast_ma_config.get('LENGTH', 4)
            slow_ma_type = slow_ma_config.get('MA', 'sma').lower()
            slow_ma_length = slow_ma_config.get('LENGTH', 7)
        else:
            # Legacy: EMA_PERIODS and SMA_LENGTH
            ema_periods = indicators_config.get('EMA_PERIODS', 3)
            if isinstance(ema_periods, int):
                fast_ma_type = 'ema'
                fast_ma_length = ema_periods
            elif isinstance(ema_periods, list) and len(ema_periods) > 0:
                fast_ma_type = 'ema'
                fast_ma_length = ema_periods[0]
            else:
                fast_ma_type = 'ema'
                fast_ma_length = 3
            
            slow_ma_type = 'sma'
            slow_ma_length = indicators_config.get('SMA_LENGTH', 7)
        
        # Calculate Fast MA
        if fast_ma_type == 'ema':
            results['fast_ma'] = df['close'].ewm(span=fast_ma_length, adjust=False).mean()
        else:  # sma
            results['fast_ma'] = df['close'].rolling(window=fast_ma_length).mean()
        
        # Calculate Slow MA
        if slow_ma_type == 'ema':
            results['slow_ma'] = df['close'].ewm(span=slow_ma_length, adjust=False).mean()
        else:  # sma
            results['slow_ma'] = df['close'].rolling(window=slow_ma_length).mean()
        
        # For backward compatibility, also create legacy column names
        # This ensures code that still references ema{period}/sma{period} continues to work
        if fast_ma_type == 'ema':
            results[f'ema{fast_ma_length}'] = results['fast_ma']
        if slow_ma_type == 'sma':
            results[f'sma{slow_ma_length}'] = results['slow_ma']
        
        return results
    
    def _calculate_swing_low(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate swing low over a defined period."""
        return {
            'swing_low': df['low'].rolling(window=self.config['SWING_LOW_PERIOD']).min()
        }
    
    def calculate_all_concurrent(self, df: pd.DataFrame, token_type: str = None) -> pd.DataFrame:
        """
        Calculate all indicators concurrently for maximum performance.
        Independent indicators are calculated in parallel threads.
        
        Args:
            df: DataFrame with OHLC data
            token_type: Type of token ('CE' for Call, 'PE' for Put, or None)
        """
        if df.empty:
            return df
        
        start_time = time.time()
        result_df = df.copy()
        
        # Define indicator calculation tasks
        indicator_tasks = [
            ('supertrend', self._calculate_supertrend),
            ('wpr', self._calculate_wpr),
            ('stoch_rsi', self._calculate_stoch_rsi),
            ('fast_slow_ma', self._calculate_fast_slow_ma),
            ('swing_low', self._calculate_swing_low)
        ]
        
        # Execute indicators concurrently
        with ThreadPoolExecutor(max_workers=len(indicator_tasks)) as executor:
            # Submit all tasks
            future_to_indicator = {
                executor.submit(task_func, df): indicator_name 
                for indicator_name, task_func in indicator_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    indicator_results = future.result()
                    # Add all columns from this indicator to result DataFrame
                    for col_name, series in indicator_results.items():
                        result_df[col_name] = series
                except Exception as e:
                    print(f"Error calculating {indicator_name}: {e}")
        
        calculation_time = time.time() - start_time
        
        # Print key indicator values for the latest candle with token type info
        if not result_df.empty:
            latest = result_df.iloc[-1]
            token_label = f" [{token_type}]" if token_type else ""
            # Use fast_ma and slow_ma columns (with fallback to legacy names)
            fast_ma_val = latest.get('fast_ma', latest.get('ema3', latest.get('ema4', 'N/A')))
            slow_ma_val = latest.get('slow_ma', latest.get('sma7', 'N/A'))
            fast_ma_str = f"{fast_ma_val:.2f}" if isinstance(fast_ma_val, (int, float)) else str(fast_ma_val)
            slow_ma_str = f"{slow_ma_val:.2f}" if isinstance(slow_ma_val, (int, float)) else str(slow_ma_val)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Concurrent indicator calculation completed in {calculation_time:.4f} seconds")
            logger.debug(f"Latest indicators{token_label}: Fast MA={fast_ma_str}, Slow MA={slow_ma_str}")
        
        return result_df

class PerformanceMonitor:
    """
    Monitor and track performance metrics for the trading loop.
    """
    def __init__(self):
        self.metrics = {
            'db_read_time': [],
            'indicator_calc_time': [],
            'signal_check_time': [],
            'total_loop_time': []
        }
    
    def record_metric(self, metric_name: str, duration: float):
        """Record a performance metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(duration)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average performance metrics."""
        return {
            name: np.mean(times) if times else 0.0 
            for name, times in self.metrics.items()
        }
    
    def print_performance_summary(self):
        """Print performance summary."""
        import logging
        logger = logging.getLogger(__name__)
        
        avg_metrics = self.get_average_metrics()
        logger.info("=== Performance Summary ===")
        for metric, avg_time in avg_metrics.items():
            logger.info(f"{metric}: {avg_time:.4f}s avg")
        logger.info("=" * 30)
