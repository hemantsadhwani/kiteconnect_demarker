import sqlite3
import json
import os
import yaml

def create_new_tables(db_path, instrument_tokens):
    """
    Create OHLC, indicators, entry_conditions, and trades tables for each instrument token and drop general unwanted tables (indicators, entry_conditions, trades).

    Args:
    db_path (str): The path to the SQLite database file.
    instrument_tokens (list): List of instrument tokens.
    """
    # Load config to get dynamic Fast MA and Slow MA periods
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    indicators_config = config.get('INDICATORS', {})
    
    # Support both new FAST_MA/SLOW_MA structure and legacy EMA_PERIODS/SMA_LENGTH
    fast_ma_config = indicators_config.get('FAST_MA', {})
    slow_ma_config = indicators_config.get('SLOW_MA', {})
    
    if fast_ma_config and slow_ma_config:
        # New structure
        fast_ma_length = fast_ma_config.get('LENGTH', 4)
        slow_ma_length = slow_ma_config.get('LENGTH', 7)
        fast_ma_type = fast_ma_config.get('MA', 'sma').lower()
        slow_ma_type = slow_ma_config.get('MA', 'sma').lower()
    else:
        # Legacy structure
        ema_periods = indicators_config.get('EMA_PERIODS', 3)
        if isinstance(ema_periods, list) and len(ema_periods) > 0:
            fast_ma_length = ema_periods[0]
        elif isinstance(ema_periods, int):
            fast_ma_length = ema_periods
        else:
            fast_ma_length = 3
        fast_ma_type = 'ema'
        slow_ma_length = indicators_config.get('SMA_LENGTH', 7)
        slow_ma_type = 'sma'
    
    # For database table column names, use the actual period values
    ema_period = fast_ma_length if fast_ma_type == 'ema' else None
    sma_period = slow_ma_length if slow_ma_type == 'sma' else None
    
    # Fallback to defaults if needed
    if ema_period is None:
        ema_period = 3
    if sma_period is None:
        sma_period = 7

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for token in instrument_tokens:
            # Create ohlc table for this token
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS ohlc_{token} (
                    timestamp TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            ''')

            # Drop and recreate indicators table for this token
            cursor.execute(f'DROP TABLE IF EXISTS indicators_{token}')
            cursor.execute(f'''
                CREATE TABLE indicators_{token} (
                    timestamp TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    supertrend REAL,
                    williams_r_9 REAL,
                    williams_r_28 REAL,
                    stockrsi_k REAL,
                    stockrsi_d REAL,
                    ema_{ema_period} REAL,
                    sma_{sma_period} REAL
                )
            ''')

            # Drop and recreate entry_conditions table for this token
            cursor.execute(f'DROP TABLE IF EXISTS entry_conditions_{token}')
            cursor.execute(f'''
                CREATE TABLE entry_conditions_{token} (
                    timestamp TEXT PRIMARY KEY,
                    fast_crossover_detected INTEGER,
                    stoch_fast_reversal INTEGER,
                    slow_crossover_detected INTEGER,
                    stoch_slow_reversal INTEGER,
                    is_bullish INTEGER,
                    swing_low_condition INTEGER,
                    trade_type TEXT,
                    entry_condition TEXT
                )
            ''')

            # Create trades table for this token
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS trades_{token} (
                    timestamp TEXT,
                    event_type TEXT,
                    signal TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    entry_order_id TEXT,
                    exit_reason TEXT,
                    PRIMARY KEY (timestamp, event_type)
                )
            ''')

        # Drop general unwanted tables
        cursor.execute('DROP TABLE IF EXISTS indicators')
        cursor.execute('DROP TABLE IF EXISTS entry_conditions')
        cursor.execute('DROP TABLE IF EXISTS trades')

        conn.commit()
        print(f"Tables created and general unwanted tables dropped successfully for tokens: {instrument_tokens}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

# --- Main Execution ---
if __name__ == "__main__":
    db_file = 'output/ohlc_data.db'

    # Read instrument tokens from subscribed_tickers.json
    subscribed_tokens_file = 'output/subscribed_tickers.json'
    if os.path.exists(subscribed_tokens_file):
        with open(subscribed_tokens_file, 'r') as f:
            subscribed_tokens = json.load(f)
        instrument_tokens = list(subscribed_tokens.keys())
        print(f"Found tokens: {instrument_tokens}")
    else:
        print("Warning: subscribed_tickers.json not found, using empty token list")
        instrument_tokens = []

    create_new_tables(db_file, instrument_tokens)
