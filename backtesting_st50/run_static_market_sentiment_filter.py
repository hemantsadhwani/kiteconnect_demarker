#!/usr/bin/env python3
"""
Market Sentiment Filter for Static Trading Strategy
Filters trades based on market sentiment (BULLISH, BEARISH, NEUTRAL)
Works with static data (ATM, OTM, ITM strike types)
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _load_config():
    """Load configuration from backtesting_config.yaml"""
    base_dir = Path(__file__).parent
    possible_config_paths = [
        base_dir / 'backtesting_config.yaml',
        base_dir.parent / 'backtesting_config.yaml',
        Path('backtesting_config.yaml'),
    ]
    
    for config_path in possible_config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load price zones
                price_zones = config.get('BACKTESTING_ANALYSIS', {}).get('PRICE_ZONES', {})
                static_atm_zone = price_zones.get('STATIC_ATM', {})
                static_otm_zone = price_zones.get('STATIC_OTM', {})
                
                static_atm_low = static_atm_zone.get('LOW_PRICE', None)
                static_atm_high = static_atm_zone.get('HIGH_PRICE', None)
                static_otm_low = static_otm_zone.get('LOW_PRICE', None)
                static_otm_high = static_otm_zone.get('HIGH_PRICE', None)
                
                # Load time distribution filter
                time_filter_config = config.get('TIME_DISTRIBUTION_FILTER', {})
                time_filter_enabled = time_filter_config.get('ENABLED', False)
                time_zones = time_filter_config.get('TIME_ZONES', {})
                
                # Load market sentiment filter config
                sentiment_filter_config = config.get('MARKET_SENTIMENT_FILTER', {})
                sentiment_filter_enabled = sentiment_filter_config.get('ENABLED', True)  # Default to True for backward compatibility
                
                return {
                    'price_zones': {
                        'STATIC_ATM': (static_atm_low, static_atm_high),
                        'STATIC_OTM': (static_otm_low, static_otm_high),
                    },
                    'time_filter_enabled': time_filter_enabled,
                    'time_zones': time_zones,
                    'sentiment_filter_enabled': sentiment_filter_enabled,
                }
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
    
    return {
        'price_zones': {
            'STATIC_ATM': (None, None),
            'STATIC_OTM': (None, None),
        },
        'time_filter_enabled': False,
        'time_zones': {},
        'sentiment_filter_enabled': True,  # Default to True for backward compatibility
    }

def _load_price_zones_config():
    """Load PRICE_ZONES configuration from backtesting_config.yaml (backward compatibility)"""
    config = _load_config()
    return config['price_zones']

def _is_time_zone_enabled(entry_time_dt, time_filter_enabled, time_zones):
    """Check if entry time falls within an enabled time zone"""
    if not time_filter_enabled:
        return True  # If filter is disabled, allow all times
    
    if pd.isna(entry_time_dt):
        return False
    
    time_obj = entry_time_dt.time()
    
    # Check each time zone
    for zone_str, enabled in time_zones.items():
        if not enabled:
            continue  # Skip disabled zones
        
        try:
            # Parse zone string like "09:15-10:00"
            start_str, end_str = zone_str.split('-')
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
            
            # Check if time falls within this zone
            if start_time <= end_time:
                if start_time <= time_obj <= end_time:
                    return True
            else:
                # Zone spans midnight (e.g., 23:00-01:00)
                if time_obj >= start_time or time_obj <= end_time:
                    return True
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid time zone format '{zone_str}': {e}")
            continue
    
    # If time doesn't fall in any enabled zone, it's filtered out
    return False

def _process_one_set(expiry_week, day_label, strike_types, entry_type='Entry2'):
    """
    Process static market sentiment filter for one entry type.
    
    Filter static trades based on market sentiment:
    - BULLISH: Only CE trades
    - BEARISH: Only PE trades  
    - NEUTRAL: Both PE and CE trades
    """
    entry_type_lower = entry_type.lower()
    logger.info(f"\n=== Processing {entry_type} Static Market Sentiment Filter ===")
    
    # Convert day label to date suffix (e.g., OCT15 -> oct15, OCT29 -> oct29)
    def day_label_to_suffix(day_label):
        """Convert day label (e.g., OCT15) to date suffix (e.g., oct15)"""
        if len(day_label) < 5:
            return None
        month = day_label[:3].lower()  # OCT -> oct
        day = day_label[3:].lower()     # 15 -> 15
        return f"{month}{day}"
    
    date_suffix = day_label_to_suffix(day_label)
    if not date_suffix:
        logger.error(f"Invalid day label format: {day_label}. Expected format: OCT15, OCT29, etc.")
        return
    
    # File paths
    base_dir = Path(f"data/{expiry_week}_STATIC/{day_label}")
    sentiment_file = base_dir / f"nifty_market_sentiment_{date_suffix}.csv"
    
    # Check if market sentiment file exists
    if not sentiment_file.exists():
        logger.error(f"Market sentiment file not found: {sentiment_file}")
        return
    
    # Load market sentiment data
    logger.info("Loading market sentiment data...")
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Initialize summary data for output CSV
    summary_data = []
    
    # Process each strike type
    for strike_type in strike_types:
        logger.info(f"\nProcessing {strike_type} strike type...")
        
        # File paths for this strike type
        ce_trades_file = base_dir / f"{entry_type_lower}_static_{strike_type.lower()}_ce_trades.csv"
        pe_trades_file = base_dir / f"{entry_type_lower}_static_{strike_type.lower()}_pe_trades.csv"
        output_file = base_dir / f"{entry_type_lower}_static_{strike_type.lower()}_mkt_sentiment_trades.csv"
        
        # Check if trade files exist
        if not ce_trades_file.exists():
            logger.warning(f"CE trades file not found: {ce_trades_file}")
            ce_trades = pd.DataFrame()
        else:
            ce_trades = pd.read_csv(ce_trades_file)
            
        if not pe_trades_file.exists():
            logger.warning(f"PE trades file not found: {pe_trades_file}")
            pe_trades = pd.DataFrame()
        else:
            pe_trades = pd.read_csv(pe_trades_file)
        
        if ce_trades.empty and pe_trades.empty:
            logger.warning(f"No trade data found for {strike_type}")
            continue
        
        # Convert entry times to datetime
        # The trade times are just time strings, so we need to add the date
        # Detect year from sentiment_df to handle year transitions (e.g., 2025 -> 2026)
        detected_year = None
        if not sentiment_df.empty and 'date' in sentiment_df.columns:
            try:
                # Get the year from the first date in sentiment_df
                first_date = pd.to_datetime(sentiment_df['date'].iloc[0])
                detected_year = first_date.year
                logger.debug(f"Detected year {detected_year} from sentiment data")
            except Exception as e:
                logger.debug(f"Could not detect year from sentiment data: {e}")
        
        # Fallback to current year or 2025 if detection fails
        if detected_year is None:
            from datetime import datetime
            current_year = datetime.now().year
            # If we're in January and the day_label is also January, use current year
            # Otherwise default to 2025 for backward compatibility
            if day_label[:3].upper() == 'JAN' and datetime.now().month == 1:
                detected_year = current_year
            else:
                detected_year = 2025  # Default fallback
            logger.debug(f"Using fallback year {detected_year} for day_label {day_label}")
        
        # Extract date from day_label using generic function (e.g., OCT15 -> 2025-10-15)
        def day_label_to_date(day_label, year):
            """Convert day label (e.g., OCT15) to date string (e.g., '2025-10-15')"""
            if len(day_label) < 5:
                return None
            month_abbrev = day_label[:3].upper()  # OCT, NOV, etc.
            day_str = day_label[3:]  # 15, 29, etc.
            
            month_map = {
                'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
            }
            
            if month_abbrev not in month_map:
                return None
            
            try:
                day_int = int(day_str)
                if 1 <= day_int <= 31:
                    return f"{year}-{month_map[month_abbrev]}-{day_str:>02s}"
            except ValueError:
                pass
            
            return None
        
        trade_date = day_label_to_date(day_label, detected_year)
        if not trade_date:
            logger.error(f"Could not convert day_label to date: {day_label}")
            return
        
        if not ce_trades.empty:
            # Convert entry_time to datetime for sentiment matching, but keep original format for output
            # Ensure timezone-aware datetime to match sentiment_df
            ce_trades['entry_time_dt'] = pd.to_datetime(trade_date + ' ' + ce_trades['entry_time'].astype(str)).dt.tz_localize('Asia/Kolkata')
            ce_trades['option_type'] = 'CE'
        
        if not pe_trades.empty:
            pe_trades['entry_time_dt'] = pd.to_datetime(trade_date + ' ' + pe_trades['entry_time'].astype(str)).dt.tz_localize('Asia/Kolkata')
            pe_trades['option_type'] = 'PE'
        
        # Drop entry_period column if it exists
        if 'entry_period' in ce_trades.columns:
            ce_trades = ce_trades.drop(columns=['entry_period'])
        if 'entry_period' in pe_trades.columns:
            pe_trades = pe_trades.drop(columns=['entry_period'])
        
        # Combine all trades for this strike type
        all_trades = pd.concat([ce_trades, pe_trades], ignore_index=True)
        
        logger.info(f"Total trades loaded for {strike_type}: {len(all_trades)} (CE: {len(ce_trades)}, PE: {len(pe_trades)})")
        
        if all_trades.empty:
            logger.warning(f"No trades to process for {strike_type}")
            continue
        
        # Filter trades based on market sentiment
        filtered_trades = []
        
        # Load time distribution filter config
        config = _load_config()
        time_filter_enabled = config['time_filter_enabled']
        time_zones = config['time_zones']
        sentiment_filter_enabled = config.get('sentiment_filter_enabled', True)  # Default to True for backward compatibility
        
        # Log sentiment filter status
        if not sentiment_filter_enabled:
            logger.info(f"Market sentiment filter DISABLED - including all trades regardless of sentiment (Entry1 behavior)")
        else:
            logger.debug(f"Market sentiment filter ENABLED - applying sentiment-based filtering")
        
        for _, trade in all_trades.iterrows():
            entry_time = trade['entry_time_dt']  # Use datetime for matching
            
            # Apply time zone filter first
            if not _is_time_zone_enabled(entry_time, time_filter_enabled, time_zones):
                logger.debug(f"Time zone filter: Excluding trade at {entry_time}")
                continue
            
            # Find the market sentiment for this trade's entry time
            matching_sentiment = None
            matching_rows = sentiment_df[sentiment_df['date'] == entry_time]
            if not matching_rows.empty:
                matching_sentiment = matching_rows.iloc[0]['sentiment']
            else:
                # Try to find nearest sentiment within 60 seconds (fallback for timestamp mismatches)
                time_diff = abs((sentiment_df['date'] - entry_time).dt.total_seconds())
                if time_diff.min() <= 60:
                    nearest_idx = time_diff.idxmin()
                    matching_sentiment = sentiment_df.loc[nearest_idx, 'sentiment']
                    logger.debug(f"Using nearest sentiment (within 60s) for trade at {entry_time}")
            
            # Skip sentiment filtering for Entry1 OR if sentiment filter is disabled (include all trades)
            if entry_type == 'Entry1' or not sentiment_filter_enabled:
                # Include all trades regardless of sentiment (even if sentiment is None)
                if matching_sentiment is not None:
                    matching_sentiment = str(matching_sentiment).upper().strip()
                else:
                    matching_sentiment = 'N/A'  # Default if no sentiment found
                
                trade_data = trade.to_dict()
                trade_data['market_sentiment'] = matching_sentiment
                filtered_trades.append(trade_data)
                continue
            
            # For Entry2, Entry3, etc., require sentiment (only if filter is enabled)
            if matching_sentiment is None:
                logger.warning(f"No sentiment found for trade at {entry_time}, skipping")
                continue
            
            # Normalize sentiment to uppercase for comparison
            matching_sentiment = str(matching_sentiment).upper().strip()
            
            # Apply sentiment filtering rules for Entry2, Entry3, etc.
            should_include = False
            
            # Handle DISABLE sentiment - reject all trades (no trades allowed)
            if matching_sentiment == 'DISABLE':
                logger.debug(f"DISABLE sentiment: Rejecting trade at {entry_time} (no trades allowed)")
                continue
            
            # Handle NEUTRAL sentiment - allow both CE and PE trades (all trades allowed)
            elif matching_sentiment == 'NEUTRAL':
                should_include = True
                logger.debug(f"NEUTRAL sentiment: Including {trade['option_type']} trade at {entry_time} (both CE and PE allowed)")
            
            # Handle BULLISH sentiment - only CE trades allowed
            elif matching_sentiment == 'BULLISH':
                if trade['option_type'] == 'CE':
                    should_include = True
                    logger.debug(f"BULLISH sentiment: Including CE trade at {entry_time}")
                else:
                    logger.debug(f"BULLISH sentiment: Excluding PE trade at {entry_time} (only CE allowed)")
                    
            # Handle BEARISH sentiment - only PE trades allowed
            elif matching_sentiment == 'BEARISH':
                if trade['option_type'] == 'PE':
                    should_include = True
                    logger.debug(f"BEARISH sentiment: Including PE trade at {entry_time}")
                else:
                    logger.debug(f"BEARISH sentiment: Excluding CE trade at {entry_time} (only PE allowed)")
            
            # Unknown sentiment value - log warning and skip
            else:
                logger.warning(f"Unknown sentiment value '{matching_sentiment}' for trade at {entry_time}, skipping")
                continue
            
            if should_include:
                # Add sentiment information to the trade
                trade_data = trade.to_dict()
                trade_data['market_sentiment'] = matching_sentiment
                filtered_trades.append(trade_data)
        
        if not filtered_trades:
            logger.warning(f"No trades passed the sentiment filter for {strike_type}!")
            continue
        
        # Create filtered DataFrame
        filtered_df = pd.DataFrame(filtered_trades)
        
        # Apply PRICE_ZONES filter as post-processing (after sentiment filtering)
        price_zones_config = _load_price_zones_config()
        strike_type_key = f'STATIC_{strike_type}'
        price_zone_low, price_zone_high = price_zones_config.get(strike_type_key, (None, None))
        
        if price_zone_low is not None and price_zone_high is not None:
            original_count = len(filtered_df)
            # Filter by entry_price (inclusive range)
            filtered_df = filtered_df[
                (pd.to_numeric(filtered_df['entry_price'], errors='coerce') >= price_zone_low) &
                (pd.to_numeric(filtered_df['entry_price'], errors='coerce') <= price_zone_high)
            ]
            filtered_count = original_count - len(filtered_df)
            if filtered_count > 0:
                logger.info(f"PRICE_ZONES filter [{price_zone_low}, {price_zone_high}]: Filtered out {filtered_count} trades outside price zone for {strike_type}")
        
        # Drop entry_time_dt column (used only for matching) and ensure entry_time is simple format
        if 'entry_time_dt' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['entry_time_dt'])
        
        # Ensure entry_time is in simple format (HH:MM:SS) - it should already be from source files
        # But if it's datetime, convert it
        if 'entry_time' in filtered_df.columns:
            if pd.api.types.is_datetime64_any_dtype(filtered_df['entry_time']):
                filtered_df['entry_time'] = filtered_df['entry_time'].dt.strftime('%H:%M:%S')
            elif isinstance(filtered_df['entry_time'].iloc[0] if len(filtered_df) > 0 else None, str):
                # If it's already a string, ensure it's in HH:MM:SS format
                filtered_df['entry_time'] = filtered_df['entry_time'].str.split().str[-1].str[:8]
        
        # Ensure exit_time is also in simple format
        if 'exit_time' in filtered_df.columns:
            if pd.api.types.is_datetime64_any_dtype(filtered_df['exit_time']):
                filtered_df['exit_time'] = filtered_df['exit_time'].dt.strftime('%H:%M:%S')
            elif isinstance(filtered_df['exit_time'].iloc[0] if len(filtered_df) > 0 else None, str):
                filtered_df['exit_time'] = filtered_df['exit_time'].str.split().str[-1].str[:8]
        
        # Save results
        filtered_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(filtered_df)} sentiment-filtered trades to {output_file}")
        
        # Generate summary statistics for this strike type
        logger.info(f"\n--- {strike_type.upper()} MARKET SENTIMENT FILTERING SUMMARY ---")
        logger.info(f"Total trades before filtering: {len(all_trades)}")
        logger.info(f"Total trades after filtering: {len(filtered_df)}")
        logger.info(f"Filtering efficiency: {len(filtered_df)/len(all_trades)*100:.1f}%")
        
        # Breakdown by sentiment
        sentiment_counts = filtered_df['market_sentiment'].value_counts()
        logger.info("\nTrades by sentiment:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count} trades")
        
        # Breakdown by option type
        option_counts = filtered_df['option_type'].value_counts()
        logger.info("\nTrades by option type:")
        for option_type, count in option_counts.items():
            logger.info(f"  {option_type}: {count} trades")
        
        # Performance metrics
        total_pnl = filtered_df['pnl'].sum()
        wins = (filtered_df['pnl'] > 0).sum()
        win_rate = (wins / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        
        # Calculate un-filtered P&L
        unfiltered_pnl = all_trades['pnl'].sum()
        
        logger.info(f"\nPerformance:")
        logger.info(f"  Un-filtered P&L: {unfiltered_pnl:.2f}%")
        logger.info(f"  Filtered P&L: {total_pnl:.2f}%")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total Trades: {len(filtered_df)}")
        logger.info(f"----------------------------------------")
        
        # Add to summary data
        summary_data.append({
            'Strike Type': strike_type,
            'Total Trades': len(all_trades),
            'Filtered Trades': len(filtered_df),
            'Winning Trades': wins,  # Add winning trades count for proper aggregation
            'Filtering Efficiency': f"{len(filtered_df)/len(all_trades)*100:.1f}%",
            'Un-Filtered P&L': f"{unfiltered_pnl:.2f}%",
            'Filtered P&L': f"{total_pnl:.2f}%",
            'Win Rate': f"{win_rate:.1f}%"
        })
    
    # Generate summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = base_dir / f"{entry_type_lower}_static_market_sentiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\n📊 SUMMARY CSV GENERATED: {summary_file}")
        logger.info("=" * 60)
        logger.info("MARKET SENTIMENT FILTERING SUMMARY TABLE")
        logger.info("=" * 60)
        print(summary_df.to_string(index=False))
        logger.info("=" * 60)

def main():
    """Main entry point - processes both Entry1 and Entry2"""
    # Get command line arguments
    if len(sys.argv) < 4:
        logger.error("Usage: python run_static_market_sentiment_filter.py <expiry_week> <day_label> <strike_type1> [strike_type2] ...")
        logger.error("Example: python run_static_market_sentiment_filter.py OCT20 OCT15 ATM OTM ITM")
        return
    
    expiry_week = sys.argv[1]  # e.g., OCT20
    day_label = sys.argv[2]    # e.g., OCT15
    strike_types = sys.argv[3:]  # e.g., ['ATM', 'OTM', 'ITM']
    
    # Load config to get enabled entry types
    base_dir = Path(__file__).parent
    config_path = base_dir / 'backtesting_config.yaml'
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Get enabled entry types from config
    enabled_entry_types = []
    strategy_config = config.get('STRATEGY', {}) if config else {}
    ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
    pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
    
    # Check if Entry1 is enabled for either CE or PE
    if ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
        enabled_entry_types.append('Entry1')
    
    # Check if Entry2 is enabled for either CE or PE
    if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
        enabled_entry_types.append('Entry2')
    
    # Check if Entry3 is enabled for either CE or PE
    if ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
        enabled_entry_types.append('Entry3')
    
    # Default to Entry2 if none specified (backward compatibility)
    if not enabled_entry_types:
        enabled_entry_types = ['Entry2']
    
    logger.info(f"Enabled entry types: {enabled_entry_types}")
    
    # Process only enabled entry types
    for entry_type in enabled_entry_types:
        _process_one_set(expiry_week, day_label, strike_types, entry_type)

if __name__ == "__main__":
    main()