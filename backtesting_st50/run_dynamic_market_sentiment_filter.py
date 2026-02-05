#!/usr/bin/env python3
"""
Market Sentiment Filter for Dynamic Trading Strategy
Filters trades based on market sentiment (BULLISH, BEARISH, NEUTRAL)
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
    
    # Load indicators_config.yaml to get CPR_MARKET_SENTIMENT_VERSION
    indicators_config_path = base_dir / 'indicators_config.yaml'
    sentiment_version = 'v2'  # Default to v2
    if indicators_config_path.exists():
        try:
            with open(indicators_config_path, 'r') as f:
                indicators_config = yaml.safe_load(f)
                sentiment_version = indicators_config.get('CPR_MARKET_SENTIMENT_VERSION', 'v2')
        except Exception as e:
            logger.warning(f"Could not load indicators_config.yaml: {e}, defaulting to v2")
    
    for config_path in possible_config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load price zones
                price_zones = config.get('BACKTESTING_ANALYSIS', {}).get('PRICE_ZONES', {})
                dynamic_atm_zone = price_zones.get('DYNAMIC_ATM', {})
                dynamic_otm_zone = price_zones.get('DYNAMIC_OTM', {})
                
                dynamic_atm_low = dynamic_atm_zone.get('LOW_PRICE', None)
                dynamic_atm_high = dynamic_atm_zone.get('HIGH_PRICE', None)
                dynamic_otm_low = dynamic_otm_zone.get('LOW_PRICE', None)
                dynamic_otm_high = dynamic_otm_zone.get('HIGH_PRICE', None)
                
                # Load time distribution filter
                time_filter_config = config.get('TIME_DISTRIBUTION_FILTER', {})
                time_filter_enabled = time_filter_config.get('ENABLED', False)
                time_zones = time_filter_config.get('TIME_ZONES', {})
                
                # Load market sentiment filter config
                sentiment_filter_config = config.get('MARKET_SENTIMENT_FILTER', {})
                sentiment_filter_enabled = sentiment_filter_config.get('ENABLED', True)  # Default to True for backward compatibility
                sentiment_mode = sentiment_filter_config.get('MODE', 'AUTO').upper()  # AUTO or MANUAL
                sentiment_version = sentiment_filter_config.get('SENTIMENT_VERSION', 'v2')  # Default to v2 (used when MODE=AUTO)
                manual_sentiment = sentiment_filter_config.get('MANUAL_SENTIMENT', 'NEUTRAL').upper()  # NEUTRAL, BULLISH, BEARISH (used when MODE=MANUAL)
                
                return {
                    'price_zones': {
                        'DYNAMIC_ATM': (dynamic_atm_low, dynamic_atm_high),
                        'DYNAMIC_OTM': (dynamic_otm_low, dynamic_otm_high),
                    },
                    'time_filter_enabled': time_filter_enabled,
                    'time_zones': time_zones,
                    'sentiment_filter_enabled': sentiment_filter_enabled,
                    'sentiment_mode': sentiment_mode,
                    'sentiment_version': sentiment_version,
                    'manual_sentiment': manual_sentiment,
                }
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
    
    return {
        'price_zones': {
            'DYNAMIC_ATM': (None, None),
            'DYNAMIC_OTM': (None, None),
        },
        'time_filter_enabled': False,
        'time_zones': {},
        'sentiment_filter_enabled': True,  # Default to True for backward compatibility
        'sentiment_mode': 'AUTO',  # Default to AUTO
        'sentiment_version': 'v2',  # Default to v2
        'manual_sentiment': 'NEUTRAL',  # Default to NEUTRAL
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

def _process_one_set(sentiment_df: pd.DataFrame, base_dir: Path, day_label: str, kind: str, entry_type: str = 'Entry2') -> dict:
    """Filter one set (ATM or OTM) and persist outputs. Returns summary metrics dict or None."""
    # Load config early to check if sentiment filter is enabled
    config = _load_config()
    sentiment_filter_enabled = config.get('sentiment_filter_enabled', True)  # Default to True for backward compatibility
    
    entry_type_lower = entry_type.lower()
    if kind == 'OTM':
        ce_path = base_dir / f'{entry_type_lower}_dynamic_otm_ce_trades.csv'
        pe_path = base_dir / f'{entry_type_lower}_dynamic_otm_pe_trades.csv'
        output_file = base_dir / f'{entry_type_lower}_dynamic_otm_mkt_sentiment_trades.csv'
    else:
        ce_path = base_dir / f'{entry_type_lower}_dynamic_atm_ce_trades.csv'
        pe_path = base_dir / f'{entry_type_lower}_dynamic_atm_pe_trades.csv'
        output_file = base_dir / f'{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv'

    # CRITICAL FIX: When sentiment filter is disabled, always use raw CE/PE files instead of filtered file
    # If sentiment filter is disabled, skip using cached filtered file and calculate directly from raw files
    should_regenerate = True
    if sentiment_filter_enabled and output_file.exists() and ce_path.exists() and pe_path.exists():
        try:
            # Check if source files are newer than filtered file
            output_mtime = output_file.stat().st_mtime
            ce_mtime = ce_path.stat().st_mtime
            pe_mtime = pe_path.stat().st_mtime
            # If both source files are older than output file, we can reuse it
            if ce_mtime < output_mtime and pe_mtime < output_mtime:
                filtered_df_existing = pd.read_csv(output_file)
                # Check for realized_pnl_pct, sentiment_pnl, or pnl column (in order of preference)
                pnl_col = 'realized_pnl_pct' if 'realized_pnl_pct' in filtered_df_existing.columns else ('sentiment_pnl' if 'sentiment_pnl' in filtered_df_existing.columns else 'pnl')
                if not filtered_df_existing.empty and pnl_col in filtered_df_existing.columns:
                    logger.info(f"Using existing filtered trades file: {output_file.name} (source files unchanged)")
                    should_regenerate = False
                    
                    # Filter out SKIPPED trades if trade_status column exists (after trailing stop processing)
                    executed_df = filtered_df_existing
                    if 'trade_status' in filtered_df_existing.columns:
                        executed_df = filtered_df_existing[
                            filtered_df_existing['trade_status'].str.contains('EXECUTED', na=False)
                        ]
                        logger.info(f"Filtering summary: {len(executed_df)} executed trades out of {len(filtered_df_existing)} total")
                    
                    # NOTE: CE/PE files remain as original total trades (not regenerated)
                    # Only the sentiment output file contains filtered trades
                    
                    # Calculate summary from executed trades only
                    total_pnl = executed_df[pnl_col].sum()
                    wins = (executed_df[pnl_col] > 0).sum()
                    win_rate = (wins / len(executed_df) * 100) if len(executed_df) > 0 else 0
                    
                    # Get total trades from source files for comparison
                    total_trades = 0
                    unfiltered_pnl = 0.0
                    try:
                        ce_trades = pd.read_csv(ce_path)
                        pe_trades = pd.read_csv(pe_path)
                        total_trades = len(ce_trades) + len(pe_trades)
                        # Use realized_pnl_pct, pnl, or sentiment_pnl column from CE/PE files (in order of preference)
                        pnl_col_ce = 'realized_pnl_pct' if 'realized_pnl_pct' in ce_trades.columns else ('pnl' if 'pnl' in ce_trades.columns else 'sentiment_pnl')
                        pnl_col_pe = 'realized_pnl_pct' if 'realized_pnl_pct' in pe_trades.columns else ('pnl' if 'pnl' in pe_trades.columns else 'sentiment_pnl')
                        unfiltered_pnl = ce_trades[pnl_col_ce].sum() + pe_trades[pnl_col_pe].sum()
                    except Exception:
                        pass
                    
                    return {
                        'Strike Type': f'DYNAMIC_{kind}',
                        'Total Trades': total_trades if total_trades > 0 else len(filtered_df_existing),
                        'Filtered Trades': len(executed_df),  # Count only executed trades
                        'Winning Trades': wins,
                        'Filtering Efficiency': f"{(len(executed_df)/total_trades*100):.1f}%" if total_trades > 0 else "100.0%",
                        'Un-Filtered P&L': f"{unfiltered_pnl:.2f}%",
                        'Filtered P&L': f"{total_pnl:.2f}%",  # P&L from executed trades only
                        'Win Rate': f"{win_rate:.1f}%",
                    }
            else:
                logger.info(f"Source files newer than filtered file - regenerating {output_file.name}")
        except Exception as e:
            logger.debug(f"Could not check file timestamps: {e}, will regenerate")
    
    if not should_regenerate:
        # This should not be reached, but just in case
        pass

    # Check if source files exist and are not empty
    files_exist = ce_path.exists() and pe_path.exists()
    ce_trades = None
    pe_trades = None
    files_have_data = False
    
    if files_exist:
        try:
            # Check if files have data (more than just headers)
            ce_size = ce_path.stat().st_size
            pe_size = pe_path.stat().st_size
            if ce_size > 2 and pe_size > 2:
                try:
                    ce_trades = pd.read_csv(ce_path)
                    pe_trades = pd.read_csv(pe_path)
                    files_have_data = not ce_trades.empty or not pe_trades.empty
                except pd.errors.EmptyDataError:
                    files_have_data = False
        except Exception:
            files_have_data = False
    
    # If sentiment filter is disabled, calculate summary directly from raw CE/PE files
    if not sentiment_filter_enabled and files_exist and files_have_data:
        try:
            # Calculate summary from raw CE/PE files (all trades included)
            # Use realized_pnl_pct, pnl, or sentiment_pnl column (in order of preference)
            pnl_col_ce = 'realized_pnl_pct' if 'realized_pnl_pct' in ce_trades.columns else ('pnl' if 'pnl' in ce_trades.columns else 'sentiment_pnl')
            pnl_col_pe = 'realized_pnl_pct' if 'realized_pnl_pct' in pe_trades.columns else ('pnl' if 'pnl' in pe_trades.columns else 'sentiment_pnl')
            
            # Filter out SKIPPED trades if trade_status column exists
            ce_executed = ce_trades
            pe_executed = pe_trades
            if 'trade_status' in ce_trades.columns:
                ce_executed = ce_trades[ce_trades['trade_status'].str.contains('EXECUTED', na=False)]
            if 'trade_status' in pe_trades.columns:
                pe_executed = pe_trades[pe_trades['trade_status'].str.contains('EXECUTED', na=False)]
            
            all_executed = pd.concat([ce_executed, pe_executed], ignore_index=True)
            total_trades = len(ce_trades) + len(pe_trades)
            total_pnl = ce_executed[pnl_col_ce].sum() + pe_executed[pnl_col_pe].sum()
            unfiltered_pnl = ce_trades[pnl_col_ce].sum() + pe_trades[pnl_col_pe].sum()
            
            # Calculate wins properly from both CE and PE
            wins = 0
            if len(ce_executed) > 0:
                wins += (ce_executed[pnl_col_ce] > 0).sum()
            if len(pe_executed) > 0:
                wins += (pe_executed[pnl_col_pe] > 0).sum()
            win_rate = (wins / len(all_executed) * 100) if len(all_executed) > 0 else 0
            
            logger.info(f"Market sentiment filter DISABLED - calculating summary from raw CE/PE files ({total_trades} total trades, {len(all_executed)} executed)")
            
            return {
                'Strike Type': f'DYNAMIC_{kind}',
                'Total Trades': total_trades,
                'Filtered Trades': len(all_executed),  # All executed trades (100% filtering efficiency)
                'Winning Trades': wins,
                'Filtering Efficiency': f"100.0%",  # All trades included when filter is disabled
                'Un-Filtered P&L': f"{unfiltered_pnl:.2f}%",
                'Filtered P&L': f"{total_pnl:.2f}%",  # P&L from all executed trades
                'Win Rate': f"{win_rate:.1f}%",
            }
        except Exception as e:
            logger.warning(f"Error calculating summary from raw files when filter disabled: {e}, will continue with normal flow")
    
    # If files don't exist or are empty, create empty sentiment-filtered file and return summary with 0 trades
    if not files_exist or not files_have_data:
        if not files_exist:
            logger.info(f"No {kind} trade files found ({ce_path.name}, {pe_path.name}) - creating empty sentiment-filtered file")
        else:
            logger.info(f"{kind} trade files are empty ({ce_path.name}, {pe_path.name}) - creating empty sentiment-filtered file")
        
        # Create empty sentiment-filtered file with proper columns so Phase 3.5 (MARK2MARKET) can process it
        # Note: Use realized_pnl_pct instead of pnl (pnl and realized_pnl are removed by convert_to_percentages)
        empty_df = pd.DataFrame(columns=['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'realized_pnl_pct', 'high', 'swing_low', 'market_sentiment', 'filter_status'])
        try:
            empty_df.to_csv(output_file, index=False)
            logger.info(f"Created empty sentiment-filtered file: {output_file}")
        except Exception as e:
            logger.warning(f"Could not create empty sentiment-filtered file {output_file}: {e}")
        
        return {
            'Strike Type': f'DYNAMIC_{kind}',
            'Total Trades': 0,
            'Filtered Trades': 0,
            'Winning Trades': 0,  # Add winning trades count
            'Filtering Efficiency': '0.0%',
            'Un-Filtered P&L': f"{0.0:.2f}%",
            'Filtered P&L': f"{0.0:.2f}%",
            'Win Rate': f"{0.0:.1f}%",
        }
    
    # At this point, ce_trades and pe_trades are guaranteed to be DataFrames with data

    # Detect year from sentiment_df to handle year transitions (e.g., 2025 -> 2026)
    detected_year = None
    if sentiment_df is not None and not sentiment_df.empty and 'date' in sentiment_df.columns:
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

    # Map day label to date using a generic function
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
        return None

    # Convert entry_time to datetime for sentiment matching, but keep original format for output
    ce_trades['entry_time_dt'] = pd.to_datetime(trade_date + ' ' + ce_trades['entry_time'].astype(str)).dt.tz_localize('Asia/Kolkata')
    pe_trades['entry_time_dt'] = pd.to_datetime(trade_date + ' ' + pe_trades['entry_time'].astype(str)).dt.tz_localize('Asia/Kolkata')
    ce_trades['option_type'] = 'CE'
    pe_trades['option_type'] = 'PE'
    
    # Drop entry_period column if it exists
    if 'entry_period' in ce_trades.columns:
        ce_trades = ce_trades.drop(columns=['entry_period'])
    if 'entry_period' in pe_trades.columns:
        pe_trades = pe_trades.drop(columns=['entry_period'])
    
    all_trades = pd.concat([ce_trades, pe_trades], ignore_index=True)

    # Check for realized_pnl_pct, sentiment_pnl, or pnl column (in order of preference)
    pnl_col_all = 'realized_pnl_pct' if 'realized_pnl_pct' in all_trades.columns else ('sentiment_pnl' if 'sentiment_pnl' in all_trades.columns else 'pnl')
    if pnl_col_all not in all_trades.columns:
        logger.error(f"Neither 'realized_pnl_pct', 'sentiment_pnl', nor 'pnl' column found in all_trades. Available columns: {list(all_trades.columns)}")
        raise KeyError(f"Neither 'realized_pnl_pct', 'sentiment_pnl', nor 'pnl' column found in trade data")
    
    unfiltered_pnl = all_trades[pnl_col_all].sum()
    filtered = []
    excluded = []  # Track excluded trades for audit
    
    # Use config already loaded at function start
    time_filter_enabled = config['time_filter_enabled']
    time_zones = config['time_zones']
    sentiment_mode = config.get('sentiment_mode', 'AUTO').upper()
    sentiment_version = config.get('sentiment_version', 'v2').lower()  # Get version once, use for all trades
    manual_sentiment = config.get('manual_sentiment', 'NEUTRAL').upper()  # Get manual sentiment if MODE=MANUAL
    
    # Log sentiment filter status
    if not sentiment_filter_enabled:
        logger.info(f"Market sentiment filter DISABLED - including all trades regardless of sentiment (Entry1 behavior)")
    elif sentiment_mode == 'MANUAL':
        logger.info(f"Market sentiment filter ENABLED - MANUAL mode with sentiment: {manual_sentiment}")
    else:
        logger.debug(f"Market sentiment filter ENABLED - AUTO mode using sentiment file (version: {sentiment_version})")
    
    # Skip sentiment filtering for Entry1 OR if sentiment filter is disabled (include all trades)
    if entry_type == 'Entry1' or not sentiment_filter_enabled:
        reason = "Entry1" if entry_type == 'Entry1' else "sentiment filter disabled"
        logger.info(f"{reason}: Skipping market sentiment filtering - including all {len(all_trades)} trades")
        for _, trade in all_trades.iterrows():
            entry_time = trade['entry_time_dt']
            # Still try to get sentiment for record-keeping, but don't filter based on it
            matching_sentiment = None
            matching_sentiment = None
            if sentiment_df is not None and not sentiment_df.empty:
                matching_rows = sentiment_df[sentiment_df['date'] == entry_time]
                if not matching_rows.empty:
                    matching_sentiment = matching_rows.iloc[0]['sentiment']
                else:
                    time_diff = abs((sentiment_df['date'] - entry_time).dt.total_seconds())
                    if time_diff.min() <= 60:
                        nearest_idx = time_diff.idxmin()
                        matching_sentiment = sentiment_df.loc[nearest_idx, 'sentiment']
            
            if matching_sentiment is not None:
                matching_sentiment = str(matching_sentiment).upper().strip()
            else:
                matching_sentiment = 'N/A'  # Default if no sentiment found
            
            # Apply time zone filter
            if not _is_time_zone_enabled(entry_time, time_filter_enabled, time_zones):
                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (TIME_ZONE)'})
                logger.debug(f"Time zone filter: Excluding trade at {entry_time}")
                continue
            
            # Include all trades regardless of sentiment
            filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
    else:
        # Apply sentiment filtering for Entry2, Entry3, etc. (only if sentiment filter is enabled)
        for _, trade in all_trades.iterrows():
            entry_time = trade['entry_time_dt']  # Use datetime for matching
            matching_sentiment = None
            matching_transition = None  # v3: Track sentiment transition status
            filter_reason = None
            
            # MANUAL MODE: Use fixed sentiment from config
            if sentiment_mode == 'MANUAL':
                matching_sentiment = manual_sentiment
                matching_transition = 'STABLE'  # Manual mode doesn't use transitions
                logger.debug(f"MANUAL mode: Using fixed sentiment {matching_sentiment} for trade at {entry_time}")
            # AUTO MODE: Find the sentiment for the exact entry time from sentiment file
            elif sentiment_df is not None and not sentiment_df.empty:
                matching_rows = sentiment_df[sentiment_df['date'] == entry_time]
                if not matching_rows.empty:
                    matching_sentiment = matching_rows.iloc[0]['sentiment']
                    # v3: Get sentiment_transition if available (fallback to 'STABLE' if column doesn't exist)
                    if 'sentiment_transition' in matching_rows.columns:
                        matching_transition = matching_rows.iloc[0].get('sentiment_transition', 'STABLE')
                    else:
                        matching_transition = 'STABLE'  # Default for v2 compatibility
                else:
                    # Try to find nearest sentiment within 60 seconds (fallback for timestamp mismatches)
                    time_diff = abs((sentiment_df['date'] - entry_time).dt.total_seconds())
                    if time_diff.min() <= 60:
                        nearest_idx = time_diff.idxmin()
                        matching_sentiment = sentiment_df.loc[nearest_idx, 'sentiment']
                        # v3: Get sentiment_transition if available
                        if 'sentiment_transition' in sentiment_df.columns:
                            matching_transition = sentiment_df.loc[nearest_idx].get('sentiment_transition', 'STABLE')
                        else:
                            matching_transition = 'STABLE'  # Default for v2 compatibility
                        logger.debug(f"Using nearest sentiment (within 60s) for trade at {entry_time}")
            
            # Normalize sentiment to uppercase for comparison
            if matching_sentiment is not None:
                matching_sentiment = str(matching_sentiment).upper().strip()
            
            # Normalize transition status (v3) - only used in AUTO mode
            if sentiment_mode == 'AUTO':
                if matching_transition is not None:
                    matching_transition = str(matching_transition).upper().strip()
                else:
                    matching_transition = 'STABLE'  # Default if not found
            else:
                matching_transition = 'STABLE'  # Manual mode doesn't use transitions
            
            # Check if sentiment is available (only required in AUTO mode)
            if sentiment_mode == 'AUTO' and not matching_sentiment:
                excluded.append({**trade.to_dict(), 'market_sentiment': 'N/A', 'filter_status': 'EXCLUDED (NO_SENTIMENT)'})
                logger.debug(f"No sentiment found for trade at {entry_time}, skipping")
                continue
            
            # Apply time zone filter
            if not _is_time_zone_enabled(entry_time, time_filter_enabled, time_zones):
                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (TIME_ZONE)'})
                logger.debug(f"Time zone filter: Excluding trade at {entry_time}")
                continue
            
            # Handle DISABLE sentiment - reject all trades (no trades allowed)
            if matching_sentiment == 'DISABLE':
                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (DISABLE_SENTIMENT)'})
                logger.debug(f"DISABLE sentiment: Rejecting trade at {entry_time} (no trades allowed)")
                continue
            
            # Handle NEUTRAL sentiment - allow both CE and PE trades (all trades allowed)
            # This applies to both AUTO and MANUAL modes
            if matching_sentiment == 'NEUTRAL':
                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
                logger.debug(f"NEUTRAL sentiment: Including {trade['option_type']} trade at {entry_time} (both CE and PE allowed)")
            
            # MANUAL MODE: Simple filtering based on fixed sentiment (BULLISH/BEARISH)
            elif sentiment_mode == 'MANUAL':
                if matching_sentiment == 'BULLISH':
                    if trade['option_type'] == 'CE':
                        filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (MANUAL_BULLISH)'})
                        logger.debug(f"MANUAL BULLISH: Including CE trade at {entry_time}")
                    else:
                        excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (MANUAL_BULLISH_ONLY_CE)'})
                        logger.debug(f"MANUAL BULLISH: Excluding PE trade at {entry_time} (only CE allowed)")
                elif matching_sentiment == 'BEARISH':
                    if trade['option_type'] == 'PE':
                        filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (MANUAL_BEARISH)'})
                        logger.debug(f"MANUAL BEARISH: Including PE trade at {entry_time}")
                    else:
                        excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (MANUAL_BEARISH_ONLY_PE)'})
                        logger.debug(f"MANUAL BEARISH: Excluding CE trade at {entry_time} (only PE allowed)")
                else:
                    # Unknown manual sentiment
                    excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (UNKNOWN_MANUAL_SENTIMENT)'})
                    logger.warning(f"Unknown manual sentiment value '{matching_sentiment}' for trade at {entry_time}, skipping")
                    continue
            
            # AUTO MODE: v2: TRADITIONAL FILTERING (simple trend-following logic)
            elif sentiment_mode == 'AUTO' and sentiment_version == 'v2':
                # v2: Traditional filtering - BULLISH → CE only, BEARISH → PE only
                if matching_sentiment == 'BULLISH':
                    if trade['option_type'] == 'CE':
                        filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
                        logger.debug(f"BULLISH sentiment (v2): Including CE trade at {entry_time}")
                    else:
                        excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (BULLISH_ONLY_CE)'})
                        logger.debug(f"BULLISH sentiment (v2): Excluding PE trade at {entry_time} (only CE allowed)")
                elif matching_sentiment == 'BEARISH':
                    if trade['option_type'] == 'PE':
                        filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
                        logger.debug(f"BEARISH sentiment (v2): Including PE trade at {entry_time}")
                    else:
                        excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (BEARISH_ONLY_PE)'})
                        logger.debug(f"BEARISH sentiment (v2): Excluding CE trade at {entry_time} (only PE allowed)")
                else:
                    # Unknown sentiment in v2
                    excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (UNKNOWN_SENTIMENT)'})
                    logger.warning(f"Unknown sentiment value '{matching_sentiment}' for trade at {entry_time}, skipping")
                    continue
            
            # AUTO MODE: v3/v5: TRANSITION-BASED + SELECTIVE ENTRY2 REVERSAL FILTERING
            elif sentiment_mode == 'AUTO' and sentiment_version in ['v3', 'v4', 'v5']:
                # v3/v5: Transition-based filtering with selective Entry2 reversal
                is_transitioning = matching_transition in ['JUST_CHANGED', 'TRANSITIONING']
                
                # Handle BULLISH sentiment
                if matching_sentiment == 'BULLISH':
                    if is_transitioning:
                        # v3/v5: During transitions, allow only PE (reversal strategy - PE performs well in transitions)
                        if trade['option_type'] == 'PE':
                            filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'INCLUDED (TRANSITION: {matching_transition} - PE_ONLY)'})
                            logger.debug(f"BULLISH sentiment (TRANSITION {matching_transition}): Including PE trade at {entry_time} (PE performs well in transitions)")
                        else:
                            excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (TRANSITION_ONLY_PE)'})
                            logger.debug(f"BULLISH sentiment (TRANSITION {matching_transition}): Excluding CE trade at {entry_time} (only PE allowed during transitions)")
                    else:
                        # v5: During stable BULLISH, allow both CE (traditional) and PE (Entry2 reversal down)
                        # v3: During stable BULLISH, allow only CE (traditional)
                        if sentiment_version == 'v5':
                            if trade['option_type'] == 'CE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (TRADITIONAL)'})
                                logger.debug(f"BULLISH sentiment (STABLE): Including CE trade at {entry_time} (traditional trend-following)")
                            elif trade['option_type'] == 'PE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (ENTRY2_REVERSAL)'})
                                logger.debug(f"BULLISH sentiment (STABLE): Including PE trade at {entry_time} (Entry2 reversal - betting on reversal down)")
                        else:  # v3
                            if trade['option_type'] == 'CE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
                                logger.debug(f"BULLISH sentiment (STABLE): Including CE trade at {entry_time}")
                            else:
                                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (BULLISH_ONLY_CE)'})
                                logger.debug(f"BULLISH sentiment (STABLE): Excluding PE trade at {entry_time} (only CE allowed)")
                
                # Handle BEARISH sentiment
                elif matching_sentiment == 'BEARISH':
                    if is_transitioning:
                        # v3/v5: During transitions, allow only PE (reversal strategy - PE performs well in transitions)
                        if trade['option_type'] == 'PE':
                            filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'INCLUDED (TRANSITION: {matching_transition} - PE_ONLY)'})
                            logger.debug(f"BEARISH sentiment (TRANSITION {matching_transition}): Including PE trade at {entry_time} (PE performs well in transitions)")
                        else:
                            excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (TRANSITION_ONLY_PE)'})
                            logger.debug(f"BEARISH sentiment (TRANSITION {matching_transition}): Excluding CE trade at {entry_time} (only PE allowed during transitions)")
                    else:
                        # v5: During stable BEARISH, allow both PE (traditional) and CE (Entry2 reversal up)
                        # v3: During stable BEARISH, allow only PE (traditional)
                        if sentiment_version == 'v5':
                            if trade['option_type'] == 'PE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (TRADITIONAL)'})
                                logger.debug(f"BEARISH sentiment (STABLE): Including PE trade at {entry_time} (traditional trend-following)")
                            elif trade['option_type'] == 'CE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED (ENTRY2_REVERSAL)'})
                                logger.debug(f"BEARISH sentiment (STABLE): Including CE trade at {entry_time} (Entry2 reversal - betting on reversal up)")
                        else:  # v3
                            if trade['option_type'] == 'PE':
                                filtered.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'INCLUDED'})
                                logger.debug(f"BEARISH sentiment (STABLE): Including PE trade at {entry_time}")
                            else:
                                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': 'EXCLUDED (BEARISH_ONLY_PE)'})
                                logger.debug(f"BEARISH sentiment (STABLE): Excluding CE trade at {entry_time} (only PE allowed)")
                else:
                    # Unknown sentiment in v3/v5
                    excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (UNKNOWN_SENTIMENT)'})
                    logger.warning(f"Unknown sentiment value '{matching_sentiment}' for trade at {entry_time}, skipping")
                    continue
            
            # Unknown sentiment mode/version or sentiment value - log warning and skip
            else:
                excluded.append({**trade.to_dict(), 'market_sentiment': matching_sentiment, 'filter_status': f'EXCLUDED (UNKNOWN_SENTIMENT_MODE_OR_VERSION)'})
                logger.warning(f"Unknown sentiment mode '{sentiment_mode}', version '{sentiment_version}' or sentiment value '{matching_sentiment}' for trade at {entry_time}, skipping")
                continue

    if not filtered:
        logger.info(f"No {kind} trades passed the sentiment filter")
        
        # Create empty DataFrame with expected columns for consistency
        # Get column structure from first trade (if available) or use all_trades structure
        if len(all_trades) > 0:
            # Create empty DataFrame with same columns as trades plus sentiment columns
            sample_trade = all_trades.iloc[0].to_dict()
            expected_columns = list(sample_trade.keys())
            # Remove entry_time_dt if present (it's only for matching)
            if 'entry_time_dt' in expected_columns:
                expected_columns.remove('entry_time_dt')
            # Add sentiment-related columns
            if 'market_sentiment' not in expected_columns:
                expected_columns.append('market_sentiment')
            if 'filter_status' not in expected_columns:
                expected_columns.append('filter_status')
            filtered_df = pd.DataFrame(columns=expected_columns)
        else:
            # Fallback: create empty DataFrame with minimal expected columns
            filtered_df = pd.DataFrame(columns=['symbol', 'option_type', 'entry_time', 'exit_time', 
                                                'entry_price', 'exit_price', 'realized_pnl_pct', 'market_sentiment', 'filter_status'])
        
        # Write empty file with headers for consistency
        try:
            filtered_df.to_csv(output_file, index=False)
            logger.info(f"Created empty {kind} sentiment-filtered trades file (no trades passed filter): {output_file}")
        except Exception as e:
            logger.warning(f"Could not create empty output file: {e}")
        
        return {
            'Strike Type': f'DYNAMIC_{kind}',
            'Total Trades': len(all_trades),
            'Filtered Trades': 0,
            'Winning Trades': 0,  # Add winning trades count
            'Filtering Efficiency': '0.0%',
            'Un-Filtered P&L': f"{unfiltered_pnl:.2f}%",
            'Filtered P&L': f"{0.0:.2f}%",
            'Win Rate': f"{0.0:.1f}%",
        }

    # Combine included and excluded trades for full audit trail
    # Only include filtered trades in the output file (as per original behavior)
    filtered_df = pd.DataFrame(filtered)
    
    # Log excluded trades for debugging
    if excluded:
        excluded_df = pd.DataFrame(excluded)
        logger.info(f"Excluded {len(excluded_df)} trades: {excluded_df['filter_status'].value_counts().to_dict()}")
    
    # Apply PRICE_ZONES filter as post-processing (after sentiment filtering)
    price_zones_config = _load_price_zones_config()
    strike_type_key = f'DYNAMIC_{kind}'
    price_zone_low, price_zone_high = price_zones_config.get(strike_type_key, (None, None))
    
    sentiment_filtered_count = len(filtered_df)
    if price_zone_low is not None and price_zone_high is not None:
        original_count = len(filtered_df)
        # Filter by entry_price (inclusive range)
        filtered_df = filtered_df[
            (pd.to_numeric(filtered_df['entry_price'], errors='coerce') >= price_zone_low) &
            (pd.to_numeric(filtered_df['entry_price'], errors='coerce') <= price_zone_high)
        ]
        filtered_count = original_count - len(filtered_df)
        if filtered_count > 0:
            logger.info(f"PRICE_ZONES filter [{price_zone_low}, {price_zone_high}]: Filtered out {filtered_count} trades outside price zone")
            if len(filtered_df) == 0:
                logger.warning(f"All {original_count} sentiment-filtered trades were removed by PRICE_ZONES filter")
                logger.warning(f"Summary: {len(all_trades)} total trades → {sentiment_filtered_count} passed sentiment → 0 passed price zone filter")
    
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
    
    # Sort by entry_time in descending order (most recent first)
    if 'entry_time' in filtered_df.columns and len(filtered_df) > 0:
        # Convert entry_time to datetime for proper sorting, then back to string
        try:
            filtered_df['_sort_time'] = pd.to_datetime(filtered_df['entry_time'], format='%H:%M:%S', errors='coerce')
            filtered_df = filtered_df.sort_values('_sort_time', ascending=False, na_position='last')
            filtered_df = filtered_df.drop(columns=['_sort_time'])
            logger.debug(f"Sorted {len(filtered_df)} trades by entry_time in descending order")
        except Exception as e:
            logger.warning(f"Could not sort by entry_time: {e}, keeping original order")
    
    # Convert high and swing_low to percentages, remove pnl and realized_pnl columns
    def convert_to_percentages(df):
        """Convert high and swing_low to percentages relative to entry_price, remove pnl and realized_pnl"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Convert high to percentage
        if 'high' in df.columns and 'entry_price' in df.columns:
            def calc_high_pct(row):
                entry_price = row.get('entry_price', 0)
                high = row.get('high', 0)
                if pd.notna(entry_price) and pd.notna(high) and entry_price > 0:
                    # Check if high is already a percentage (if it's between -200 and 200, it's likely a percentage)
                    # Absolute prices should be much larger (typically 40-600+ for options)
                    if abs(high) < 200 and abs(high) < entry_price * 0.5:
                        # Already a percentage, clamp to >= 0 (high cannot be negative)
                        return round(max(high, 0), 2)
                    # Otherwise, convert from absolute price to percentage
                    # High should be >= entry_price, so percentage should be >= 0
                    pct = ((high - entry_price) / entry_price) * 100
                    return round(max(pct, 0), 2)  # Clamp to >= 0
                return None
            df['high'] = df.apply(calc_high_pct, axis=1)
        
        # Convert swing_low to percentage
        if 'swing_low' in df.columns and 'entry_price' in df.columns:
            def calc_swing_low_pct(row):
                entry_price = row.get('entry_price', 0)
                swing_low = row.get('swing_low', 0)
                if pd.notna(entry_price) and pd.notna(swing_low) and entry_price > 0:
                    # Check if swing_low is already a percentage (if it's between -200 and 200, it's likely a percentage)
                    # Absolute prices should be much larger (typically 40-600+ for options)
                    if abs(swing_low) < 200 and abs(swing_low) < entry_price * 0.5:
                        # Already a percentage, return as-is
                        return round(swing_low, 2)
                    # Otherwise, convert from absolute price to percentage
                    return round(((swing_low - entry_price) / entry_price) * 100, 2)
                return None
            df['swing_low'] = df.apply(calc_swing_low_pct, axis=1)
        
        # Remove pnl and realized_pnl columns (keep only realized_pnl_pct)
        columns_to_remove = []
        if 'pnl' in df.columns:
            columns_to_remove.append('pnl')
        if 'realized_pnl' in df.columns:
            columns_to_remove.append('realized_pnl')
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        
        return df
    
    # Apply conversion before saving
    filtered_df = convert_to_percentages(filtered_df)
    
    # Try to write the file with error handling
    try:
        filtered_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(filtered_df)} sentiment-filtered {kind} trades to {output_file} (sorted by entry_time descending)")
    except PermissionError as e:
        logger.error(f"Permission denied writing to {output_file}. File might be open in another application.")
        logger.error(f"Error details: {e}")
        # Try to write to a temporary file instead
        temp_file = output_file.with_suffix('.tmp.csv')
        try:
            filtered_df.to_csv(temp_file, index=False)
            logger.info(f"Saved to temporary file: {temp_file}")
            logger.info("Please close any applications using the original file and rename the temp file.")
        except Exception as temp_e:
            logger.error(f"Failed to write to temporary file: {temp_e}")
            raise
    except Exception as e:
        logger.error(f"Error writing to {output_file}: {e}")
        raise

    # NOTE: CE/PE files remain as original total trades (not regenerated after sentiment filtering)
    # Only the sentiment output file (*_mkt_sentiment_trades.csv) contains filtered trades
    # This preserves the original trade counts for analysis
    
    # Determine executed_df for summary calculation
    if 'trade_status' in filtered_df.columns:
        # Filter out SKIPPED trades for summary calculation
        executed_df = filtered_df[filtered_df['trade_status'].str.contains('EXECUTED', na=False)]
        logger.info(f"Summary calculation: Using {len(executed_df)} executed trades out of {len(filtered_df)} total (excluding {len(filtered_df) - len(executed_df)} skipped)")
    else:
        # No trailing stop applied, use all filtered trades
        executed_df = filtered_df
        logger.info(f"No trailing stop applied - using all {len(executed_df)} filtered trades")

    # Check for realized_pnl_pct, sentiment_pnl, or pnl column (in order of preference)
    pnl_col = 'realized_pnl_pct' if 'realized_pnl_pct' in filtered_df.columns else ('sentiment_pnl' if 'sentiment_pnl' in filtered_df.columns else 'pnl')
    
    # Calculate P&L from executed trades only
    total_pnl = executed_df[pnl_col].sum() if len(executed_df) > 0 else 0
    wins = (executed_df[pnl_col] > 0).sum() if len(executed_df) > 0 else 0
    win_rate = (wins / len(executed_df) * 100) if len(executed_df) > 0 else 0

    # Calculate unfiltered P&L using the detected column
    unfiltered_pnl_value = all_trades[pnl_col_all].sum() if pnl_col_all in all_trades.columns else 0
    
    return {
        'Strike Type': f'DYNAMIC_{kind}',
        'Total Trades': len(all_trades),
        'Filtered Trades': len(executed_df),  # Count only executed trades
        'Winning Trades': wins,  # Add winning trades count for proper aggregation
        'Filtering Efficiency': f"{len(executed_df)/len(all_trades)*100:.1f}%" if len(all_trades) > 0 else "0.0%",
        'Un-Filtered P&L': f"{unfiltered_pnl_value:.2f}%",
        'Filtered P&L': f"{total_pnl:.2f}%",  # P&L from executed trades only
        'Win Rate': f"{win_rate:.1f}%",
    }


def main():
    # Args
    if len(sys.argv) != 3:
        logger.error("Usage: python run_dynamic_market_sentiment_filter.py <expiry_week> <day_label>")
        logger.error("Example: python run_dynamic_market_sentiment_filter.py OCT20_DYNAMIC OCT16")
        return

    expiry_week = sys.argv[1]
    day_label = sys.argv[2]
    
    # Extract expiry week from the input (remove _DYNAMIC suffix if present)
    if expiry_week.endswith('_DYNAMIC'):
        expiry_week = expiry_week[:-8]  # Remove '_DYNAMIC' suffix

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

    data_base_dir = Path(f"data/{expiry_week}_DYNAMIC/{day_label}")
    sentiment_file = data_base_dir / f"nifty_market_sentiment_{date_suffix}.csv"
    
    # Load config first to check if sentiment filter is enabled
    script_base_dir = Path(__file__).parent
    config_path = script_base_dir / 'backtesting_config.yaml'
    config = {}
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Check if sentiment filter is enabled
    sentiment_filter_config = config.get('MARKET_SENTIMENT_FILTER', {})
    sentiment_filter_enabled = sentiment_filter_config.get('ENABLED', False)
    sentiment_mode = sentiment_filter_config.get('MODE', 'AUTO').upper()
    
    # Load sentiment data only if sentiment filter is enabled
    sentiment_df = None
    if sentiment_filter_enabled:
        if sentiment_mode == 'MANUAL':
            # MANUAL mode: No sentiment file needed, use fixed sentiment from config
            logger.info(f"MANUAL mode: Using fixed sentiment '{sentiment_filter_config.get('MANUAL_SENTIMENT', 'NEUTRAL').upper()}' - no sentiment file required")
            sentiment_df = None  # Not needed for manual mode
        else:
            # AUTO mode: Load sentiment file
            if not sentiment_file.exists():
                logger.error(f"Market sentiment file not found: {sentiment_file}")
                logger.error("Cannot proceed with AUTO sentiment filtering - sentiment filter is enabled but file is missing")
                return
            logger.info("AUTO mode: Loading market sentiment data from file...")
            sentiment_df = pd.read_csv(sentiment_file)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    else:
        if sentiment_file.exists():
            logger.info("Sentiment filter is disabled, but sentiment file exists - loading for reference")
            sentiment_df = pd.read_csv(sentiment_file)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        else:
            logger.info("Sentiment filter is disabled and sentiment file not found - will generate summary from raw trades")
    
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

    # Process only enabled entry types for OTM and ATM
    # If sentiment_df is None, create an empty DataFrame to avoid errors
    if sentiment_df is None:
        sentiment_df = pd.DataFrame(columns=['date', 'sentiment'])
    
    for entry_type in enabled_entry_types:
        summaries = []
        otm_summary = _process_one_set(sentiment_df, data_base_dir, day_label, 'OTM', entry_type)
        if otm_summary:
            summaries.append(otm_summary)
        atm_summary = _process_one_set(sentiment_df, data_base_dir, day_label, 'ATM', entry_type)
        if atm_summary:
            summaries.append(atm_summary)

        # Always create summary file, even if all trades are 0 (so day appears in HTML report)
        if not summaries:
            logger.warning(f"No summaries generated for {entry_type} (OTM/ATM both returned None) - this should not happen")
            continue

        summary_df = pd.DataFrame(summaries)
        entry_type_lower = entry_type.lower()
        summary_file = data_base_dir / f"{entry_type_lower}_dynamic_market_sentiment_summary.csv"
        
        # Try to write the summary file with error handling
        try:
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"\n📊 SUMMARY CSV GENERATED: {summary_file}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {summary_file}. File might be open in another application.")
            logger.error(f"Error details: {e}")
            # Try to write to a temporary file instead
            temp_summary_file = summary_file.with_suffix('.tmp.csv')
            try:
                summary_df.to_csv(temp_summary_file, index=False)
                logger.info(f"Saved summary to temporary file: {temp_summary_file}")
                logger.info("Please close any applications using the original file and rename the temp file.")
            except Exception as temp_e:
                logger.error(f"Failed to write to temporary summary file: {temp_e}")
                raise
        except Exception as e:
            logger.error(f"Error writing to {summary_file}: {e}")
            raise
        
        logger.info("=" * 60)
        logger.info(f"MARKET SENTIMENT FILTERING SUMMARY TABLE ({entry_type})")
        logger.info("=" * 60)
        print(summary_df.to_string(index=False))
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
