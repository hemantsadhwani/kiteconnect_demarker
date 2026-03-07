#!/usr/bin/env python3
"""
Regenerate CE and PE trade files from sentiment-filtered file after MARK2MARKET filtering.

This script splits the sentiment-filtered file (which has MARK2MARKET columns) back into
separate CE and PE files, preserving all MARK2MARKET columns.
"""
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def regenerate_ce_pe_files(sentiment_file: Path):
    """Regenerate CE and PE files from sentiment-filtered file"""
    if not sentiment_file.exists():
        logger.warning(f"Sentiment file not found: {sentiment_file}")
        return False
    
    # Check if file is empty
    if sentiment_file.stat().st_size == 0:
        logger.info(f"Sentiment file is empty: {sentiment_file}")
        # Create empty CE/PE files
        base_dir = sentiment_file.parent
        file_name = sentiment_file.name
        
        # Determine file type (ATM or OTM) and entry type
        if 'atm' in file_name.lower():
            kind = 'atm'
        elif 'otm' in file_name.lower():
            kind = 'otm'
        else:
            logger.error(f"Could not determine strike type from filename: {file_name}")
            return False
        
        if 'entry1' in file_name.lower():
            entry_type = 'entry1'
        elif 'entry2' in file_name.lower():
            entry_type = 'entry2'
        else:
            logger.error(f"Could not determine entry type from filename: {file_name}")
            return False
        
        # Create empty DataFrames with proper columns
        # Note: Use realized_pnl_pct instead of sentiment_pnl/realized_pnl (they are removed by convert_to_percentages)
        empty_df = pd.DataFrame(columns=['symbol', 'option_type', 'entry_time', 'exit_time', 
                                        'entry_price', 'exit_price', 'realized_pnl_pct', 
                                        'high', 'swing_low', 'running_capital', 'high_water_mark', 
                                        'drawdown_limit', 'trade_status'])
        
        ce_file = base_dir / f"{entry_type}_dynamic_{kind}_ce_trades.csv"
        pe_file = base_dir / f"{entry_type}_dynamic_{kind}_pe_trades.csv"
        
        empty_df.to_csv(ce_file, index=False)
        empty_df.to_csv(pe_file, index=False)
        logger.info(f"Created empty CE/PE files: {ce_file.name}, {pe_file.name}")
        return True
    
    try:
        df = pd.read_csv(sentiment_file)
        # Check if file only has headers (no data rows)
        if len(df) == 0:
            logger.warning(f"Sentiment file has no data rows (only headers): {sentiment_file}")
            logger.warning(f"SKIPPING regeneration - will keep original CE/PE files from Phase 2")
            # Don't overwrite original CE/PE files if sentiment file is empty
            # This preserves the original trades from Phase 2
            return True  # Return True to indicate "success" (we're intentionally skipping)
    except pd.errors.EmptyDataError:
        logger.warning(f"Sentiment file is completely empty (no columns): {sentiment_file}")
        logger.warning(f"SKIPPING regeneration - will keep original CE/PE files from Phase 2")
        return True  # Return True to indicate "success" (we're intentionally skipping)
    except Exception as e:
        logger.error(f"Error reading sentiment file: {e}")
        return False
    
    if df.empty:
        logger.info(f"Sentiment file has no data rows: {sentiment_file}")
        # Create empty CE/PE files directly (don't recurse)
        base_dir = sentiment_file.parent
        file_name = sentiment_file.name
        
        # Determine file type (ATM or OTM) and entry type
        if 'atm' in file_name.lower():
            kind = 'atm'
        elif 'otm' in file_name.lower():
            kind = 'otm'
        else:
            logger.error(f"Could not determine strike type from filename: {file_name}")
            return False
        
        if 'entry1' in file_name.lower():
            entry_type = 'entry1'
        elif 'entry2' in file_name.lower():
            entry_type = 'entry2'
        else:
            logger.error(f"Could not determine entry type from filename: {file_name}")
            return False
        
        # Create empty DataFrames with proper columns (use columns from df if available, otherwise default)
        if len(df.columns) > 0:
            # Remove sentiment-specific columns
            columns = [col for col in df.columns if col not in ['market_sentiment', 'filter_status']]
            empty_df = pd.DataFrame(columns=columns)
        else:
            # Note: Use realized_pnl_pct instead of sentiment_pnl/realized_pnl (they are removed by convert_to_percentages)
            empty_df = pd.DataFrame(columns=['symbol', 'option_type', 'entry_time', 'exit_time', 
                                            'entry_price', 'exit_price', 'realized_pnl_pct', 
                                            'high', 'swing_low', 'running_capital', 'high_water_mark', 
                                            'drawdown_limit', 'trade_status'])
        
        ce_file = base_dir / f"{entry_type}_dynamic_{kind}_ce_trades.csv"
        pe_file = base_dir / f"{entry_type}_dynamic_{kind}_pe_trades.csv"
        
        empty_df.to_csv(ce_file, index=False)
        empty_df.to_csv(pe_file, index=False)
        logger.info(f"Created empty CE/PE files: {ce_file.name}, {pe_file.name}")
        return True
    
    # Determine file paths
    base_dir = sentiment_file.parent
    file_name = sentiment_file.name
    
    # Determine file type (ATM or OTM) and entry type
    if 'atm' in file_name.lower():
        kind = 'atm'
    elif 'otm' in file_name.lower():
        kind = 'otm'
    else:
        logger.error(f"Could not determine strike type from filename: {file_name}")
        return False
    
    if 'entry1' in file_name.lower():
        entry_type = 'entry1'
    elif 'entry2' in file_name.lower():
        entry_type = 'entry2'
    else:
        logger.error(f"Could not determine entry type from filename: {file_name}")
        return False
    
    ce_file = base_dir / f"{entry_type}_dynamic_{kind}_ce_trades.csv"
    pe_file = base_dir / f"{entry_type}_dynamic_{kind}_pe_trades.csv"
    
    # Preserve all trades from Phase 2: merge sentiment file (filtered subset) with existing CE/PE files
    # so we don't drop trades that were excluded by sentiment filter (e.g. post-12pm roll trades).
    def merge_with_original(original_file: Path, sentiment_subset: pd.DataFrame, option_type: str):
        if not original_file.exists():
            return sentiment_subset.copy()
        try:
            orig = pd.read_csv(original_file)
            if orig.empty:
                return sentiment_subset.copy()
            # Keep only rows of this option_type (CE or PE)
            orig = orig[orig['option_type'] == option_type].copy()
            if orig.empty:
                return sentiment_subset.copy()
            # Match by (entry_time, entry_price) for stability (avoids hyperlink/symbol differences)
            def row_key(r):
                et = str(r.get('entry_time', ''))
                ep = r.get('entry_price')
                ep = float(ep) if ep is not None and pd.notna(ep) and str(ep).strip() else 0.0
                return (et, ep)
            sentiment_subset = sentiment_subset.copy()
            sentiment_keys = set(row_key(sentiment_subset.loc[i]) for i in sentiment_subset.index)
            orig['_key'] = orig.apply(row_key, axis=1)
            extra = orig[~orig['_key'].apply(lambda k: k in sentiment_keys)].drop(columns=['_key'])
            if extra.empty:
                return sentiment_subset
            logger.info(f"Keeping {len(extra)} {option_type} trades from Phase 2 not in sentiment file (preserving all trades)")
            # Align columns: add missing columns to extra with None
            for c in sentiment_subset.columns:
                if c not in extra.columns and c not in ('market_sentiment', 'filter_status'):
                    extra[c] = None
            for c in extra.columns:
                if c not in sentiment_subset.columns:
                    sentiment_subset[c] = None
            common_cols = [c for c in sentiment_subset.columns if c in extra.columns and c not in ('market_sentiment', 'filter_status')]
            extra = extra[[c for c in common_cols if c in extra.columns]]
            sentiment_subset = sentiment_subset[[c for c in common_cols if c in sentiment_subset.columns]]
            merged = pd.concat([sentiment_subset, extra], ignore_index=True)
            return merged
        except Exception as e:
            logger.warning(f"Could not merge with original {original_file.name}: {e}, using sentiment subset only")
            return sentiment_subset.copy()
    
    # Split sentiment file into CE and PE
    ce_from_sentiment = df[df['option_type'] == 'CE'].copy()
    pe_from_sentiment = df[df['option_type'] == 'PE'].copy()
    
    # Merge with existing CE/PE files so we keep all Phase 2 trades (sentiment file may have fewer)
    ce_trades = merge_with_original(ce_file, ce_from_sentiment, 'CE')
    pe_trades = merge_with_original(pe_file, pe_from_sentiment, 'PE')
    
    # Remove sentiment-specific columns that shouldn't be in CE/PE files
    columns_to_remove = ['market_sentiment', 'filter_status']
    for col in columns_to_remove:
        if col in ce_trades.columns:
            ce_trades = ce_trades.drop(columns=[col])
        if col in pe_trades.columns:
            pe_trades = pe_trades.drop(columns=[col])
    
    # Convert high and swing_low to percentages, remove pnl and realized_pnl columns (safety check)
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
    
    # Apply conversion to both DataFrames
    ce_trades = convert_to_percentages(ce_trades)
    pe_trades = convert_to_percentages(pe_trades)
    
    # Save CE and PE files
    try:
        ce_trades.to_csv(ce_file, index=False)
        logger.info(f"Saved {len(ce_trades)} CE trades to {ce_file.name}")
    except Exception as e:
        logger.error(f"Error writing CE file: {e}")
        return False
    
    try:
        pe_trades.to_csv(pe_file, index=False)
        logger.info(f"Saved {len(pe_trades)} PE trades to {pe_file.name}")
    except Exception as e:
        logger.error(f"Error writing PE file: {e}")
        return False
    
    logger.info(f"Successfully regenerated CE/PE files from {sentiment_file.name}")
    logger.info(f"  CE: {len(ce_trades)} trades, PE: {len(pe_trades)} trades")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate CE and PE trade files from sentiment-filtered file'
    )
    parser.add_argument('sentiment_file', type=str, 
                       help='Path to sentiment-filtered trade CSV file')
    
    args = parser.parse_args()
    sentiment_file = Path(args.sentiment_file)
    
    if not sentiment_file.exists():
        logger.error(f"File not found: {sentiment_file}")
        return 1
    
    success = regenerate_ce_pe_files(sentiment_file)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
