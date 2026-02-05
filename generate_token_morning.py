#!/usr/bin/env python3
"""
Morning Access Token Generator
This script generates a new Kite Connect access token first thing in the morning.
Can be scheduled to run automatically via Windows Task Scheduler or cron.

Usage:
    python generate_token_morning.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from access_token import generate_new_access_token

def main():
    """Generate a new access token and log the result."""
    # Ensure unbuffered output
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
    
    # Also log to file
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'token_generation.log'
    
    def log_print(*args, **kwargs):
        """Print to both console and log file."""
        msg = ' '.join(str(arg) for arg in args)
        print(msg, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    log_print("=" * 60)
    log_print(f"Morning Access Token Generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 60)
    
    try:
        success = generate_new_access_token()
        
        if success:
            log_print("\n[SUCCESS] New access token generated successfully!")
            log_print(f"Token saved to: {project_root / 'key_secrets' / 'access_token.txt'}")
            return 0
        else:
            log_print("\n[FAILED] Could not generate new access token.")
            return 1
            
    except Exception as e:
        log_print(f"\n[ERROR] Exception occurred during token generation: {e}")
        import traceback
        error_trace = traceback.format_exc()
        log_print(error_trace)
        print(error_trace)  # Also print to console
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

