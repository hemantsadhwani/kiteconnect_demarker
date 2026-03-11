"""
config_resolver.py — STRIKE_MODE resolution for backtesting_config.yaml
                     and indicators_config.yaml.

Usage for backtesting_config.yaml:

    import yaml
    from config_resolver import resolve_strike_mode

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_strike_mode(config)
    # config['PATHS']['DATA_DIR'] and config['BACKTESTING_EXPIRY']['BACKTESTING_DAYS']
    # now reflect the active STRIKE_MODE (ST50 or ST100).

Usage for indicators_config.yaml:

    import yaml
    from config_resolver import resolve_indicators_config

    with open('indicators_config.yaml') as f:
        config = yaml.safe_load(f)
    config = resolve_indicators_config(config, Path(__file__).parent)
    # config['PATHS']['DATA_DIR'] and config['TARGET_EXPIRY']['TRADING_DAYS']
    # now reflect the active STRIKE_MODE read from backtesting_config.yaml.

Or, to get just the resolved data directory path:

    from config_resolver import get_data_dir
    data_dir = get_data_dir(BACKTESTING_DIR)   # returns a Path object
"""

import yaml
from pathlib import Path


def resolve_strike_mode(config: dict) -> dict:
    """
    Resolve STRIKE_MODE-specific settings into the live config dict.

    Reads STRIKE_MODE (default: ST50) and STRIKE_MODE_SETTINGS from the config,
    then overrides:
        PATHS.DATA_DIR                         <- STRIKE_MODE_SETTINGS.<mode>.DATA_DIR
        BACKTESTING_EXPIRY.BACKTESTING_DAYS    <- BACKTESTING_EXPIRY.BACKTESTING_DAYS_<mode>
        DATA_COLLECTION.STRIKE_DIFFERENCE      <- STRIKE_MODE_SETTINGS.<mode>.STRIKE_DIFFERENCE
        DYNAMIC_COLLECTION.RANGE_SIZE          <- STRIKE_MODE_SETTINGS.<mode>.RANGE_SIZE
        DYNAMIC_COLLECTION.ATM_RULE.RANGE_SIZE <- same
        DYNAMIC_COLLECTION.OTM_RULE.RANGE_SIZE <- same
    """
    if not config:
        return config

    mode = config.get('STRIKE_MODE', 'ST50')
    settings = config.get('STRIKE_MODE_SETTINGS', {}).get(mode, {})

    # DATA_DIR
    data_dir = settings.get('DATA_DIR')
    if data_dir:
        config.setdefault('PATHS', {})['DATA_DIR'] = data_dir

    # BACKTESTING_DAYS — pick the mode-specific list
    days_key = f'BACKTESTING_DAYS_{mode}'
    expiry = config.get('BACKTESTING_EXPIRY', {})
    if days_key in expiry:
        expiry['BACKTESTING_DAYS'] = expiry[days_key]

    # STRIKE_DIFFERENCE
    strike_diff = settings.get('STRIKE_DIFFERENCE')
    if strike_diff is not None:
        config.setdefault('DATA_COLLECTION', {})['STRIKE_DIFFERENCE'] = strike_diff

    # RANGE_SIZE (flat + nested ATM/OTM rules)
    range_size = settings.get('RANGE_SIZE')
    if range_size is not None:
        dyn = config.setdefault('DYNAMIC_COLLECTION', {})
        dyn['RANGE_SIZE'] = range_size
        dyn.setdefault('ATM_RULE', {})['RANGE_SIZE'] = range_size
        dyn.setdefault('OTM_RULE', {})['RANGE_SIZE'] = range_size

    return config


def resolve_indicators_config(indicators_config: dict, backtesting_dir: Path) -> dict:
    """
    Resolve STRIKE_MODE into an indicators_config dict.

    Reads STRIKE_MODE and STRIKE_MODE_SETTINGS from backtesting_config.yaml
    (located in backtesting_dir), then overrides inside indicators_config:

        PATHS.DATA_DIR                         <- STRIKE_MODE_SETTINGS.<mode>.DATA_DIR
        TARGET_EXPIRY.TRADING_DAYS             <- TARGET_EXPIRY.TRADING_DAYS_<mode>
    """
    if not indicators_config:
        return indicators_config

    # Determine active mode:
    #   1. backtesting_config.yaml STRIKE_MODE (keeps both scripts in sync)
    #   2. indicators_config.yaml STRIKE_MODE   (standalone / independent run)
    #   3. default ST50
    bt_config: dict = {}
    try:
        with open(Path(backtesting_dir) / 'backtesting_config.yaml', 'r') as f:
            bt_config = yaml.safe_load(f) or {}
        bt_config = resolve_strike_mode(bt_config)
    except Exception:
        pass

    # indicators_config.yaml STRIKE_MODE is the authority for standalone runs
    # (run_indicators.py is always run standalone).
    # Fall back to backtesting_config.yaml only if indicators_config has no STRIKE_MODE set.
    mode = (
        indicators_config.get('STRIKE_MODE')
        or bt_config.get('STRIKE_MODE')
        or 'ST50'
    )

    # DATA_DIR: prefer indicators_config's own STRIKE_MODE_SETTINGS, then
    # fall back to what backtesting_config resolved.
    ind_settings = indicators_config.get('STRIKE_MODE_SETTINGS', {}).get(mode, {})
    data_dir = (
        ind_settings.get('DATA_DIR')
        or bt_config.get('PATHS', {}).get('DATA_DIR', f'data_{mode.lower()}')
    )

    # Override DATA_DIR
    indicators_config.setdefault('PATHS', {})['DATA_DIR'] = data_dir

    # Override TRADING_DAYS from the mode-specific key
    days_key = f'TRADING_DAYS_{mode}'
    target = indicators_config.setdefault('TARGET_EXPIRY', {})
    if days_key in target:
        target['TRADING_DAYS'] = target[days_key]

    return indicators_config


def get_data_dir(backtesting_dir: Path) -> Path:
    """
    Load backtesting_config.yaml, resolve STRIKE_MODE, and return the
    active data directory as an absolute Path.

    Falls back to <backtesting_dir>/data_st50 if config cannot be loaded.
    """
    config_path = backtesting_dir / 'backtesting_config.yaml'
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        resolve_strike_mode(config)
    except Exception:
        pass
    data_dir_name = config.get('PATHS', {}).get('DATA_DIR', 'data_st50')
    return backtesting_dir / data_dir_name
