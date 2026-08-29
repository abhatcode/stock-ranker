
MODEL_VERSION = 3

# Universe grouped by sector. Sector labels are used for sector-relative
# features and the sector-neutral target, which keep the model unbiased
# across categories of stock.
SECTORS = {
    'semis_hardware': [
        'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'INTC', 'TXN', 'ADI', 'LRCX',
        'AMAT', 'KLAC', 'MRVL', 'ON', 'NXPI', 'AAPL', 'DELL', 'HPQ',
    ],
    'software_internet': [
        'MSFT', 'GOOG', 'META', 'AMZN', 'CRM', 'ADBE', 'ORCL', 'NOW', 'INTU',
        'PLTR', 'SNOW', 'CRWD', 'PANW', 'NET', 'DDOG', 'SHOP', 'UBER', 'ABNB',
        'NFLX', 'EA',
    ],
    'healthcare': [
        'LLY', 'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'AMGN', 'GILD', 'BMY',
        'VRTX', 'REGN', 'ISRG', 'MDT', 'CVS', 'CI', 'ZTS', 'BSX', 'SYK', 'MRNA',
    ],
    'financials': [
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'BLK',
        'SCHW', 'USB', 'PNC', 'COF', 'MET', 'AIG', 'PGR', 'CB', 'CME',
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'PSX', 'VLO',
        'KMI', 'WMB', 'HAL', 'DVN', 'FANG',
    ],
    'materials': [
        'LIN', 'FCX', 'NEM', 'APD', 'SHW', 'DOW', 'DD', 'NUE', 'ECL',
    ],
    'consumer_staples': [
        'WMT', 'COST', 'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS',
        'MDLZ', 'KR', 'SYY', 'STZ',
    ],
    'consumer_discretionary': [
        'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR',
        'ROST', 'YUM', 'CMG', 'DPZ', 'RCL', 'DIS', 'TSLA', 'F', 'GM',
    ],
    'industrials_defense': [
        'CAT', 'DE', 'BA', 'LMT', 'RTX', 'GE', 'HON', 'UPS', 'UNP', 'NOC',
        'GD', 'EMR', 'ETN', 'ITW', 'CSX', 'FDX', 'MMM', 'PCAR', 'CARR',
    ],
    'utilities_reits_telecom': [
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'T', 'VZ', 'TMUS',
        'AMT', 'PLD', 'CCI', 'EQIX', 'O', 'SPG', 'PSA',
    ],
}

STOCKS = [s for sector in SECTORS.values() for s in sector]
STOCK_SECTOR = {s: name for name, sector in SECTORS.items() for s in sector}

BENCHMARK = 'SPY'

# Forward-return horizon in TRADING days (2 calendar weeks).
HORIZON_DAYS = 10

# How much history to download for training (calendar days, ~8 years).
HISTORY_DAYS = 2920

# Purge gap between train/val/test splits, in trading days. Must be >=
# HORIZON_DAYS so overlapping forward-return windows can't leak across splits.
PURGE_DAYS = HORIZON_DAYS
