MODEL_VERSION = 5

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

SECTOR_SPECIFIC_FEATURES = {
    'rate_beta_63': ('TLT', ['financials', 'utilities_reits_telecom']),
    'oil_beta_63': ('USO', ['energy']),
}
MACRO_TICKERS = sorted({t for t, _ in SECTOR_SPECIFIC_FEATURES.values()})

# INTERACTION_MODE: 'sector' or 'none'. TARGET_MODE: 'universe', 'sector_rank',
# 'blend', or 'pairwise_sector'. MODEL_ARCH: 'single' or 'moe'. See README.
INTERACTION_MODE = 'sector'
TARGET_MODE = 'pairwise_sector'
MODEL_ARCH = 'single'

HORIZON_DAYS = 10
HISTORY_DAYS = 2920
PURGE_DAYS = HORIZON_DAYS
