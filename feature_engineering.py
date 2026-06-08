import pandas as pd


def calculate_features(df, stock_name, spy_df=None):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['daily_return'] = df['Close'].pct_change()
    df['momentum_30'] = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)
    df['momentum_60'] = (df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60)

    df['volatility_30'] = df['daily_return'].rolling(window=30).std()
    df['volatility_60'] = df['daily_return'].rolling(window=60).std()

    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()

    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['momentum_strength'] = df['momentum_60'] / df['volatility_30']

    low_50 = df['Close'].rolling(window=50).min()
    high_50 = df['Close'].rolling(window=50).max()
    df['price_position'] = (df['Close'] - low_50) / (high_50 - low_50)

    df['roc_acceleration'] = df['momentum_60'] - df['momentum_30']
    df['volume_confirmation'] = df['daily_return'] * df['volume_ratio']

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26

    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = df['sma_20'] + (bb_std * 2)
    bb_lower = df['sma_20'] - (bb_std * 2)
    df['bollinger_pos'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    df['price_accel_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['volatility_trend'] = df['volatility_30'] - df['volatility_60']

    if spy_df is not None:
        spy_close = spy_df['Close']
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_returns = spy_close.pct_change()
        df = df.join(spy_returns.rename('spy_return'))
        df['relative_strength'] = df['daily_return'] - df['spy_return']

    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['price_vs_ema_20'] = (df['Close'] - df['ema_20']) / df['ema_20']
    df['range_50'] = (high_50 - low_50) / df['Close']

    df['returns_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['returns_30d'] = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)
    df['volume_trend_30'] = df['Volume'] / df['Volume'].rolling(window=30).mean()
    df['price_vs_sma_200'] = (df['Close'] - df['sma_200']) / df['sma_200']
    df['sma_20_vs_50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

    df['target'] = (df['Close'].shift(-30) - df['Close']) / df['Close']
    df['Date'] = df.index
    df['stock'] = stock_name

    feature_cols = [
        'sma_20', 'sma_50', 'sma_200', 'momentum_60', 'volatility_30', 'volume_ratio', 'rsi',
        'momentum_strength', 'price_position', 'roc_acceleration',
        'volume_confirmation', 'macd', 'bollinger_pos', 'price_accel_5d', 'volatility_trend',
        'relative_strength', 'price_vs_ema_20', 'range_50',
        'returns_10d', 'returns_30d', 'volume_trend_30', 'price_vs_sma_200', 'sma_20_vs_50',
        'target', 'Date', 'stock'
    ]

    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    df = df[existing_feature_cols].copy()
    return df.dropna()