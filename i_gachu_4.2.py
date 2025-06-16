import os
import time
import json
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
from sklearn.ensemble import RandomForestClassifier
from oandapyV20.endpoints.instruments import InstrumentsCandles
import oandapyV20

# Load environment variables
load_dotenv()

# Session config
start_counter = time.perf_counter()

ssid = os.getenv("SSID")
demo = True

# Bot Settings
min_payout = 10
period = 60           # 1-minute candles
expiration = 60       # 60s expiry
INITIAL_AMOUNT = 1
PROB_THRESHOLD = 0.76
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Trading instrument
instrument = "EUR_USD"     # OANDA symbol
currency_pair = "EURUSD"   # Pocket Option symbol (verify with get_payout)

# OANDA setup
client = oandapyV20.API(access_token=ACCESS_TOKEN)
granularity = "M1"
params = {
    "granularity": granularity,
    "count": 5000,
    "price": "M"
}

# Pocket Option setup
api = PocketOption(ssid, demo)
api.connect()

FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']


def wait_until_next_candle(period_seconds=60, seconds_before=10):
    while True:
        now = datetime.now(timezone.utc)
        now_seconds = now.hour * 3600 + now.minute * 60 + now.second
        next_candle = ((now_seconds // period_seconds) + 1) * period_seconds
        if now_seconds >= next_candle - seconds_before:
            break
        time.sleep(0.2)


def wait_for_candle_start(period_seconds=60):
    while True:
        now = datetime.now(timezone.utc)
        now_seconds = now.hour * 3600 + now.minute * 60 + now.second
        if now_seconds % period_seconds == 0:
            break
        time.sleep(0.1)


def get_payout():
    try:
        d = json.loads(global_value.PayoutData)
        for pair in d:
            if pair[1] == currency_pair and pair[14] is True:
                p = {'payout': pair[5], 'type': pair[3]}
                global_value.pairs[pair[1]] = p
        return True
    except Exception as e:
        global_value.logger(f"Payout Error: {str(e)}", "ERROR")
        return False


def get_df():
    r = InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)
    candles = r.response['candles']
    data = []
    for c in candles:
        time_ = c['time']
        o = float(c['mid']['o'])
        h = float(c['mid']['h'])
        l = float(c['mid']['l'])
        c_ = float(c['mid']['c'])
        data.append((time_, o, h, l, c_))
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
    filename = f"{instrument}_{granularity}.csv"
    df.to_csv(filename, index=False)
    global_value.logger(f"Fetched {len(df)} candles for {instrument}", "INFO")
    return df


def prepare_data(df):
    df = df[['time', 'open', 'high', 'low', 'close']]
    df.sort_values(by='time', inplace=True)
    df['change_in_price'] = df['close'].diff()

    # Indicator parameters
    rsi_period = 14
    stochastic_period = 14
    macd_ema_long = 26
    macd_ema_short = 12
    macd_signal = 9
    roc_period = 9

    # RSI
    up_df = df['change_in_price'].where(df['change_in_price'] > 0, 0)
    down_df = abs(df['change_in_price'].where(df['change_in_price'] < 0, 0))
    ewma_up = up_df.ewm(span=rsi_period).mean()
    ewma_down = down_df.ewm(span=rsi_period).mean()
    rs = ewma_up / ewma_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # Stochastic
    df['low_14'] = df['low'].rolling(window=stochastic_period).min()
    df['high_14'] = df['high'].rolling(window=stochastic_period).max()
    df['k_percent'] = 100 * ((df['close'] - df['low_14']) / (df['high_14'] - df['low_14']))
    df['r_percent'] = ((df['high_14'] - df['close']) / (df['high_14'] - df['low_14'])) * -100

    # MACD
    ema_26 = df['close'].ewm(span=macd_ema_long).mean()
    ema_12 = df['close'].ewm(span=macd_ema_short).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_EMA'] = df['MACD'].ewm(span=macd_signal).mean()

    # ROC
    df['Price_Rate_Of_Change'] = df['close'].pct_change(periods=roc_period)

    # Prediction target: will next candle be bullish?
    df['Prediction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df


def train_and_predict(df):
    X_train = df[FEATURE_COLS].iloc[:-1]
    y_train = df['Prediction'].iloc[:-1]
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    X_test = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X_test)[0]
    call_conf = proba[1]
    put_conf = proba[0]

    if call_conf > PROB_THRESHOLD:
        return 'call', call_conf
    else:
        return 'put', put_conf


def perform_trade(amount, pair, action, expiration):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]
    global_value.logger(f"🟡 Trade placed: {action.upper()} {pair} | ID {trade_id}", "INFO")
    return result


def strategie():
    wait_until_next_candle(period_seconds=period, seconds_before=10)

    for pair in list(global_value.pairs.keys()):
        payout = global_value.pairs[pair].get('payout', 0)
        if payout < min_payout:
            continue

        df = get_df()
        global_value.logger(f"Collected {len(df)} candles for {pair}", "INFO")
        processed_df = prepare_data(df.copy())
        if processed_df.empty:
            continue

        decision, confidence = train_and_predict(processed_df)
        if not decision:
            continue

        global_value.logger(f"🧠 Predicted: {decision.upper()} | Confidence: {confidence:.2f}", "INFO")
        wait_for_candle_start(period_seconds=period)
        perform_trade(INITIAL_AMOUNT, pair, decision, expiration)

        get_payout()
        get_df()


def prepare():
    try:
        payout_success = get_payout()
        df = get_df()
        return payout_success and not df.empty
    except Exception as e:
        global_value.logger(f"Prepare error: {e}", "ERROR")
        return False


def start():
    while not global_value.websocket_is_connected:
        time.sleep(0.1)
    time.sleep(2)
    if prepare():
        while True:
            strategie()


if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    global_value.logger(f"CPU-bound Task Time: {int(end_counter - start_counter)}s", "INFO")
