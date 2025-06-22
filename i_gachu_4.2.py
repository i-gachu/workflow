import os
import time
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
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
period = 60
expiration = 60
INITIAL_AMOUNT = 1
PROB_THRESHOLD = 0.60
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Trading instrument
instrument = "EUR_USD"
currency_pair = "EURUSD"
max_candles = 5000

# OANDA setup
client = oandapyV20.API(access_token=ACCESS_TOKEN)
granularity = "M1"
params = {
    "granularity": granularity,
    "count": 100,
    "price": "M"
}

# Pocket Option setup
api = PocketOption(ssid, demo)
api.connect()

FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']
model = None  # Global model object

def wait_until_next_candle(period_seconds=60, seconds_before=5):
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
        global_value.logger(f"{datetime.now()} : [ERROR]: Payout Error: {str(e)}", "ERROR")
        return False

def get_training_data():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    all_data = []

    while start_time < end_time:
        params = {
            "granularity": granularity,
            "price": "M",
            "from": start_time.isoformat(),
            "count": max_candles
        }
        r = InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response["candles"]
        if not candles:
            break
        for c in candles:
            timestamp = c["time"]
            o = float(c["mid"]["o"])
            h = float(c["mid"]["h"])
            l = float(c["mid"]["l"])
            c_ = float(c["mid"]["c"])
            all_data.append((timestamp, o, h, l, c_))
        last_time = datetime.fromisoformat(candles[-1]["time"].replace("Z", "+00:00"))
        start_time = last_time + timedelta(minutes=1)
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=["time", "open", "high", "low", "close"])
    global_value.logger(f"{datetime.now()} : [INFO]: Fetched {len(df)} training data for {instrument}", "INFO")
    return df

def get_prediction_data():
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
    global_value.logger(f"{datetime.now()} : [INFO]: Fetched {len(df)} prediction data for {instrument}", "INFO")
    return df

def prepare_data(df):
    df = df[['time', 'open', 'high', 'low', 'close']]
    df.sort_values(by='time', inplace=True)
    df['change_in_price'] = df['close'].diff()

    rsi_period = 14
    stochastic_period = 14
    macd_ema_long = 26
    macd_ema_short = 12
    macd_signal = 9
    roc_period = 9

    up_df = df['change_in_price'].where(df['change_in_price'] > 0, 0)
    down_df = abs(df['change_in_price'].where(df['change_in_price'] < 0, 0))
    ewma_up = up_df.ewm(span=rsi_period).mean()
    ewma_down = down_df.ewm(span=rsi_period).mean()
    rs = ewma_up / ewma_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    df['low_14'] = df['low'].rolling(window=stochastic_period).min()
    df['high_14'] = df['high'].rolling(window=stochastic_period).max()
    df['k_percent'] = 100 * ((df['close'] - df['low_14']) / (df['high_14'] - df['low_14']))
    df['r_percent'] = ((df['high_14'] - df['close']) / (df['high_14'] - df['low_14'])) * -100

    ema_26 = df['close'].ewm(span=macd_ema_long).mean()
    ema_12 = df['close'].ewm(span=macd_ema_short).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_EMA'] = df['MACD'].ewm(span=macd_signal).mean()

    df['Price_Rate_Of_Change'] = df['close'].pct_change(periods=roc_period)
    df['Prediction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    df = prepare_data(df)
    X = df[FEATURE_COLS].iloc[:-2]
    y = df['Prediction'].iloc[:-2]
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)
    global_value.logger(f"{datetime.now()} : [INFO]: Model trained with {len(X)} data points", "INFO")
    return model

def predict_next_move(model, df):
    df = prepare_data(df)
    X_test = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X_test)[0]
    call_conf = proba[1]
    put_conf = proba[0]
    global_value.logger(f"{datetime.now()} : [DEBUG]: Probabilities - CALL: {call_conf:.2f}, PUT: {put_conf:.2f}", "INFO")
    if call_conf > PROB_THRESHOLD:
        return "call"
    elif put_conf > PROB_THRESHOLD:
        return "put"
    else:
        global_value.logger(f"{datetime.now()} : [INFO]: ⏭️ Skipping trade due to low confidence ({max(call_conf, put_conf):.2f})", "INFO")
        return None

def perform_trade(amount, pair, action, expiration):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]
    global_value.logger(f"{datetime.now()} : [INFO]: 🟡 Trade placed: {action.upper()} {pair} | ID {trade_id}", "INFO")
    return result

def prepare():
    try:
        payout_success = get_payout()
        return payout_success
    except Exception as e:
        global_value.logger(f"{datetime.now()} : [ERROR]: Prepare error: {e}", "ERROR")
        return False

def strategie():
    global model
    wait_until_next_candle(period_seconds=period, seconds_before=1)
    df_latest = get_prediction_data()
    decision = predict_next_move(model, df_latest)

    if decision:
        global_value.logger(f"{datetime.now()} : [INFO]: 🧠 Final Decision: {decision.upper()}", "INFO")
        wait_for_candle_start(period_seconds=period)
        perform_trade(INITIAL_AMOUNT, currency_pair, decision, expiration)
        get_payout()
    else:
        global_value.logger(f"{datetime.now()} : [INFO]: No trade executed this cycle", "INFO")
        # Prepare for next cycle: collect and train early
        df_train = get_training_data()
        if not df_train.empty:
            model = train_model(df_train)
        wait_for_candle_start(period_seconds=period)


def start():
    while not global_value.websocket_is_connected:
        time.sleep(0.1)
    time.sleep(2)
    if prepare():
        # Initial model training
        df_train = get_training_data()
        if not df_train.empty:
            global model
            model = train_model(df_train)
        while True:
            strategie()

if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    global_value.logger(f"{datetime.now()} : [INFO]: CPU-bound Task Time: {int(end_counter - start_counter)}s", "INFO")
