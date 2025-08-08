import os
import time
import csv
import logging
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from typing import Dict, List, Optional, Union, Tuple, Any
from requests.exceptions import Timeout, ConnectionError

# -----------------------------
# Configuration (validated)
# -----------------------------
class Config:
    # API Configuration (REPLACE with your real keys or load from env)
    API_KEY: str = os.getenv('BYBIT_API_KEY', "")
    API_SECRET: str = os.getenv('BYBIT_API_SECRET', "")

    # Trading Parameters
    TRADING_PAIRS: List[str] = ["PUMPUSDT"]  # default pair(s)
    TIME_FRAME: int = 15  # Candle timeframe in minutes
    TRADE_SIZE_PERCENT: float = 50.0  # Percentage of balance to use per trade
    MAX_TRADES_PER_DAY: int = 20  # Maximum trades per pair per day
    MIN_TRADE_USDT: float = 1.0  # Minimum USDT required to trade
    MAX_POSITION_SIZE: float = 1000.0  # Maximum position size in USDT

    # Range Filter Parameters
    RANGE_FILTER_PERIOD: int = 500
    RANGE_FILTER_MULTIPLIER: float = 2.0

    # Margin Trading Settings
    USE_SPOT_MARGIN: bool = True
    LEVERAGE: int = 3
    MARGIN_BUFFER: float = 0.05

    # Order Settings
    USE_MARKET_ORDERS: bool = True
    ORDER_PRICE_OFFSET: float = 0.001

    # API Settings
    RECV_WINDOW: int = 10000
    MAX_RETRIES: int = 5
    API_RETRY_DELAY: float = 5.0

    # Logging
    LOG_FILE: str = "trading_bot.log"
    CSV_LOG_FILE: str = "trades_history.csv"

    # Caching TTLs (seconds)
    SYMBOL_INFO_TTL: int = 60 * 60  # 1 hour
    KLINES_TTL: int = 60  # 1 minute
    BALANCE_TTL: int = 5  # 5 seconds

    # Defaults
    DEFAULT_MIN_ORDER: float = 1.0
    DEFAULT_PRICE_PRECISION: int = 4
    DEFAULT_QTY_PRECISION: int = 0

    # Signal validation thresholds
    MIN_VOLATILITY: float = 0.002  # 0.2% baseline volatility


# -----------------------------
# Logging setup
# -----------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.Formatter.converter = time.gmtime


setup_logging()


# -----------------------------
# Utility types and small helpers
# -----------------------------
CacheEntry = Dict[str, Any]


def utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


# -----------------------------
# BybitTradingBot class
# -----------------------------
class BybitTradingBot:
    def __init__(self):
        # Validate config
        if not Config.API_KEY or not Config.API_SECRET:
            logging.error("API credentials not configured!")
            raise ValueError("API credentials required")

        # Pybit session
        self.session = HTTP(api_key=Config.API_KEY, api_secret=Config.API_SECRET, recv_window=Config.RECV_WINDOW)

        # Dynamic internal state (do NOT mutate Config.TRADING_PAIRS directly; use set_trading_pairs)
        self.trading_pairs: List[str] = list(Config.TRADING_PAIRS)
        self.trade_counts: Dict[str, int] = {p: 0 for p in self.trading_pairs}
        self.last_trade_day: int = datetime.now(timezone.utc).day

        # Caches
        self._symbol_info_cache: Dict[str, CacheEntry] = {}
        self._klines_cache: Dict[Tuple[str, str, int], CacheEntry] = {}
        self._balance_cache: CacheEntry = {}

        # Runtime state
        self.time_diff: int = 0
        self.current_positions: Dict[str, float] = {p: 0.0 for p in self.trading_pairs}
        self.open_orders: Dict[str, Any] = {}

        # Init
        self._initialize_log_file()
        self._sync_server_time_ultra()
        self.load_symbol_info(force_update=True)
        self.check_account_type()
        self.update_positions()
        self.update_open_orders()
        logging.info("BybitTradingBot initialized")

    # -----------------------------
    # --- TIME & NETWORK helpers
    # -----------------------------
    def _sync_server_time_ultra(self) -> None:
        try:
            self._sync_server_time_with_retry()
        except Exception as e:
            logging.warning(f"Time sync failed: {e}")
            self.time_diff = 0

    def _sync_server_time_with_retry(self, max_attempts: int = 3) -> bool:
        endpoints = [
            ("https://api.bybit.com/v5/market/time", "result.timeSecond"),
            ("https://api.bybit.com/v3/public/time", "result.timeNow"),
            ("https://api.bybit.com/v2/public/time", "time_now"),
        ]
        for attempt in range(1, max_attempts + 1):
            for url, field in endpoints:
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code != 200:
                        continue
                    j = r.json()
                    # drill down
                    keys = field.split('.')
                    v = j
                    for k in keys:
                        v = v.get(k, {}) if isinstance(v, dict) else None
                    server_time = int(float(v or 0))
                    local_time = utc_now_ts()
                    self.time_diff = server_time - local_time
                    logging.info(f"Time synced from {url} (diff {self.time_diff}s)")
                    return True
                except Exception:
                    continue
            time.sleep(2 ** attempt)
        return False

    def check_network_connection(self) -> bool:
        try:
            r = requests.get("https://api.bybit.com", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # -----------------------------
    # --- API request wrapper with retries
    # -----------------------------
    def api_request_with_retry(self, func, *args, **kwargs):
        last_exc = None
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                if abs(self.time_diff) > 30 and 'recv_window' not in kwargs:
                    kwargs['recv_window'] = Config.RECV_WINDOW + abs(self.time_diff) * 1000
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = 30
                resp = func(*args, **kwargs)
                # pybit may return a dict or object; check for typical fields
                if isinstance(resp, dict) and resp.get('retCode') == 10002:
                    logging.warning("Timestamp mismatch detected; re-syncing time")
                    self._sync_server_time_ultra()
                    time.sleep(Config.API_RETRY_DELAY)
                    continue
                return resp
            except Timeout:
                logging.warning(f"API timeout attempt {attempt}")
                last_exc = 'Timeout'
            except ConnectionError:
                logging.warning(f"Connection error attempt {attempt}")
                last_exc = 'ConnectionError'
            except Exception as e:
                logging.error(f"API exception: {e}")
                last_exc = str(e)
            time.sleep(Config.API_RETRY_DELAY * attempt)
        logging.error(f"API failed after retries: {last_exc}")
        return None

    # -----------------------------
    # --- Caching helpers
    # -----------------------------
    def _cache_get(self, cache: Dict, key, ttl: int):
        entry = cache.get(key)
        if not entry:
            return None
        if utc_now_ts() - entry['ts'] > ttl:
            cache.pop(key, None)
            return None
        return entry['value']

    def _cache_set(self, cache: Dict, key, value):
        cache[key] = {'ts': utc_now_ts(), 'value': value}

    # -----------------------------
    # --- Symbol info (cached)
    # -----------------------------
    def _initialize_symbol_info(self) -> Dict[str, Dict]:
        # lightweight default
        return {p: {
            'min_order_qty': Config.DEFAULT_MIN_ORDER,
            'price_precision': Config.DEFAULT_PRICE_PRECISION,
            'qty_precision': Config.DEFAULT_QTY_PRECISION,
            'min_order_value': 10.0,
            'last_updated': 0
        } for p in self.trading_pairs}

    def load_symbol_info(self, force_update: bool = False) -> None:
        key = 'symbol_info'
        cached = self._cache_get(self._symbol_info_cache, key, Config.SYMBOL_INFO_TTL)
        if cached and not force_update:
            self.symbol_info = cached
            return

        # Build fresh
        symbol_info: Dict[str, Dict] = self._initialize_symbol_info()
        resp = self.api_request_with_retry(self.session.get_instruments_info, category='spot')
        if resp and isinstance(resp, dict) and resp.get('retCode') == 0:
            for instrument in resp['result'].get('list', []):
                symbol = instrument.get('symbol')
                if symbol in self.trading_pairs:
                    lot = instrument.get('lotSizeFilter', {})
                    symbol_info[symbol] = {
                        'min_order_qty': float(lot.get('minOrderQty', Config.DEFAULT_MIN_ORDER)),
                        'price_precision': int(instrument.get('priceScale', Config.DEFAULT_PRICE_PRECISION)),
                        'qty_precision': int(instrument.get('lotSizeFilter', {}).get('basePrecision', Config.DEFAULT_QTY_PRECISION)),
                        'min_order_value': float(lot.get('minOrderAmt', 10)),
                        'last_updated': utc_now_ts()
                    }
        else:
            logging.warning("Could not load instrument list; using defaults for symbol info")

        self.symbol_info = symbol_info
        self._cache_set(self._symbol_info_cache, key, symbol_info)
        logging.info("Symbol info loaded / cached")

    # -----------------------------
    # --- Balance (cached)
    # -----------------------------
    def get_spot_balance(self, coin: str = "USDT") -> float:
        key = f"balance_{coin}"
        cached = self._cache_get(self._balance_cache, key, Config.BALANCE_TTL)
        if cached is not None:
            return cached

        resp = self.api_request_with_retry(self.session.get_wallet_balance, accountType="UNIFIED", coin=coin)
        if not resp or not isinstance(resp, dict) or resp.get('retCode') != 0:
            logging.error("Failed to fetch balance")
            return 0.0

        balance = 0.0
        for account in resp['result'].get('list', []):
            for c in account.get('coin', []):
                if c.get('coin') == coin:
                    balance = float(c.get('availableToWithdraw') or c.get('free') or c.get('walletBalance') or 0)
                    break
        self._cache_set(self._balance_cache, key, balance)
        logging.info(f"Balance {coin}: {balance:.6f}")
        return max(balance, 0.0)

    def get_pump_balance(self) -> float:
        return self.get_spot_balance('PUMP')

    # -----------------------------
    # --- Klines (cached, optimized)
    # -----------------------------
    def get_klines(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        cache_key = (symbol, interval, limit)
        cached = self._cache_get(self._klines_cache, cache_key, Config.KLINES_TTL)
        if cached is not None:
            return cached.copy()

        resp = self.api_request_with_retry(self.session.get_kline, category='spot', symbol=symbol, interval=interval, limit=limit)
        if not resp or not isinstance(resp, dict) or resp.get('retCode') != 0:
            logging.error(f"Kline fetch failed for {symbol}")
            return None

        data = resp['result'].get('list', [])
        if not data:
            logging.error(f"No klines for {symbol}")
            return None

        df = pd.DataFrame(data)
        # ensure columns and numeric conversion
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = df.rename(columns={c: c for c in df.columns if c in expected_cols})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        if df.empty or len(df) < Config.RANGE_FILTER_PERIOD:
            logging.error(f"Insufficient candles for {symbol}: {len(df)}")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        self._cache_set(self._klines_cache, cache_key, df.copy())
        return df

    # -----------------------------
    # --- Positions & orders
    # -----------------------------
    def update_positions(self) -> None:
        for pair in self.trading_pairs:
            base = pair.replace('USDT', '')
            try:
                bal = self.get_spot_balance(base)
                self.current_positions[pair] = bal
            except Exception as e:
                logging.error(f"Position update error for {pair}: {e}")
                self.current_positions[pair] = 0.0

    def update_open_orders(self) -> None:
        resp = self.api_request_with_retry(self.session.get_open_orders, category='spot')
        if resp and isinstance(resp, dict) and resp.get('retCode') == 0:
            orders = resp['result'].get('list', [])
            self.open_orders = {o['symbol']: o for o in orders}
        else:
            self.open_orders = {}

    # -----------------------------
    # --- Core helpers: pricing, volatility, sizing
    # -----------------------------
    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            resp = self.api_request_with_retry(self.session.get_tickers, category='spot', symbol=symbol)
            if not resp or resp.get('retCode') != 0:
                return None
            result = resp.get('result')
            if isinstance(result, list) and result:
                return float(result[0].get('lastPrice') or result[0].get('close') or 0)
            if isinstance(result, dict):
                return float(result.get('lastPrice') or result.get('close') or 0)
        except Exception as e:
            logging.error(f"Price fetch error for {symbol}: {e}")
        return None

    def _get_price_volatility(self, symbol: str, lookback: int = 20) -> float:
        df = self.get_klines(symbol, str(Config.TIME_FRAME), limit=lookback)
        if df is None or len(df) < 3:
            return 0.0
        returns = df['close'].pct_change().dropna()
        return float(returns.std() * 2)

    def _calculate_order_quantity(self, symbol: str, side: str, usdt_amount: float, price: float) -> Tuple[float, float]:
        if price <= 0 or usdt_amount <= 0:
            return 0.0, 0.0
        symbol_data = self.symbol_info.get(symbol, {})
        min_order = float(symbol_data.get('min_order_qty', Config.DEFAULT_MIN_ORDER))
        min_order_value = float(symbol_data.get('min_order_value', 10.0))
        qty_precision = int(symbol_data.get('qty_precision', Config.DEFAULT_QTY_PRECISION))

        if Config.USE_SPOT_MARGIN and side.lower() == 'buy':
            usdt_amount = usdt_amount * Config.LEVERAGE

        quantity = usdt_amount / price

        if side.lower() == 'sell':
            available = self.current_positions.get(symbol, 0.0)
            if available <= 0:
                return 0.0, 0.0
            quantity = min(quantity, available)

        quantity = round(quantity, qty_precision)
        if quantity < min_order:
            return 0.0, 0.0

        order_value = price * quantity
        if order_value < min_order_value:
            return 0.0, 0.0

        return quantity, order_value

    # -----------------------------
    # --- Order placement, logging and safety
    # -----------------------------
    def _log_order_execution(self, response: Dict, symbol: str, side: str) -> None:
        try:
            filled_qty = response.get('result', {}).get('cumExecQty') or response.get('result', {}).get('filledQty') or 0
            price = response.get('result', {}).get('price') or 0
            order_id = response.get('result', {}).get('orderId') or response.get('result', {}).get('order_id') or ''
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            with open(Config.CSV_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([ts, symbol, side, price, filled_qty, 'executed', self.get_spot_balance('USDT'), 0.0, price * float(filled_qty or 0), order_id])
        except Exception as e:
            logging.error(f"Failed to log order: {e}")

    def place_order(self, symbol: str, side: str, usdt_amount: Optional[float] = None) -> bool:
        try:
            # daily trade limit
            if self.trade_counts.get(symbol, 0) >= Config.MAX_TRADES_PER_DAY:
                logging.warning(f"Max trades reached for {symbol}")
                return False

            price = self._get_current_price(symbol)
            if not price or price <= 0:
                logging.error("Invalid price")
                return False

            if usdt_amount is None:
                usdt_balance = self.get_spot_balance('USDT')
                effective_pct = Config.TRADE_SIZE_PERCENT * 0.8
                usdt_amount = min(usdt_balance * (effective_pct / 100.0), Config.MAX_POSITION_SIZE)

            quantity, order_value = self._calculate_order_quantity(symbol, side, usdt_amount, price)
            if quantity <= 0 or order_value <= 0:
                logging.warning("Quantity calculation failed or below minimums")
                return False

            if Config.USE_SPOT_MARGIN and side.lower() == 'buy':
                available = self.get_spot_balance('USDT')
                required_margin = (order_value / Config.LEVERAGE) * (1 + Config.MARGIN_BUFFER)
                if required_margin > available:
                    logging.warning("Insufficient margin for buy")
                    return False

            if symbol in self.open_orders:
                logging.warning(f"Existing open order for {symbol}")
                return False

            params = {
                'category': 'spot',
                'symbol': symbol,
                'side': side,
                'qty': str(quantity),
                'timeInForce': 'GTC',
                'orderType': 'Market' if Config.USE_MARKET_ORDERS else 'Limit'
            }
            if not Config.USE_MARKET_ORDERS:
                order_price = price * (1 - Config.ORDER_PRICE_OFFSET) if side.lower() == 'buy' else price * (1 + Config.ORDER_PRICE_OFFSET)
                params['price'] = str(round(order_price, self.symbol_info.get(symbol, {}).get('price_precision', Config.DEFAULT_PRICE_PRECISION)))

            logging.info(f"Placing order {side} {quantity} {symbol} (~{order_value:.2f} USDT)")
            resp = self.api_request_with_retry(self.session.place_order, **params)
            if not resp or (isinstance(resp, dict) and resp.get('retCode') != 0):
                logging.error(f"Order failed: {resp}")
                return False

            self._log_order_execution(resp, symbol, side)
            self.trade_counts[symbol] = self.trade_counts.get(symbol, 0) + 1
            # refresh positions and open orders
            self.update_positions()
            self.update_open_orders()
            return True

        except Exception as e:
            logging.error(f"place_order exception: {e}")
            return False

    # -----------------------------
    # --- Dynamic pair switching
    # -----------------------------
    def set_trading_pairs(self, new_pairs: List[str], auto_close_positions: bool = True) -> bool:
        # sanitize pairs
        valid = [p for p in new_pairs if isinstance(p, str) and p.upper().endswith('USDT')]
        valid = [p.upper() for p in valid]
        if not valid:
            logging.error("No valid trading pairs provided to set_trading_pairs")
            return False

        to_remove = set(self.trading_pairs) - set(valid)
        to_add = set(valid) - set(self.trading_pairs)

        if auto_close_positions and to_remove:
            for pair in list(to_remove):
                pos = self.current_positions.get(pair, 0.0)
                if pos > 0:
                    logging.info(f"Auto-closing position for {pair} before removing")
                    price = self._get_current_price(pair)
                    if price:
                        trade_val = pos * price
                        self.place_order(pair, 'Sell', trade_val)

        # update internal list and dependent structures
        self.trading_pairs = valid
        self.trade_counts = {p: self.trade_counts.get(p, 0) for p in self.trading_pairs}
        # ensure every pair in symbol info
        self.load_symbol_info(force_update=True)
        self.update_positions()
        self.update_open_orders()
        logging.info(f"Trading pairs updated to: {self.trading_pairs}")
        return True

    # -----------------------------
    # --- Signal calculation / Range Filter
    # -----------------------------
    def calculate_range_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            period = Config.RANGE_FILTER_PERIOD
            multiplier = Config.RANGE_FILTER_MULTIPLIER

            # vectorized: absolute diff and EWM
            df = df.copy()
            df['abs_diff'] = (df['close'] - df['close'].shift(1)).abs()
            df['avrng'] = df['abs_diff'].ewm(span=period, adjust=False).mean()
            wper = period * 2 - 1
            df['smrng'] = df['avrng'].ewm(span=wper, adjust=False).mean() * multiplier

            # iterative filter still required to maintain state per row
            filt = df['close'].copy()
            for i in range(1, len(df)):
                prev = filt.iat[i - 1]
                cur_close = df['close'].iat[i]
                cur_smrng = df['smrng'].iat[i]
                if cur_close > prev:
                    filt.iat[i] = max(prev, cur_close - cur_smrng)
                else:
                    filt.iat[i] = min(prev, cur_close + cur_smrng)
            df['filt'] = filt

            # direction counters - vectorized approach with fillna
            df['filt_shift'] = df['filt'].shift(1)
            df['upward'] = (df['filt'] > df['filt_shift']).astype(int)
            df['downward'] = (df['filt'] < df['filt_shift']).astype(int)

            df['hband'] = df['filt'] + df['smrng']
            df['lband'] = df['filt'] - df['smrng']

            df.drop(columns=['filt_shift'], inplace=True)
            return df
        except Exception as e:
            logging.error(f"Range Filter error: {e}")
            return pd.DataFrame()

    def generate_range_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df['above_filter'] = df['close'] > df['filt']
        df['below_filter'] = df['close'] < df['filt']
        df['cross_above'] = df['above_filter'] & (~df['above_filter'].shift(1).fillna(False))
        df['cross_below'] = df['below_filter'] & (~df['below_filter'].shift(1).fillna(False))

        # initialize signals
        df['signal'] = 0
        prev_signal = 0
        for i in range(1, len(df)):
            buy_cond = (
                df['above_filter'].iat[i] and
                (df['cross_above'].iat[i] or (df['close'].iat[i] < df['close'].iat[i - 1] and df['close'].iat[i] <= df['filt'].iat[i] * 1.005))
            )
            sell_cond = (
                df['below_filter'].iat[i] and
                (df['cross_below'].iat[i] or (df['close'].iat[i] > df['close'].iat[i - 1] and df['close'].iat[i] >= df['filt'].iat[i] * 0.995))
            )
            if buy_cond and prev_signal != 1:
                df['signal'].iat[i] = 1
                prev_signal = 1
            elif sell_cond and prev_signal != -1:
                df['signal'].iat[i] = -1
                prev_signal = -1
            else:
                df['signal'].iat[i] = prev_signal
        df['buy_signal'] = df['signal'] == 1
        df['sell_signal'] = df['signal'] == -1
        return df

    def calculate_signals(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty or len(df) < Config.RANGE_FILTER_PERIOD:
            return None
        df = self.calculate_range_filter(df)
        if df.empty:
            return None
        df = self.generate_range_signals(df)
        return df.dropna()

    # -----------------------------
    # --- Strategy loop
    # -----------------------------
    def check_and_reset_trade_counts(self) -> None:
        now = datetime.now(timezone.utc)
        if now.day != self.last_trade_day:
            self.trade_counts = {p: 0 for p in self.trading_pairs}
            self.last_trade_day = now.day
            logging.info("Daily trade counts reset")

    def run_strategy(self) -> None:
        logging.info("Starting strategy loop")
        try:
            while True:
                if not self.check_network_connection():
                    logging.error("Network unavailable - backing off 60s")
                    time.sleep(60)
                    continue

                now_utc = datetime.now(timezone.utc)
                # periodic maintenance
                if int(now_utc.timestamp()) % (Config.TIME_FRAME * 60) == 0:
                    self._sync_server_time_ultra()
                    self.load_symbol_info(force_update=True)

                self.check_and_reset_trade_counts()
                usdt_balance = self.get_spot_balance('USDT')
                if usdt_balance < Config.MIN_TRADE_USDT:
                    logging.warning("Insufficient USDT")
                    time.sleep(60)
                    continue

                for pair in list(self.trading_pairs):
                    try:
                        # refresh local state for pair
                        self.update_positions()
                        pos_amount = self.current_positions.get(pair, 0.0)

                        df = self.get_klines(pair, str(Config.TIME_FRAME))
                        if df is None:
                            continue
                        df_sig = self.calculate_signals(df)
                        if df_sig is None:
                            continue
                        latest = df_sig.iloc[-1]

                        # validate volatility
                        vol = self._get_price_volatility(pair)
                        if vol < Config.MIN_VOLATILITY:
                            logging.info(f"Skipping {pair} due low volatility: {vol:.6f}")
                            continue

                        # position sizing scaled by volatility (more volatile => smaller position)
                        effective_pct = max(0.01, Config.TRADE_SIZE_PERCENT * (1.0 / (1.0 + vol * 100)))

                        if latest['buy_signal'] and pos_amount <= 0:
                            trade_size = min(usdt_balance * (effective_pct / 100.0), Config.MAX_POSITION_SIZE)
                            # tiny safeguard
                            if trade_size < Config.MIN_TRADE_USDT:
                                continue
                            if self.place_order(pair, 'Buy', trade_size):
                                logging.info(f"Executed BUY for {pair}")

                        elif latest['sell_signal'] and pos_amount > 0:
                            price = self._get_current_price(pair)
                            if not price:
                                continue
                            trade_value = pos_amount * price
                            if self.place_order(pair, 'Sell', trade_value):
                                logging.info(f"Executed SELL for {pair}")

                    except Exception as e:
                        logging.error(f"Processing error for {pair}: {e}")
                        continue

                # sleep to the next candle precisely
                next_candle = self._get_next_candle_time()
                sleep_seconds = (next_candle - datetime.now(timezone.utc)).total_seconds()
                if sleep_seconds > 0:
                    logging.info(f"Sleeping {sleep_seconds:.1f}s until next candle {next_candle}")
                    time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logging.info("Strategy stopped by user")

    # -----------------------------
    # --- Misc helpers
    # -----------------------------
    def _get_next_candle_time(self) -> datetime:
        now = datetime.now(timezone.utc)
        minutes = (Config.TIME_FRAME - (now.minute % Config.TIME_FRAME)) % Config.TIME_FRAME
        if minutes == 0 and now.second == 0:
            return now.replace(second=0, microsecond=0)
        next_ts = (now + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
        return next_ts

    def _initialize_log_file(self) -> None:
        try:
            if not os.path.exists(Config.CSV_LOG_FILE):
                with open(Config.CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    headers = ["Timestamp(UTC)", "Pair", "Action", "Price", "Quantity", "Reason", "Balance", "PnL", "Filled Value", "Order ID"]
                    writer.writerow(headers)
        except Exception as e:
            logging.error(f"Failed to init CSV log file: {e}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == '__main__':
    # Basic validation of pairs
    invalid = [p for p in Config.TRADING_PAIRS if not isinstance(p, str) or 'USDT' not in p]
    if invalid:
        logging.error(f"Invalid default pairs in Config: {invalid}")
        exit(1)

    bot = BybitTradingBot()
    # Example of dynamically setting pairs at runtime
    # bot.set_trading_pairs(['BTCUSDT', 'ETHUSDT'])
    bot.run_strategy()
