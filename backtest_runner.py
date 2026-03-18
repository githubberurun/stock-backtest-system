import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def is_recently_updated(filepath: str, hours: int = 12) -> bool:
    """キャッシュファイルの有効期限を判定"""
    if not isinstance(filepath, str):
        return False
    if not os.path.exists(filepath):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    return (datetime.now() - file_mtime) < timedelta(hours=hours)

# ==========================================
# 1. 大型株専用・統合分析エンジン
# ==========================================
class AdvancedStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        df['prev_close'] = df['close'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['prev_low'] = df['low'].shift(1)
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        df['is_bullish'] = df['close'] > df['open']
        
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        df['macd_hist'] = df['macd'] - df['sig']
        df['macd_hist_slope'] = df['macd_hist'].diff()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['tr'] = tr
        df['atr'] = df['tr'].rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
            df['rs'] = df['rs_21']
        else:
            df['rs_21'] = 0.0
            df['rs'] = 0.0
            df['close_bm'] = 0.0
            df['bm_ma200'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        
        # 【完全復元】最高益(927%)を叩き出したアグレッシブなエントリー条件
        if n_chg <= -2.5 or vix >= 33.0:
            return False, 0.0, False
            
        tr_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('tr', 0.0))
        atr_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        if atr_val > 0 and (tr_val / atr_val) >= 2.5:
            return False, 0.0, False
        
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        ma75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75', 0.0))
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0), 50.0)
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0), 0.0)
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0), 1.0)
        macd_hist_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('macd_hist_slope', 0.0))
        
        is_bear_market = (bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200)
        trend_penalty = 20.0 if curr_price > 0 and ma75 > 0 and curr_price < ma75 else 0.0

        if is_bear_market:
            if rsi_val > 30.0 or vol_ratio < 1.5:
                return False, 0.0, is_bear_market
            total_score = 90.0 - trend_penalty
        else:
            if rs_21_val < 0.0:
                return False, 0.0, is_bear_market
            main_score = 30.0
            if vol_ratio >= 1.5: main_score += 20
            if 50 <= rsi_val <= 75: main_score += 15
            elif rsi_val < 40: main_score += 20
            
            if macd_hist_slope > 0: main_score += 15
            total_score = main_score + 30.0 - trend_penalty
            
        is_entry = (total_score >= 80)
        return is_entry, float(total_score), is_bear_market

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float, is_bear_market: bool, vix: float) -> float:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        
        # 【完全復元】元の指値距離（深い指値すぎると約定しなかったため）
        if is_bear_market or vix >= 20.0:
            base_offset = 1.2
        else:
            base_offset = 0.1
            
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -1.0 else 0.0
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場データキャッシュ & ポートフォリオバックテスター
# ==========================================
class USMarketCache:
    def __init__(self, data_dir: str) -> None:
        debug_log("Initializing US market data cache...")
        self.cache_file = os.path.join(data_dir, "us_market_cache.parquet")
        
        # 【修正】yfinanceデータの完全キャッシュ化（API無駄打ち排除）
        if is_recently_updated(self.cache_file, hours=12):
            debug_log("Loading US market data from local cache (SKIPPED download)...")
            try:
                df = pd.read_parquet(self.cache_file)
                self.ndx = df['ndx']
                self.vix = df['vix']
                return
            except Exception as e:
                debug_log(f"Cache read failed: {e}. Falling back to download.")
                
        debug_log("Fetching US market data from yfinance...")
        try:
            ndx_data = yf.Ticker("^IXIC").history(period="10y")
            vix_data = yf.Ticker("^VIX").history(period="10y")
            if not ndx_data.empty and not vix_data.empty:
                ndx_series = ndx_data['Close'].pct_change() * 100
                vix_series = vix_data['Close']
                df = pd.DataFrame({'ndx': ndx_series, 'vix': vix_series})
                df.index = df.index.tz_localize(None).strftime('%Y-%m-%d')
                df.to_parquet(self.cache_file)
                self.ndx = df['ndx']
                self.vix = df['vix']
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as e:
            debug_log(f"Failed to fetch US market data: {e}")
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if not isinstance(date_str, str): raise TypeError("date_str must be string")
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.vix.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 10) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.max_positions: int = max_positions
        self.attr: str = "スイング" 
        self.us_market = USMarketCache(data_dir)
        
        self.stats: Dict[str, int] = {
            'limit_placed': 0, 'limit_exec': 0,
            'time_stops': 0, 'time_stop_wins': 0,
            'hard_stops': 0, 'trailing_stops': 0,
            'gap_down_cancels': 0,
            'circuit_breaker_hits': 0  # 【新規】サーキットブレーカー発動回数
        }
        
        cache_dir = f"{data_dir}_cache"
        os.makedirs(cache_dir, exist_ok=True)
        debug_log(f"Using cache directory: {cache_dir}")
        
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        dates_set: Set[str] = set()
        
        bm_path = f"{data_dir}/13060.parquet"
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"])
        debug_log(f"Loading {len(files)} tickers. Checking cache...")
        
        for file in files:
            ticker = file.replace(".parquet", "")
            raw_path = f"{data_dir}/{file}"
            cache_path = f"{cache_dir}/{file}"
            
            try:
                raw_mtime = os.path.getmtime(raw_path)
                if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= raw_mtime:
                    df = pd.read_parquet(cache_path)
                else:
                    df = pd.read_parquet(raw_path)
                    df = AdvancedStrategyAnalyzer.calculate_indicators(df, bm_df)
                    if not df.empty:
                        df.to_parquet(cache_path)
            except Exception as e:
                debug_log(f"Error processing {ticker}: {e}")
                continue
                
            if df.empty: continue
            
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            records = df.to_dict(orient='records')
            
            for row in records:
                d_str = str(row['date_str'])
                dates_set.add(d_str)
                if d_str not in self.timeline: self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
                
        self.sorted_dates = sorted(list(dates_set))
        debug_log(f"Timeline built. Total trading days: {len(self.sorted_dates)}")

    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_buy_orders: List[Dict[str, Any]] = [] 
        pending_sell_orders: List[str] = []
        equity_curve: List[float] = []
        total_trades = 0
        
        # 【新規】ポートフォリオ・サーキットブレーカーの管理変数
        circuit_breaker_active = False
        circuit_breaker_cooldown = 0
        current_max_equity = self.initial_cash

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # クーリングオフ期間の消化
            if circuit_breaker_cooldown > 0:
                circuit_breaker_cooldown -= 1
                if circuit_breaker_cooldown == 0:
                    circuit_breaker_active = False

            executed_sells = []
            for ticker in pending_sell_orders:
                if ticker in today_market and ticker in positions:
                    row = today_market[ticker]
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
                    if open_p > 0:
                        pos = positions[ticker]
                        sell_val = pos['qty'] * (open_p * 0.998)
                        cash += sell_val
                        total_trades += 1
                        del positions[ticker]
                        executed_sells.append(ticker)
            pending_sell_orders = [t for t in pending_sell_orders if t not in executed_sells and t in positions]

            for order in pending_buy_orders:
                ticker = str(order['ticker'])
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0.0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
                    limit_p = float(order['limit_price'])
                    
                    self.stats['limit_placed'] += 1
                    
                    if open_p < limit_p * 0.95:
                        self.stats['gap_down_cancels'] += 1
                        continue 
                        
                    if low_p <= limit_p:
                        exec_price = limit_p * 1.002
                        alloc_cash = float(order['allocated_cash'])
                        qty = alloc_cash // exec_price
                        
                        if qty > 0 and cash >= (qty * exec_price):
                            cash -= qty * exec_price
                            positions[ticker] = {
                                'qty': qty, 'entry_p': exec_price, 'high_p': exec_price, 
                                'took_2r': False, 'took_3r': False, 'days_held': 0
                            }
                            self.stats['limit_exec'] += 1

            pending_buy_orders = []

            new_sells_for_tomorrow = []
            for ticker, pos in positions.items():
                if ticker in today_market and ticker not in pending_sell_orders:
                    row = today_market[ticker]
                    curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', pos['entry_p']))
                    current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
                    
                    pos['days_held'] += 1
                    pos['high_p'] = max(pos['high_p'], curr_c)
                    exit_triggered = False
                    
                    # 【完全復元】927%を出したゆとりあるイグジット（無駄な損切り排除）
                    hard_stop_price = max(pos['entry_p'] - (current_atr * 2.0), pos['entry_p'] * 0.88)
                    if curr_c <= hard_stop_price:
                        self.stats['hard_stops'] += 1
                        exit_triggered = True
                    
                    trailing_stop_price = pos['high_p'] - (current_atr * 3.0)
                    if curr_c <= trailing_stop_price and not exit_triggered:
                        self.stats['trailing_stops'] += 1
                        exit_triggered = True
                    
                    if pos['days_held'] >= 20 and curr_c < (pos['entry_p'] * 1.03) and not exit_triggered: 
                        self.stats['time_stops'] += 1
                        if curr_c > pos['entry_p']: self.stats['time_stop_wins'] += 1
                        exit_triggered = True

                    if current_atr > 0 and curr_c > pos['entry_p'] and not exit_triggered:
                        r_mult = (curr_c - pos['entry_p']) / (current_atr * 2)
                        if r_mult >= 5.0 and not pos['took_3r']:
                            sell_qty = int(pos['qty'] // 2)
                            if sell_qty > 0:
                                cash += sell_qty * (curr_c * 0.998)
                                pos['qty'] -= sell_qty
                                total_trades += 1
                            pos['took_3r'] = True
                            pos['took_2r'] = True
                        elif r_mult >= 3.0 and not pos['took_2r']:
                            sell_qty = int(pos['qty'] // 3)
                            if sell_qty > 0:
                                cash += sell_qty * (curr_c * 0.998)
                                pos['qty'] -= sell_qty
                                total_trades += 1
                            pos['took_2r'] = True

                    if exit_triggered:
                        new_sells_for_tomorrow.append(ticker)

            pending_sell_orders.extend(new_sells_for_tomorrow)

            # --- ポートフォリオ資産の計算とサーキットブレーカー判定 ---
            daily_equity = cash
            for ticker, pos in positions.items():
                if ticker in today_market:
                    curr_c = AdvancedStrategyAnalyzer._to_float(today_market[ticker].get('close', pos['entry_p']))
                    daily_equity += pos['qty'] * (curr_c * 0.998)
            equity_curve.append(daily_equity)
            
            # 最高資産の更新とドローダウンの監視
            current_max_equity = max(current_max_equity, daily_equity)
            current_dd = (daily_equity - current_max_equity) / current_max_equity if current_max_equity > 0 else 0
            
            # 【新規】資産がピークから -25% 沈んだら、強制的に新規買いを10日間停止
            if current_dd <= -0.25 and not circuit_breaker_active:
                self.stats['circuit_breaker_hits'] += 1
                circuit_breaker_active = True
                circuit_breaker_cooldown = 10 

            open_slots = self.max_positions - len(positions)
            
            # CB発動中は新規指値の抽出をスキップ
            if not circuit_breaker_active and open_slots > 0 and cash > 0:
                candidates = []
                for ticker, row in today_market.items():
                    if ticker in positions or ticker in pending_sell_orders: 
                        continue 
                    
                    is_entry, score, is_bear = AdvancedStrategyAnalyzer.evaluate_entry(row, self.attr, n_chg, vix)
                    if is_entry:
                        limit_p = AdvancedStrategyAnalyzer.calculate_limit_price(row, self.attr, n_chg, is_bear, vix)
                        vol_ratio = AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio', 0.0))
                        candidates.append((score, vol_ratio, ticker, limit_p))
                
                candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
                
                # 【完全復元】買い枠の復活
                is_high_risk = vix >= 20.0
                max_daily_new_orders = 5 if is_high_risk else self.max_positions
                allowed_slots_today = min(open_slots, max_daily_new_orders)
                
                target_alloc = cash / open_slots if open_slots > 0 else 0
                
                for score, vol_ratio, ticker, limit_p in candidates[:allowed_slots_today]:
                    pending_buy_orders.append({
                        'ticker': ticker,
                        'limit_price': limit_p,
                        'allocated_cash': target_alloc
                    })

        final_equity = equity_curve[-1] if equity_curve else self.initial_cash
        
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            cummax = eq_series.cummax()
            mdd_series = (eq_series - cummax) / cummax
            mdd = float(mdd_series.min()) if not pd.isna(mdd_series.min()) else 0.0
        else:
            mdd = 0.0
            
        ret_val = (final_equity - self.initial_cash) / self.initial_cash
        
        return {
            "Initial_Cash": self.initial_cash,
            "Final_Cash": final_equity,
            "Net_Profit": final_equity - self.initial_cash,
            "Return": f"{ret_val:.2%}",
            "MDD": f"{mdd:.2%}",
            "Total_Trades": total_trades,
            "Stats": self.stats
        }

# ==========================================
# 3. 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running integrity and edge-case tests...")
    
    empty_df = pd.DataFrame()
    res_df = AdvancedStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    dummy_row_err = {'rsi': np.nan, 'dev25': 'invalid', 'rs_21': None}
    try:
        is_entry, score, is_bear = AdvancedStrategyAnalyzer.evaluate_entry(dummy_row_err, "スイング", 0.0, 15.0)
        assert isinstance(score, float), "Corrupted data should be processed safely into a float score."
        assert is_entry is False, "Corrupted data should not trigger an entry."
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data safely: {e}")
        
    try:
        AdvancedStrategyAnalyzer.evaluate_entry("invalid_type", "スイング", 0.0, 15.0) # type: ignore
        assert False, "evaluate_entry should raise TypeError for non-dict input"
    except TypeError:
        pass

    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        data_dir = "Colog_github"
        if not os.path.exists(data_dir):
            data_dir = "data"
            if not os.path.exists(data_dir):
                print(f"[ERROR] Directories 'Colog_github' and 'data' not found. Please run data_fetcher.py first.")
                exit(1)
            
        print("\n==================================================")
        print(" 🚀 STARTING REAL-WORLD PORTFOLIO BACKTEST (PROFIT MAX + PORTFOLIO CB)")
        print("==================================================")
        
        STARTING_CAPITAL = 1000000.0
        MAX_CONCURRENT_POSITIONS = 10
        
        tester = PortfolioBacktester(data_dir=data_dir, initial_cash=STARTING_CAPITAL, max_positions=MAX_CONCURRENT_POSITIONS)
        res = tester.run()
        
        print(f"\n==================================================")
        print(f" 📊 PORTFOLIO SIMULATION RESULTS (Smart Defense V2)")
        print(f"==================================================")
        print(f" ▶ 初期資金 (Initial Cash) : ¥{int(res['Initial_Cash']):,}")
        print(f" ▶ 最終資産 (Final Cash)   : ¥{int(res['Final_Cash']):,}")
        print(f" ▶ 純利益 (Net Profit)     : ¥{int(res['Net_Profit']):,}")
        print(f" ▶ 総利回り (Return)       : {res['Return']}")
        print(f" ▶ 最大下落率 (MDD)        : {res['MDD']}")
        print(f" ▶ 総取引回数 (Trades)     : {res['Total_Trades']} 回")
        
        st = res['Stats']
        exec_rate = (st['limit_exec'] / st['limit_placed']) * 100 if st['limit_placed'] > 0 else 0
        ts_win_rate = (st['time_stop_wins'] / st['time_stops']) * 100 if st['time_stops'] > 0 else 0
        
        print(f"==================================================")
        print(f" 🔬 詳細分析レポート")
        print(f" [1] 指値の約定状況: {st['limit_exec']}/{st['limit_placed']} ({exec_rate:.1f}%)")
        print(f"     ┗ 危険な窓開け回避(注文キャンセル): {st['gap_down_cancels']} 回")
        print(f" [2] 時間切れ撤退(20日): {st['time_stops']} 回 (うち微益: {ts_win_rate:.1f}%)")
        print(f" [3] ハードストップ(絶対防衛線): {st['hard_stops']} 回")
        print(f" [4] トレイリングストップ発動: {st['trailing_stops']} 回")
        print(f" [5] サーキットブレーカー発動: {st['circuit_breaker_hits']} 回 (MDD軽減用)")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
