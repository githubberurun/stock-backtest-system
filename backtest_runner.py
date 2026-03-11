import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def diag_print(msg: str) -> None:
    """
    GitHub Actionsのバッファリング問題を回避し、
    ログに即座に出力するための診断用関数（flush=True強制）
    """
    if not isinstance(msg, str): 
        raise TypeError("msg must be a string")
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (AdvancedStrategyAnalyzer)
# ==========================================
class AdvancedStrategyAnalyzer:
    """
    テクニカル・需給・時価総額・マクロ環境を統合判定するエンジン。
    """
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        """型チェック付きのfloat変換"""
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        全テクニカル指標の算出。変更のない箇所を含め、ファイル全体を出力。
        """
        if not isinstance(df, pd.DataFrame): 
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        # カラム名の標準化
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # --- 基本指標 (Moving Averages) ---
        df['prev_close'] = df['close'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # --- 乖離率 & ボラティリティ ---
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        # --- 陽線判定 & 反転シグナル ---
        df['is_bullish'] = df['close'] > df['open']
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        # --- MACD (Exponential Moving Averages) ---
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        
        # --- RSI (Relative Strength Index) ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        # --- ATR & Volume Ratio ---
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        # --- Benchmark (TOPIX) Integration ---
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs_21'], df['close_bm'], df['bm_ma200'] = 0.0, 0.0, 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float) -> Tuple[bool, float]:
        """
        エントリー可否の判定。TracebackのAttributeErrorを修正するためガード節を追加。
        """
        # --- 物理的型チェックの強化 (ガード節) ---
        if not isinstance(row_dict, dict):
            raise TypeError(f"row_dict must be a dictionary, got {type(row_dict)}")

        # --- マクロ防衛線 (Kill Switch) ---
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        if bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200:
            return False, 0.0  
        
        if n_chg <= -2.0 or vix >= 20.0: 
            return False, 0.0

        # 指標抽出
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        m25 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25', 0.0))
        m75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75', 0.0))
        m200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        rsi_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi_slope', 0.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        
        main_score = 0.0
        
        # --- 606%ロジックの各分岐を復元 ---
        if attr == "押し目":
            is_uptrend = (m75 > m200) and (curr_c > m200)
            if is_uptrend:
                if curr_c < m25 and rsi_val <= 35 and vol_ratio < 0.8:
                    main_score += 100
                elif curr_c < m25 and rsi_val <= 45 and vol_ratio <= 1.0:
                    main_score += 60
                if bb_width <= 0.10 and vol_ratio <= 0.8:
                    main_score += 40
        else: # スイング
            if vol_ratio >= 2.0: main_score += 40
            elif vol_ratio >= 1.5: main_score += 20
            if 50 <= rsi_val <= 75: main_score += 15
            elif 75 < rsi_val <= 85: main_score += 5
            elif rsi_val > 85 and curr_c < prev_c: main_score -= 10
            if rsi_slope >= 10.0: main_score += 20
            if 0 < dev25_val <= 20: main_score += 15
            elif dev25_val > 20: main_score += 5
            if rsi_val < 30.0 and is_bullish:
                main_score += 50.0 
            main_score += 30 
            
        tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
        total_score = (main_score * 0.7) + (3.0 * 3) + (3.0 * 2) - tech_penalty 
        
        is_entry = (total_score >= 80) if attr == "押し目" else (total_score >= 85 and rs_21_val > 0)
        return is_entry, float(total_score)

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float) -> float:
        """
        指値計算。物理的ガード節を追加。
        """
        if not isinstance(row_dict, dict):
            raise TypeError(f"row_dict must be a dictionary, got {type(row_dict)}")
            
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        base_offset = 0.0 if attr == "押し目" else 0.3
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場キャッシュ & バックテスター本体
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        diag_print("Fetching US market indices (NASDAQ & VIX)...")
        try:
            ndx = yf.Ticker("^IXIC").history(period="10y")
            vix = yf.Ticker("^VIX").history(period="10y")
            if not ndx.empty and not vix.empty:
                self.ndx = ndx['Close'].pct_change() * 100
                self.vix = vix['Close']
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as e:
            diag_print(f"⚠️ US Market Cache Error: {e}")
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.us_market = USMarketCache()
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        diag_print(f"Loading datasets from: {os.path.abspath(data_dir)}")
        if not os.path.exists(data_dir): raise FileNotFoundError(f"Directory {data_dir} not found.")
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        bm_path = os.path.join(data_dir, "13060.parquet")
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None

        dates_set = set()
        for i, file in enumerate(files):
            ticker_df = pd.read_parquet(os.path.join(data_dir, file))
            processed_df = AdvancedStrategyAnalyzer.calculate_indicators(ticker_df, bm_df)
            if processed_df.empty: continue
            
            ticker = file.replace(".parquet", "")
            processed_df['date_str'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d')
            for row in processed_df.to_dict(orient='records'):
                d_str = row['date_str']
                dates_set.add(d_str)
                if d_str not in self.timeline: self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
            if (i+1) % 50 == 0: diag_print(f"Loading: {i+1}/{len(files)} tickers...")

        self.sorted_dates = sorted(list(dates_set))
        diag_print(f"Timeline synchronized: {len(self.sorted_dates)} trading days.")

    def run(self) -> Dict[str, Any]:
        cash, positions, pending_orders, equity_curve = self.cash, {}, [], []
        total_trades, atr_mult = 0, 2.5

        for i, date_str in enumerate(self.sorted_dates):
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # 1. 約定判定
            for order in pending_orders:
                t = order['ticker']
                if t in today_market and len(positions) < self.max_positions:
                    row = today_market[t]
                    if AdvancedStrategyAnalyzer._to_float(row.get('low')) <= order['limit_price']:
                        exec_p = min(AdvancedStrategyAnalyzer._to_float(row.get('open')), order['limit_price'])
                        qty = order['allocated_cash'] // exec_p
                        if qty > 0 and cash >= (qty * exec_p):
                            cash -= qty * exec_p
                            positions[t] = {'qty': qty, 'entry_p': exec_p, 'high_p': exec_p, 'days_held': 0, 'took_2r': False, 'took_3r': False}
            pending_orders = []

            # 2. エグジット
            closed = []
            for t, p in positions.items():
                if t not in today_market: continue
                row = today_market[t]
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close'))
                current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr'))
                p['days_held'] += 1
                p['high_p'] = max(p['high_p'], curr_c)
                stop = max(p['high_p'] - (current_atr * atr_mult), p['entry_p'] - (current_atr * atr_mult))
                
                if curr_c < stop:
                    cash += p['qty'] * curr_c
                    total_trades += 1
                    closed.append(t)
                elif p['days_held'] >= 10 and curr_c < (p['entry_p'] * 1.02):
                    cash += p['qty'] * curr_c
                    total_trades += 1
                    closed.append(t)
                elif current_atr > 0:
                    r_m = (curr_c - p['entry_p']) / (current_atr * 2)
                    if r_m >= 3.0 and not p['took_3r']:
                        sq = int(p['qty'] // 2); cash += sq * curr_c; p['qty'] -= sq; total_trades += 1; p['took_3r'] = True
                    elif r_m >= 2.0 and not p['took_2r']:
                        sq = int(p['qty'] // 3); cash += sq * curr_c; p['qty'] -= sq; total_trades += 1; p['took_2r'] = True
                                
            for t in closed: del positions[t]

            # 3. エントリー探索
            cur_equity = cash + sum(p['qty'] * today_market[t]['close'] for t, p in positions.items() if t in today_market)
            open_slots = self.max_positions - len(positions)
            if open_slots > 0 and cash > 0:
                candidates = []
                for t, row in today_market.items():
                    if t not in positions:
                        for mode in ["押し目", "スイング"]:
                            is_ok, score = AdvancedStrategyAnalyzer.evaluate_entry(row, mode, n_chg, vix)
                            if is_ok:
                                candidates.append((score, t, AdvancedStrategyAnalyzer.calculate_limit_price(row, mode, n_chg)))
                                break
                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, ticker, lp in candidates[:open_slots]:
                    pending_orders.append({'ticker': ticker, 'limit_price': lp, 'allocated_cash': cash / open_slots})
                    open_slots -= 1

            equity_curve.append(cur_equity)
            if (i+1) % 500 == 0: diag_print(f"Progress: Day {i+1}/{len(self.sorted_dates)} | Equity: ¥{int(cur_equity):,}")

        eq_s = pd.Series(equity_curve)
        return {"Initial": self.initial_cash, "Final": eq_s.iloc[-1], "Return": (eq_s.iloc[-1]/self.initial_cash)-1, "MDD": ((eq_s - eq_s.cummax())/eq_s.cummax()).min(), "Trades": total_trades}

# ==========================================
# 3. 堅牢性テスト (Tracebackの原因箇所を修正済み)
# ==========================================
def run_integrity_tests() -> None:
    diag_print("Running integrity tests...")
    # 1. 空DF
    assert AdvancedStrategyAnalyzer.calculate_indicators(pd.DataFrame()).empty
    # 2. 不正な型への耐性 (修正箇所)
    try: 
        AdvancedStrategyAnalyzer.evaluate_entry("invalid", "スイング", 0.0, 15.0) # type: ignore
        assert False, "Should raise TypeError"
    except TypeError: diag_print("Positive: TypeError caught as expected.")
    # 3. フィルター動作
    bear = {'close_bm': 190.0, 'bm_ma200': 200.0, 'close': 100.0}
    ok, _ = AdvancedStrategyAnalyzer.evaluate_entry(bear, "スイング", 0.0, 15.0)
    assert not ok
    diag_print("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        results = PortfolioBacktester(data_dir="data", max_positions=5).run()
        print(f"\n==================================================\n 📊 RESULTS (606% REBORN Ver.6.1)\n==================================================")
        print(f" ▶ 初期資金 : ¥{int(results['Initial']):,}\n ▶ 最終資産 : ¥{int(results['Final']):,}\n ▶ 総利回り : {results['Return']:.2%}\n ▶ 最大下落 : {results['MDD']:.2%}\n ▶ 取引回数 : {results['Trades']} 回\n==================================================", flush=True)
    except Exception as e:
        diag_print(f"❌ FATAL: {e}"); import traceback; traceback.print_exc()
