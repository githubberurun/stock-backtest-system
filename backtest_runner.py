import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def diag_print(msg: str) -> None:
    """GitHub Actionsのバッファリング問題を回避するための即時出力関数"""
    if not isinstance(msg, str): 
        raise TypeError("msg must be a string")
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (AdvancedStrategyAnalyzer)
# ==========================================
class AdvancedStrategyAnalyzer:
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
        """全テクニカル指標の算出。最新のdeep_analyzer等の指標を網羅"""
        if not isinstance(df, pd.DataFrame): 
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # 基本指標
        df['prev_close'] = df['close'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # 乖離率 & ボラティリティ
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        # 陽線判定 & 連続陽線 (deep_analyzer同期)
        df['is_bullish'] = df['close'] > df['open']
        df['bull_streak'] = df['is_bullish'].groupby((~df['is_bullish']).cumsum()).cumsum()
        
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        # MACD (deep_analyzer同期)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        df['macd_hist_diff'] = df['macd_hist'].diff()
        df['is_macd_improving'] = df['macd_hist_diff'] > 0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        # ATR & Volume Ratio
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        # Benchmark (TOPIX) Integration
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
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float, jpy_chg: float) -> Tuple[bool, float, bool]:
        """エントリー判定。最新ロジックを統合し、is_small_capフラグも返す。"""
        if not isinstance(row_dict, dict):
            raise TypeError(f"row_dict must be a dictionary, got {type(row_dict)}")

        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        is_small_cap = 0 < mcap < 500.0

        # --- マクロ防衛線 (Kill Switch) ---
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        is_bear_market = (bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200)
        
        if (jpy_chg <= -1.0) or (n_chg <= -2.0) or (is_bear_market and vix >= 20.0):
            return False, 0.0, is_small_cap

        # --- deep_analyzer同期のネガティブカット ---
        mr_zscore = AdvancedStrategyAnalyzer._to_float(row_dict.get('mr_zscore', 0.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        
        if is_small_cap:
            if mr_zscore >= 1.0: return False, 0.0, is_small_cap
            if dev25_val <= -20.0: return False, 0.0, is_small_cap

        # 指標抽出
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        m25 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25', 0.0))
        m75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75', 0.0))
        m200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        rsi_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi_slope', 0.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        bull_streak = int(row_dict.get('bull_streak', 0))
        is_macd_improving = bool(row_dict.get('is_macd_improving', False))
        ma5 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma5', 0.0))
        
        main_score = 0.0
        
        if attr == "押し目":
            is_uptrend = (m75 > m200) and (curr_c > m200)
            if is_uptrend:
                if curr_c < m25 and rsi_val <= 35 and vol_ratio < 0.8: main_score += 100
                elif curr_c < m25 and rsi_val <= 45 and vol_ratio <= 1.0: main_score += 60
                if bb_width <= 0.10 and vol_ratio <= 0.8: main_score += 40
        else: # スイング
            if vol_ratio >= 2.0: main_score += 40
            elif vol_ratio >= 1.5: main_score += 20
            if 50 <= rsi_val <= 75: main_score += 15
            elif 75 < rsi_val <= 85: main_score += 5
            elif rsi_val > 85 and curr_c < prev_c: main_score -= 10
            if rsi_slope >= 10.0: main_score += 20
            if 0 < dev25_val <= 20: main_score += 15
            elif dev25_val > 20: main_score += 5
            
            # deep_analyzer同期ボーナス＆ペナルティ
            if rsi_val < 30.0 and is_bullish:
                if is_small_cap:
                    if is_macd_improving and vol_ratio >= 1.5: main_score += 50.0
                else:
                    main_score += 50.0 
            main_score += 30 
            
        tech_penalty = 0.0
        if rsi_val > 80: tech_penalty += 20.0
        if dev25_val > 20: tech_penalty += 15.0
        if curr_c >= ma5: tech_penalty += 30.0
        if is_small_cap and bull_streak >= 2: tech_penalty += 50.0

        total_score = (main_score * 0.7) + (3.0 * 3) + (3.0 * 2) - tech_penalty 
        is_entry = (total_score >= 80) if attr == "押し目" else (total_score >= 85 and rs_21_val > 0)
        return is_entry, float(total_score), is_small_cap

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float, is_small_cap: bool) -> float:
        """deep_analyzer同期の深い指値ロジック"""
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be dict")
            
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        
        # 小型株は深い指値(ATR*1.0)を採用。大型は確実性重視(ATR*0.1)
        base_offset = 1.0 if is_small_cap else 0.1
        if attr == "押し目": base_offset = 2.0 if is_small_cap else 0.3
        
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場キャッシュ & バックテスター本体
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        diag_print("Fetching US market indices (NASDAQ, VIX, USDJPY)...")
        try:
            ndx = yf.Ticker("^IXIC").history(period="10y")
            vix = yf.Ticker("^VIX").history(period="10y")
            usdjpy = yf.Ticker("USDJPY=X").history(period="10y")
            
            if not ndx.empty and not vix.empty and not usdjpy.empty:
                self.ndx = ndx['Close'].pct_change() * 100
                self.vix = vix['Close']
                self.jpy = usdjpy['Close'].pct_change() * 100
                
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
                self.jpy.index = self.jpy.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix, self.jpy = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as e:
            diag_print(f"⚠️ US Market Cache Error: {e}")
            self.ndx, self.vix, self.jpy = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float, float]:
        if self.ndx.empty or self.vix.empty or self.jpy.empty: return 0.0, 15.0, 0.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.jpy.index: 
                return float(self.ndx[prev]), float(self.vix[prev]), float(self.jpy[prev])
        return 0.0, 15.0, 0.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.us_market = USMarketCache()
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # バックテスト検証項目用ログデータ
        self.stats = {
            'limit_placed_small': 0, 'limit_exec_small': 0,
            'limit_placed_large': 0, 'limit_exec_large': 0,
            'ts_small': 0, 'ts_small_win': 0,
            'ts_large': 0, 'ts_large_win': 0,
            'kill_switch_days': 0, 'kill_switch_avoids': 0
        }
        
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
        total_trades = 0

        for i, date_str in enumerate(self.sorted_dates):
            today_market = self.timeline[date_str]
            n_chg, vix, jpy_chg = self.us_market.get_state(date_str)
            
            # マクロキルスイッチの判定
            is_bear_market = False
            if '13060' in today_market:
                bm_c = AdvancedStrategyAnalyzer._to_float(today_market['13060'].get('close', 0))
                bm_m200 = AdvancedStrategyAnalyzer._to_float(today_market['13060'].get('bm_ma200', 0))
                is_bear_market = (bm_c > 0 and bm_m200 > 0 and bm_c < bm_m200)
                
            is_kill_switch = (jpy_chg <= -1.0) or (n_chg <= -2.0) or (is_bear_market and vix >= 20.0)
            if is_kill_switch: self.stats['kill_switch_days'] += 1
            
            # 1. 約定判定（深い指値の検証）
            unfilled_orders = []
            for order in pending_orders:
                t = order['ticker']
                if t in today_market and len(positions) < self.max_positions:
                    row = today_market[t]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low'))
                    
                    if order['is_small_cap']: self.stats['limit_placed_small'] += 1
                    else: self.stats['limit_placed_large'] += 1
                    
                    if low_p <= order['limit_price']:
                        exec_p = min(AdvancedStrategyAnalyzer._to_float(row.get('open')), order['limit_price'])
                        qty = order['allocated_cash'] // exec_p
                        if qty > 0 and cash >= (qty * exec_p):
                            cash -= qty * exec_p
                            positions[t] = {'qty': qty, 'entry_p': exec_p, 'high_p': exec_p, 'days_held': 0, 'is_sc': order['is_small_cap'], 'took_2r': False, 'took_3r': False}
                            if order['is_small_cap']: self.stats['limit_exec_small'] += 1
                            else: self.stats['limit_exec_large'] += 1
            pending_orders = []

            # 2. エグジット判定
            closed = []
            for t, p in positions.items():
                if t not in today_market: continue
                row = today_market[t]
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close'))
                current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr'))
                dev25 = AdvancedStrategyAnalyzer._to_float(row.get('dev25'))
                rsi = AdvancedStrategyAnalyzer._to_float(row.get('rsi'))
                vol_ratio = AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio'))
                
                p['days_held'] += 1
                p['high_p'] = max(p['high_p'], curr_c)
                
                # exit_strategy 同期: クライマックス売り
                is_climax = False
                if p['is_sc'] and dev25 > 50.0 and rsi > 90.0 and vol_ratio >= 3.0: is_climax = True
                elif not p['is_sc'] and dev25 > 20.0 and rsi > 85.0 and vol_ratio >= 2.0: is_climax = True
                
                # exit_strategy 同期: タイムストップ
                time_stop_days = 20 if p['is_sc'] else 10
                is_time_stop = (p['days_held'] >= time_stop_days and curr_c < (p['entry_p'] * 1.02))
                
                atr_mult = 3.5 if p['is_sc'] else 2.5
                stop = max(p['high_p'] - (current_atr * atr_mult), p['entry_p'] - (current_atr * atr_mult))
                
                if curr_c < stop or is_climax or is_time_stop:
                    cash += p['qty'] * curr_c
                    total_trades += 1
                    closed.append(t)
                    
                    if is_time_stop:
                        win = curr_c > p['entry_p']
                        if p['is_sc']:
                            self.stats['ts_small'] += 1
                            if win: self.stats['ts_small_win'] += 1
                        else:
                            self.stats['ts_large'] += 1
                            if win: self.stats['ts_large_win'] += 1
                            
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
                if is_kill_switch:
                    self.stats['kill_switch_avoids'] += open_slots
                else:
                    candidates = []
                    for t, row in today_market.items():
                        if t not in positions:
                            for mode in ["押し目", "スイング"]:
                                is_ok, score, is_sc = AdvancedStrategyAnalyzer.evaluate_entry(row, mode, n_chg, vix, jpy_chg)
                                if is_ok:
                                    candidates.append((score, t, AdvancedStrategyAnalyzer.calculate_limit_price(row, mode, n_chg, is_sc), is_sc))
                                    break
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for score, ticker, lp, is_sc in candidates[:open_slots]:
                        pending_orders.append({'ticker': ticker, 'limit_price': lp, 'allocated_cash': cash / open_slots, 'is_small_cap': is_sc})
                        open_slots -= 1

            equity_curve.append(cur_equity)
            if (i+1) % 500 == 0: diag_print(f"Progress: Day {i+1}/{len(self.sorted_dates)} | Equity: ¥{int(cur_equity):,}")

        eq_s = pd.Series(equity_curve)
        return {
            "Initial": self.initial_cash, "Final": eq_s.iloc[-1], 
            "Return": (eq_s.iloc[-1]/self.initial_cash)-1, 
            "MDD": ((eq_s - eq_s.cummax())/eq_s.cummax()).min(), 
            "Trades": total_trades, "Stats": self.stats
        }

# ==========================================
# 3. 堅牢性テスト証明
# ==========================================
def run_integrity_tests() -> None:
    diag_print("Running integrity tests...")
    # 1. 空DF
    assert AdvancedStrategyAnalyzer.calculate_indicators(pd.DataFrame()).empty
    # 2. 不正な型への耐性
    try: 
        AdvancedStrategyAnalyzer.evaluate_entry("invalid", "スイング", 0.0, 15.0, 0.0) # type: ignore
        assert False, "Should raise TypeError"
    except TypeError: pass
    
    # 3. 小型株のフィルター動作検証
    row_small = {'mcap': 300.0, 'close_bm': 210, 'bm_ma200': 200, 'mr_zscore': 1.5} # しこり異常
    ok, _, is_sc = AdvancedStrategyAnalyzer.evaluate_entry(row_small, "スイング", 0.0, 15.0, 0.0)
    assert is_sc is True, "Small cap flag failed"
    assert ok is False, "Z-score cutoff for small cap failed"

    diag_print("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        results = PortfolioBacktester(data_dir="data", max_positions=5).run()
        print(f"\n==================================================\n 📊 RESULTS (Deep Analyzer Synchronized Ver.)\n==================================================")
        print(f" ▶ 初期資金 : ¥{int(results['Initial']):,}\n ▶ 最終資産 : ¥{int(results['Final']):,}\n ▶ 総利回り : {results['Return']:.2%}\n ▶ 最大下落 : {results['MDD']:.2%}\n ▶ 取引回数 : {results['Trades']} 回")
        
        # 検証項目レポート出力
        st = results['Stats']
        sm_rate = (st['limit_exec_small'] / st['limit_placed_small']) * 100 if st['limit_placed_small'] > 0 else 0
        lg_rate = (st['limit_exec_large'] / st['limit_placed_large']) * 100 if st['limit_placed_large'] > 0 else 0
        ts_sm_win_rate = (st['ts_small_win'] / st['ts_small']) * 100 if st['ts_small'] > 0 else 0
        
        print(f"==================================================")
        print(f" 🔬 アドバイザリー検証レポート")
        print(f" [1] 深い指値の約定率 (小型株ダマシ回避検証)")
        print(f"     - 小型株(深め) : {st['limit_exec_small']}/{st['limit_placed_small']} ({sm_rate:.1f}%)")
        print(f"     - 大型株(浅め) : {st['limit_exec_large']}/{st['limit_placed_large']} ({lg_rate:.1f}%)")
        print(f" [2] タイムストップと小型株の握力 (20日 vs 10日)")
        print(f"     - 小型特例発動回数 : {st['ts_small']} 回 (うち微益撤退: {ts_sm_win_rate:.1f}%)")
        print(f"     - 大型通常発動回数 : {st['ts_large']} 回")
        print(f" [3] マクロ・キルスイッチ有効性")
        print(f"     - 稼働日数 : {st['kill_switch_days']} 日")
        print(f"     - 回避した危険なエントリー数 : {st['kill_switch_avoids']} 回")
        print(f"==================================================", flush=True)
    except Exception as e:
        diag_print(f"❌ FATAL: {e}"); import traceback; traceback.print_exc()
