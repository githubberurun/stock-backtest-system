import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def diag_print(msg: str) -> None:
    """GitHub Actionsのログに即座に出力するための診断用関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (Entry & Exit & Limit Price)
# ==========================================
class AdvancedStrategyAnalyzer:
    """
    テクニカル・需給・時価総額・マクロ環境を統合して判断する分析クラス
    """
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # --- 基本指標の算出 ---
        df['prev_close'] = df['close'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # --- ボリンジャーバンド & 反転サイン ---
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        df['is_bullish'] = df['close'] > df['open']
        
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        # --- オシレーター (MACD & RSI) ---
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        # --- ボラティリティ & 出来高 ---
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        # --- ベンチマーク(TOPIX 200MA) フィルター用マージ ---
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs_21'] = 0.0
            df['close_bm'] = 0.0
            df['bm_ma200'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], n_chg: float, vix: float) -> Tuple[bool, float]:
        """
        マクロ・テクニカル・需給を複合判定しエントリー可否を返す
        """
        # 1. 強力なマクロフィルター (TOPIX 200MA) - 防御の要
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        if bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200:
            return False, 0.0  
        
        # NASDAQ急落またはVIX上昇による Kill Switch
        if n_chg <= -2.0 or vix >= 20.0: 
            return False, 0.0

        # 変数抽出
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        
        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        mr_zscore = AdvancedStrategyAnalyzer._to_float(row_dict.get('mr_zscore', 0.0))
        is_small_cap = 0.0 < mcap < 500.0 if mcap > 0 else False

        # 2. 小型株の需給悪化・異常乖離ネガティブカット
        if is_small_cap and (mr_zscore >= 1.0 or dev25_val <= -20.0):
            return False, 0.0

        # 3. スコアリング (実戦的な重み付け)
        score = 0.0
        if vol_ratio >= 1.5: score += 40
        if 40 <= rsi_val <= 70: score += 20
        if rsi_val < 30.0 and is_bullish: score += 50 # 逆張り反発期待
        if rs_21_val > 0: score += 20 # 相対的強さ
        if bb_width <= 0.15: score += 15 # スクイーズ（ボラティリティの収束）
        
        score += 30 # ベース加点
        
        # ペナルティ設定
        tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
        total_score = (score * 0.7) - tech_penalty 
        
        # エントリー閾値判定
        is_entry = (total_score >= 85 and rs_21_val > 0)
        return is_entry, float(total_score)

    @staticmethod
    def get_order_params(row_dict: Dict[str, Any], n_chg: float) -> Tuple[float, float]:
        """
        時価総額に応じた指値の深さとATR損切幅を算出
        """
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        
        is_small_cap = 0.0 < mcap < 500.0 if mcap > 0 else False
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        
        # 小型株は深い押し目を狙い、損切幅（ATR倍数）も広く取る
        if is_small_cap:
            base_offset = 0.8
            atr_mult = 2.5 
        else:
            base_offset = 0.3
            atr_mult = 2.0
            
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price)), float(atr_mult)

# ==========================================
# 2. 市場環境 & ポートフォリオバックテスター
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        diag_print("Caching US market data via yfinance...")
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
            if prev in self.ndx.index: return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 10) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.us_market = USMarketCache()
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        diag_print(f"Data directory: {os.path.abspath(data_dir)}")
        if not os.path.exists(data_dir): raise FileNotFoundError(f"Directory {data_dir} not found.")
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        diag_print(f"Found {len(files)} potential ticker files.")
        
        bm_path = os.path.join(data_dir, "13060.parquet")
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        if bm_df is None: diag_print("⚠️ WARNING: 13060.parquet (TOPIX) not found. Macro filter will be inactive.")

        # --- データのロードとタイムライン構築 ---
        dates_set = set()
        for i, file in enumerate(files):
            df = pd.read_parquet(os.path.join(data_dir, file))
            df = AdvancedStrategyAnalyzer.calculate_indicators(df, bm_df)
            if df.empty: continue
            
            ticker = file.replace(".parquet", "")
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            for row in df.to_dict(orient='records'):
                d = row['date_str']
                dates_set.add(d)
                if d not in self.timeline: self.timeline[d] = {}
                self.timeline[d][ticker] = row
            
            if (i+1) % 50 == 0: diag_print(f"Loading progress: {i+1}/{len(files)} files loaded.")

        self.sorted_dates = sorted(list(dates_set))
        diag_print(f"Timeline built. Total trading days: {len(self.sorted_dates)}")

    def run(self) -> Dict[str, Any]:
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0

        if not self.sorted_dates:
            diag_print("❌ Error: No trading days found in timeline. Ending simulation.")
            return {"Initial": self.initial_cash, "Final": self.initial_cash, "Return": 0, "MDD": 0, "Trades": 0}

        for i, date_str in enumerate(self.sorted_dates):
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # --- 1. 前日の予約注文の約定処理 ---
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0))
                    
                    if low_p <= order['limit_price']:
                        exec_p = min(open_p, order['limit_price'])
                        qty = order['qty']
                        required_cash = qty * exec_p
                        if self.cash >= required_cash:
                            self.cash -= required_cash
                            positions[ticker] = {
                                'qty': qty, 'entry_p': exec_p, 'high_p': exec_p, 
                                'atr_mult': order['atr_mult'], 'days_held': 0,
                                'took_2r': False, 'took_3r': False
                            }
            pending_orders = [] # 当日の注文は翌朝処理

            # --- 2. エグジット判定 (Chandelier & Time stop & R-Multi) ---
            closed = []
            for t, p in positions.items():
                if t not in today_market: continue
                row = today_market[t]
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', 0))
                atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0))
                
                p['days_held'] += 1
                p['high_p'] = max(p['high_p'], curr_c)
                
                # 損切ライン (シャンデリアストップ：高値または買値からATRの一定倍率下落)
                stop_line = max(p['high_p'] - (atr * p['atr_mult']), p['entry_p'] - (atr * p['atr_mult']))
                
                if curr_c < stop_line or p['days_held'] >= 10:
                    self.cash += p['qty'] * curr_c
                    total_trades += 1
                    closed.append(t)
                else:
                    # 部分利確ロジック (R倍数ベース)
                    if atr > 0:
                        r_mult = (curr_c - p['entry_p']) / (atr * 2) # リスク単位をATR*2と仮定
                        if r_mult >= 3.0 and not p['took_3r']:
                            sell_qty = int(p['qty'] // 2)
                            if sell_qty > 0:
                                self.cash += sell_qty * curr_c
                                p['qty'] -= sell_qty
                                total_trades += 1
                                p['took_3r'] = True
                        elif r_mult >= 2.0 and not p['took_2r']:
                            sell_qty = int(p['qty'] // 3)
                            if sell_qty > 0:
                                self.cash += sell_qty * curr_c
                                p['qty'] -= sell_qty
                                total_trades += 1
                                p['took_2r'] = True
                                
            for t in closed: del positions[t]

            # --- 3. 新規エントリー探索と動的資金管理 ---
            # 建玉の時価評価額を含めた現在の総資産
            current_equity = self.cash + sum(
                p['qty'] * AdvancedStrategyAnalyzer._to_float(today_market[t].get('close', p['entry_p'])) 
                for t, p in positions.items() if t in today_market
            )
            
            open_slots = self.max_positions - len(positions)
            if open_slots > 0:
                candidates = []
                for t, row in today_market.items():
                    if t not in positions:
                        is_ok, score = AdvancedStrategyAnalyzer.evaluate_entry(row, n_chg, vix)
                        if is_ok:
                            lp, am = AdvancedStrategyAnalyzer.get_order_params(row, n_chg)
                            candidates.append((score, t, lp, am, row))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                for score, ticker, lp, am, row in candidates[:open_slots]:
                    # 1%リスク許容度に基づく枚数計算 (Deep Analyzer準拠)
                    atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 1.0))
                    risk_per_share = max(0.1, lp - (lp - atr * am))
                    risk_fund = current_equity * 0.01
                    raw_shares = int((risk_fund / risk_per_share) // 100 * 100)
                    
                    # 1銘柄あたりの最大投資枠制限 (スロット均等割：10%)
                    max_alloc = current_equity / self.max_positions
                    max_shares = int((max_alloc / lp) // 100 * 100)
                    
                    qty = min(raw_shares, max_shares)
                    
                    if qty >= 100 and self.cash >= (qty * lp):
                        pending_orders.append({
                            'ticker': ticker, 
                            'limit_price': lp, 
                            'atr_mult': am, 
                            'qty': qty
                        })
            
            equity_curve.append(current_equity)
            if (i+1) % 500 == 0: 
                diag_print(f"Simulation progress: Day {i+1}/{len(self.sorted_dates)} | Equity: ¥{int(current_equity):,}")

        # --- 結果集計 ---
        eq_series = pd.Series(equity_curve)
        if eq_series.empty: 
            return {"Initial": self.initial_cash, "Final": self.initial_cash, "Return": 0, "MDD": 0, "Trades": 0}
        
        mdd = (eq_series - eq_series.cummax()) / eq_series.cummax()
        return {
            "Initial": self.initial_cash, 
            "Final": eq_series.iloc[-1], 
            "Return": (eq_series.iloc[-1] / self.initial_cash) - 1, 
            "MDD": float(mdd.min()), 
            "Trades": total_trades
        }

# ==========================================
# 3. 堅牢性テスト & メイン実行
# ==========================================
def run_integrity_tests() -> None:
    diag_print("Running integrity tests...")
    
    # 1. 指値ロジックの妥当性チェック
    dummy_small = {'close': 1000.0, 'atr': 50.0, 'mcap': 300.0}
    dummy_large = {'close': 1000.0, 'atr': 50.0, 'mcap': 1000.0}
    lp_s, am_s = AdvancedStrategyAnalyzer.get_order_params(dummy_small, 0.0)
    lp_l, am_l = AdvancedStrategyAnalyzer.get_order_params(dummy_large, 0.0)
    assert lp_s < lp_l, "Diagnostic fail: Small cap limit price should be deeper."
    assert am_s > am_l, "Diagnostic fail: Small cap ATR mult should be wider."
    
    # 2. マクロフィルター(TOPIX 200MA)の動作チェック
    bear_row = {'close_bm': 190.0, 'bm_ma200': 200.0, 'close': 100.0, 'rsi': 25.0, 'rs_21': 5.0}
    ok, _ = AdvancedStrategyAnalyzer.evaluate_entry(bear_row, 0.0, 15.0)
    assert not ok, "Diagnostic fail: Entry must be blocked in bear market."

    diag_print("All integrity tests passed.")

if __name__ == "__main__":
    diag_print("--- STARTING HYBRID BACKTEST Runner Ver.4.1 (FULL) ---")
    run_integrity_tests()
    
    try:
        data_path = "data"
        # スロット制（10銘柄）を導入し、分散とリスク管理を強化
        tester = PortfolioBacktester(data_dir=data_path, initial_cash=1000000.0, max_positions=10)
        results = tester.run()
        
        print("\n" + "="*50)
        print(" 📊 PORTFOLIO SIMULATION RESULTS (HYBRID OPTIMIZED)")
        print("="*50)
        print(f" ▶ 初期資金 : ¥{int(results['Initial']):,}")
        print(f" ▶ 最終資産 : ¥{int(results['Final']):,}")
        print(f" ▶ 純利益   : ¥{int(results['Final'] - results['Initial']):,}")
        print(f" ▶ 総利回り : {results['Return']:.2%}")
        print(f" ▶ 最大下落 : {results['MDD']:.2%}")
        print(f" ▶ 取引回数 : {results['Trades']} 回")
        print("="*50, flush=True)
        
    except Exception as e:
        diag_print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
