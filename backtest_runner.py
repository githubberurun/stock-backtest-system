import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, Final
from datetime import datetime, timedelta

# ==========================================
# 参照ドキュメント (2025-2026年最新版)
# Pandas: https://pandas.pydata.org/docs/reference/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数（GitHub Actions等のコンソール出力に最適化）"""
    if not isinstance(msg, str):
        raise TypeError("msg must be a string")
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[DEBUG {timestamp}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (AdvancedStrategyAnalyzer)
# ==========================================
class AdvancedStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        """内部的な型安全性を確保したfloat変換"""
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
        全分析指標の算出
        - ボリンジャーバンド、RSI、MACD、ATR、RS(相対強弱)等
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200:
            return df

        # カラム名の正規化（大文字小文字の差異を吸収）
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # --- 基本指標の算出 ---
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

        # 陽線判定（洗練されたエントリー用）
        df['is_bullish'] = df['close'] > df['open']

        # 反転・逆張りシグナル
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)

        # ATR & Volume
        tr = pd.concat([
            (df['high'] - df['low']),
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        # ベンチマーク（RS分析）
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
            df['rs'] = df['rs_21']
        else:
            for col in ['rs_21', 'rs', 'close_bm', 'bm_ma200']:
                df[col] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float) -> Tuple[bool, float]:
        """
        エントリー評価ロジック（スコアリング方式）
        """
        if not isinstance(row_dict, dict):
            raise TypeError("row_dict must be a dictionary")

        # マクロ・キルスイッチ
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        if bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200:
            return False, 0.0
        if n_chg <= -2.0 or vix >= 20.0:
            return False, 0.0

        # 数値取得
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close'))
        prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close'))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        ma25 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25'))
        ma75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75'))
        ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma200'))
        bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        rsi_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi_slope', 0.0))
        is_bullish = bool(row_dict.get('is_bullish', False))

        main_score = 0.0
        if attr == "押し目":
            is_uptrend = (ma75 > ma200) and (curr_c > ma200)
            if is_uptrend:
                if curr_c < ma25 and rsi_val <= 35 and vol_ratio < 0.8:
                    main_score += 100
                elif curr_c < ma25 and rsi_val <= 45 and vol_ratio <= 1.0:
                    main_score += 60
                if bb_width <= 0.10 and vol_ratio <= 0.8:
                    main_score += 40
        else:  # スイング（順張りメイン）
            if vol_ratio >= 2.0: main_score += 40
            elif vol_ratio >= 1.5: main_score += 20
            if 50 <= rsi_val <= 75: main_score += 15
            elif 75 < rsi_val <= 85: main_score += 5
            elif rsi_val > 85 and curr_c < prev_c: main_score -= 10
            if rsi_slope >= 10.0: main_score += 20
            if 0 < dev25_val <= 20: main_score += 15
            elif dev25_val > 20: main_score += 5
            if bb_width <= 0.10 and vol_ratio <= 0.8: main_score += 20
            
            # 売られすぎ水準での反発サイン検知
            if rsi_val < 30.0 and is_bullish:
                main_score += 50.0
            main_score += 30

        # 最終スコア算出
        surrogate_base = min(100.0, main_score)
        mock_fin_score, mock_appear_count = 3.0, 3.0
        tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
        total_score = (surrogate_base * 0.7) + (mock_fin_score * 3) + (mock_appear_count * 2) - tech_penalty

        is_entry = (total_score >= 80) if attr == "押し目" else (total_score >= 85 and rs_21_val > 0)
        return is_entry, float(total_score)

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float) -> float:
        """
        指値（エントリー価格）の計算
        ATRとNASDAQの下落率に応じた動的な深さ設定
        """
        if not isinstance(row_dict, dict):
            raise TypeError("row_dict must be a dictionary")
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        
        base_offset = 0.5 if "中長期" in attr else (0.0 if attr == "押し目" else 0.3)
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場キャッシュ & バックテスター
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        debug_log("Caching US market data...")
        try:
            ndx_data = yf.Ticker("^IXIC").history(period="10y")
            vix_data = yf.Ticker("^VIX").history(period="10y")
            if not ndx_data.empty and not vix_data.empty:
                self.ndx = ndx_data['Close'].pct_change() * 100
                self.vix = vix_data['Close']
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as e:
            debug_log(f"Market cache error: {e}")
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if not isinstance(date_str, str):
            raise TypeError("date_str must be string")
        if self.ndx.empty or self.vix.empty:
            return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.vix.index:
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str):
            raise TypeError("data_dir must be string")
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.max_positions: int = max_positions
        self.attr: str = "スイング"
        self.us_market = USMarketCache()
        
        debug_log("Loading and calculating indicators for all tickers...")
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        dates_set = set()
        
        bm_path = os.path.join(data_dir, "13060.parquet")
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        
        for file in files:
            ticker = file.replace(".parquet", "")
            df = pd.read_parquet(os.path.join(data_dir, file))
            df = AdvancedStrategyAnalyzer.calculate_indicators(df, bm_df)
            if df.empty:
                continue
            
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            records = df.to_dict(orient='records')
            
            for row in records:
                d_str = row['date_str']
                dates_set.add(d_str)
                if d_str not in self.timeline:
                    self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
                
        self.sorted_dates = sorted(list(dates_set))
        debug_log(f"Timeline built. Total trading days: {len(self.sorted_dates)}")

    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {}
        pending_orders: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        total_trades = 0
        atr_mult: Final[float] = 2.5

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # 約定判定
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0.0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
                    limit_p = order['limit_price']
                    
                    if low_p <= limit_p:
                        exec_price = min(open_p, limit_p)
                        alloc_cash = order['allocated_cash']
                        qty = alloc_cash // exec_price
                        
                        if qty > 0 and cash >= (qty * exec_price):
                            cash -= qty * exec_price
                            positions[ticker] = {
                                'qty': qty, 'entry_p': exec_price, 'high_p': exec_price,
                                'took_2r': False, 'took_3r': False, 'days_held': 0
                            }
                else:
                    new_pending.append(order)
            pending_orders = new_pending

            # エグジット判定
            closed_tickers = []
            for ticker, pos in positions.items():
                if ticker not in today_market:
                    continue
                row = today_market[ticker]
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', 0.0))
                curr_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
                dev25_val = AdvancedStrategyAnalyzer._to_float(row.get('dev25', 0.0))
                rsi_val = AdvancedStrategyAnalyzer._to_float(row.get('rsi', 50.0))
                vol_ratio = AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio', 1.0))
                
                pos['days_held'] += 1
                pos['high_p'] = max(pos['high_p'], curr_c)
                ch_stop = max(pos['high_p'] - (curr_atr * atr_mult), pos['entry_p'] - (curr_atr * atr_mult))
                
                exit_score = 0
                if bool(row.get('bb_3_reversal', False)): exit_score += 40
                if curr_c < ch_stop: exit_score += 100
                
                # クライマックス売り判定
                if dev25_val > 20.0 and rsi_val > 85.0 and vol_ratio > 2.0:
                    exit_score += 100
                
                # タイムストップ（10日経過で利益が2%未満なら撤退）
                if pos['days_held'] >= 10 and curr_c < (pos['entry_p'] * 1.02):
                    exit_score += 100
                
                # 段階的利益確定
                if curr_atr > 0 and curr_c > pos['entry_p']:
                    r_val = (curr_c - pos['entry_p']) / (curr_atr * 2)
                    if r_val >= 3.0 and not pos['took_3r']:
                        sell_qty = int(pos['qty'] // 2)
                        if sell_qty > 0:
                            cash += sell_qty * curr_c
                            pos['qty'] -= sell_qty
                            total_trades += 1
                        pos['took_3r'], pos['took_2r'] = True, True
                    elif r_val >= 2.0 and not pos['took_2r']:
                        sell_qty = int(pos['qty'] // 3)
                        if sell_qty > 0:
                            cash += sell_qty * curr_c
                            pos['qty'] -= sell_qty
                            total_trades += 1
                        pos['took_2r'] = True
                        
                # テクニカル指標によるエグジット（加点）
                if bool(row.get('bb_p1_cross_down', False)): exit_score += 20
                if AdvancedStrategyAnalyzer._to_float(row.get('ma5', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('ma25', 0)) and vol_ratio >= 1.0: exit_score += 15
                if AdvancedStrategyAnalyzer._to_float(row.get('macd', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('sig', 0)) and vol_ratio >= 1.0: exit_score += 15
                if AdvancedStrategyAnalyzer._to_float(row.get('rs', 0)) < -5: exit_score += 5
                
                if exit_score >= 80:
                    cash += pos['qty'] * curr_c
                    total_trades += 1
                    closed_tickers.append(ticker)
                    
            for ct in closed_tickers:
                del positions[ct]

            # 新規エントリー候補の探索
            slots = self.max_positions - len(positions)
            if slots > 0 and cash > 0:
                cands = []
                for ticker, row in today_market.items():
                    if ticker in positions: continue
                    is_entry, score = AdvancedStrategyAnalyzer.evaluate_entry(row, self.attr, n_chg, vix)
                    if is_entry:
                        l_price = AdvancedStrategyAnalyzer.calculate_limit_price(row, self.attr, n_chg)
                        cands.append((score, ticker, l_price))
                
                cands.sort(key=lambda x: x[0], reverse=True)
                for score, ticker, l_price in cands[:slots]:
                    alloc = cash / slots
                    pending_orders.append({'ticker': ticker, 'limit_price': l_price, 'allocated_cash': alloc})
                    slots -= 1

            # 資産曲線記録
            d_equity = cash
            for ticker, pos in positions.items():
                if ticker in today_market:
                    c_price = AdvancedStrategyAnalyzer._to_float(today_market[ticker].get('close', pos['entry_p']))
                    d_equity += pos['qty'] * c_price
            equity_curve.append(d_equity)

        final_equity = equity_curve[-1] if equity_curve else self.initial_cash
        eq_ser = pd.Series(equity_curve)
        mdd = float((eq_ser - eq_ser.cummax()) / eq_ser.cummax()).min() if not eq_ser.empty else 0.0
        
        return {
            "Initial_Cash": self.initial_cash,
            "Final_Cash": final_equity,
            "Net_Profit": final_equity - self.initial_cash,
            "Return": f"{(final_equity - self.initial_cash) / self.initial_cash:.2%}",
            "MDD": f"{mdd:.2%}",
            "Total_Trades": total_trades
        }

# ==========================================
# 3. 堅牢性テスト (GitHub CI/CD用)
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running integrity and edge-case tests...")
    
    # 1. 空データ対応
    empty_df = pd.DataFrame()
    assert AdvancedStrategyAnalyzer.calculate_indicators(empty_df).empty
    
    # 2. 売られすぎ反発判定
    dummy = {'close_bm': 2100, 'bm_ma200': 2000, 'close': 100, 'open': 90, 'rsi': 25, 'is_bullish': True, 'vol_ratio': 1.0, 'rs_21': 5.0}
    is_e, sc = AdvancedStrategyAnalyzer.evaluate_entry(dummy, "スイング", 0.0, 15.0)
    assert sc > 50.0, "Rebound scoring failed"

    # 3. 異常値・欠損値ハンドリング
    bad_data = {'rsi': np.nan, 'dev25': None}
    try:
        AdvancedStrategyAnalyzer.evaluate_entry(bad_data, "スイング", 0.0, 15.0)
    except Exception as e:
        raise AssertionError(f"Error handling corrupted data: {e}")

    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        DATA_PATH = "data"
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Directory '{DATA_PATH}' not found.")
            
        print("\n" + "="*50)
        print(" 🚀 STARTING PORTFOLIO BACKTEST (OPTIMIZED)")
        print("="*50)
        
        tester = PortfolioBacktester(data_dir=DATA_PATH)
        res = tester.run()
        
        print("\n" + "="*50)
        print(" 📊 SIMULATION RESULTS")
        print("="*50)
        for k, v in res.items():
            print(f" ▶ {k:20} : {v}")
        print("="*50)
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
