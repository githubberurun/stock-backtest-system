import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 1. 統合分析エンジン (Entry & Exit & Limit Price)
# ==========================================
class AdvancedStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        """異常値や欠損値を安全にfloat型へ変換する内部メソッド"""
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 25: 
            return df
            
        df.columns = [str(c).lower() for c in df.columns]

        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}。列名が異なる場合は事前に小文字にリネームしてください。")

        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['prev_low'] = df['low'].shift(1)
        
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])

        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            df = df.merge(benchmark_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
            df['rs'] = df['rs_21']
        else:
            df['rs_21'] = 0.0
            df['rs'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row: pd.Series, attr: str, n_chg: float, vix: float) -> bool:
        if not isinstance(row, pd.Series): raise TypeError("row must be pd.Series")
        if n_chg <= -2.0 or vix >= 20.0: return False
        
        # 【改修】安全な型変換メソッドを使用して値を取得
        rsi_val = AdvancedStrategyAnalyzer._to_float(row.get('rsi', 50.0), 50.0)
        dev25_val = AdvancedStrategyAnalyzer._to_float(row.get('dev25', 0.0), 0.0)
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row.get('rs_21', 0.0), 0.0)
        
        tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
        total_score = (50 * 0.7) + (2 * 3) + (1 * 2) - tech_penalty 
        
        if attr == "押し目": return total_score >= 80
        return total_score >= 85 and rs_21_val > 0

    @staticmethod
    def calculate_limit_price(row: pd.Series, attr: str, n_chg: float) -> float:
        if not isinstance(row, pd.Series): raise TypeError("row must be pd.Series")
        
        # 【改修】安全な型変換メソッドを使用
        curr_price = AdvancedStrategyAnalyzer._to_float(row.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
        
        if "中長期" in attr: base_offset = 0.5
        elif attr == "押し目": base_offset = 0.0
        else: base_offset = 0.3

        is_hybrid_guardrail = (n_chg <= -0.8)
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if is_hybrid_guardrail else 0.0
        price_shift = curr_price * nasdaq_drop_ratio

        limit_price = curr_price - (atr * base_offset) - price_shift
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場キャッシュ & バックテスター
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        print("[INFO] Caching US market data...")
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
            print(f"[WARN] USMarketCache init failed: {e}")
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
            
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.vix.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class IntegratedBacktester:
    def __init__(self, ticker: str, initial_cash: float = 1000000.0, attr: str = "スイング") -> None:
        self.ticker: str = ticker
        self.cash: float = initial_cash
        self.attr: str = attr
        
        file_path = f"data/{ticker}.parquet" 
        if os.path.exists(file_path):
            target_df = pd.read_parquet(file_path)
            bm_df = pd.read_parquet("data/13060.parquet") if os.path.exists("data/13060.parquet") else None
        else:
            print(f"[WARN] {file_path} not found. Creating dummy data for testing.")
            dates = pd.date_range(end=datetime.now(), periods=100)
            target_df = pd.DataFrame({'date': dates, 'open': 1000, 'high': 1050, 'low': 950, 'close': 1020, 'volume': 10000})
            bm_df = pd.DataFrame({'date': dates, 'close': 2000})

        self.df = AdvancedStrategyAnalyzer.calculate_indicators(target_df, bm_df)
        self.df['date_str'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')
        self.us_market = USMarketCache()

    def run(self) -> Dict[str, Any]:
        cash, pos, entry_p, high_p = self.cash, 0.0, 0.0, 0.0
        atr_mult = 3.0 if "中長期" in self.attr else (2.0 if self.attr == "押し目" else 2.5)
        trades, equity = [], []
        
        pending_entry = False
        target_limit_price = 0.0

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            # 【改修】すべて安全な型変換を使用
            curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', 0.0))
            curr_l = AdvancedStrategyAnalyzer._to_float(row.get('low', 0.0))
            curr_o = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
            n_chg, vix = self.us_market.get_state(str(row['date_str']))

            if pos == 0:
                if pending_entry:
                    if curr_l <= target_limit_price:
                        entry_p = min(curr_o, target_limit_price)
                        pos = cash // entry_p
                        high_p = entry_p
                        cash -= pos * entry_p
                        pending_entry = False
                        equity.append(cash + (pos * curr_c))
                        continue 
                    else:
                        pending_entry = False

                if not pending_entry and AdvancedStrategyAnalyzer.evaluate_entry(row, self.attr, n_chg, vix):
                    target_limit_price = AdvancedStrategyAnalyzer.calculate_limit_price(row, self.attr, n_chg)
                    pending_entry = True

            else:
                high_p = max(high_p, curr_c)
                current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
                ch = max(high_p - (current_atr * atr_mult), entry_p - (current_atr * atr_mult))
                
                score = 0
                if bool(row.get('bb_3_reversal', False)): score += 40
                if curr_c < ch: score += 100
                
                if current_atr > 0 and curr_c > entry_p:
                    r_mult = (curr_c - entry_p) / (current_atr * 2)
                    if "中長期" not in self.attr and r_mult >= 3.0:
                        score += 100

                if self.attr in ["スイング", "押し目"]:
                    if bool(row.get('bb_p1_cross_down', False)): score += 20
                    if AdvancedStrategyAnalyzer._to_float(row.get('ma5', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('ma25', 0)) and AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio', 0)) >= 1.0: score += 15
                    if AdvancedStrategyAnalyzer._to_float(row.get('macd', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('sig', 0)) and AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio', 0)) >= 1.0: score += 15
                elif "中長期" in self.attr:
                    if AdvancedStrategyAnalyzer._to_float(row.get('rsi', 0)) > 85: score += 10
                
                if AdvancedStrategyAnalyzer._to_float(row.get('rs', 0)) < -5: score += 5

                if score >= 80:
                    cash += pos * curr_c
                    trades.append((curr_c - entry_p) / entry_p)
                    pos = 0

            equity.append(cash + (pos * curr_c if pos > 0 else 0))

        final = equity[-1] if equity else self.cash
        mdd_series = (pd.Series(equity) - pd.Series(equity).cummax()) / pd.Series(equity).cummax()
        mdd = float(mdd_series.min()) if not mdd_series.empty and not pd.isna(mdd_series.min()) else 0.0
        
        return {"Ticker": self.ticker, "Return": f"{(final-self.cash)/self.cash:.2%}", "MDD": f"{mdd:.2%}", "Trades": len(trades)}

# ==========================================
# 3. 堅牢性テスト & メイン実行
# ==========================================
def run_integrity_tests() -> None:
    print("[TEST] Running integrity and edge-case tests...")
    
    empty_df = pd.DataFrame()
    res_df = AdvancedStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    dummy_row_limit = pd.Series({'close': 1000.0, 'atr': 30.0})
    limit_p = AdvancedStrategyAnalyzer.calculate_limit_price(dummy_row_limit, "スイング", 0.0)
    assert limit_p == 991.0, f"Limit price calculation failed. Expected 991.0, got {limit_p}"
    
    limit_p_gap = AdvancedStrategyAnalyzer.calculate_limit_price(dummy_row_limit, "スイング", -1.0)
    assert limit_p_gap == 981.0, f"Hybrid limit price calculation failed. Expected 981.0, got {limit_p_gap}"

    # 【検証】異常な文字列やNaNが混入してもクラッシュしないことの証明
    dummy_row_err = pd.Series({'rsi': np.nan, 'dev25': 'invalid', 'rs_21': None})
    try:
        res = AdvancedStrategyAnalyzer.evaluate_entry(dummy_row_err, "スイング", 0.0, 15.0)
        assert isinstance(res, bool), "evaluate_entry must return bool even with corrupted data"
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data: {e}")
        
    print("[TEST] All tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        tester = IntegratedBacktester("7203")
        res = tester.run()
        print(f"[RESULT] {res}")
        pd.DataFrame([res]).to_csv("backtest_report.csv", index=False)
    except Exception as e:
        print(f"[FATAL] {e}")
