import pandas as pd
import numpy as np
import os
import yfinance as yf
import time
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# 公式ドキュメント参照URL
# J-Quants API v2: https://jpx-jquants.com/jp/api/overview/
# yfinance: https://github.com/ranaroussi/yfinance

# ==========================================
# 1. 指標・シグナル分析エンジン
# ==========================================
class AdvancedStrategyAnalyzer:
    """エントリー（ガードレール）とエグジット（シャンデリア）を統合管理"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
        if df.empty or len(df) < 20: return df
        
        # 物理クレンジング: カラム名を小文字に統一
        df.columns = [str(c).lower() for c in df.columns]
        
        # 基本指標
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        
        # ボリンジャーバンド (2σ, 3σ, +1σ)
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        
        # エグジット判定用
        df['prev_low'] = df['low'].shift(1)
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])

        if 'open' in df.columns:
            df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
            is_reversal = (df['close'] < df['prev_low']) | (df['close'] < df['open'])
            df['bb_3_reversal'] = df['was_above_bb_up_3'] & is_reversal
        else:
            df['bb_3_reversal'] = False 

        # MACD & RSI
        df['ema12'], df['ema26'] = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        
        # ATR & 出来高比
        tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=20).mean()
        df['ma25_vol'] = df['volume'].rolling(window=25).mean()
        df['vol_ratio'] = (df['volume'] / df['ma25_vol'].replace(0, np.nan)).fillna(0)

        # 相対力指数 (RS_21)
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            df = df.merge(benchmark_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm'))
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs_21'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row: pd.Series, attr: str, nasdaq_chg: float, vix: float) -> bool:
        """エントリー・ガードレール判定"""
        # 米国市場警戒
        if nasdaq_chg <= -2.0 or vix >= 20.0: return False
        
        # 物理計算: 総合スコア（AIニュース判定なし版）
        rsi_val, d25_val, rs_val = float(row.get('rsi', 50)), float(row.get('dev25', 0)), float(row.get('rs_21', 0))
        
        tech_penalty = 0.0
        if attr != "中長期(グロース)":
            if rsi_val > 80: tech_penalty += 20.0
            if d25_val > 20: tech_penalty += 15.0

        total_score = round((50.0 * 0.7) + (2 * 3) + (1 * 2) - tech_penalty, 1) # 基本点35+財務6+登場2

        if attr == "押し目": return total_score >= 80
        elif "中長期" in attr: return total_score >= 70
        else: return total_score >= 85 and rs_val > 0

# ==========================================
# 2. 米国市場キャッシュ & バックテスター
# ==========================================
class USMarketCache:
    """過去10年分の米国指標を保持"""
    def __init__(self):
        self.ndx = yf.Ticker("^IXIC").history(period="10y")['Close'].pct_change() * 100
        self.vix = yf.Ticker("^VIX").history(period="10y")['Close']
        self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
        self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')

    def get_market_state(self, date_str: str) -> Tuple[float, float]:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index: return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class IntegratedBacktester:
    def __init__(self, ticker: str, initial_cash: float = 1000000.0, attr: str = "スイング"):
        self.ticker, self.cash, self.attr = ticker, initial_cash, attr
        self.df = AdvancedStrategyAnalyzer.calculate_indicators(
            pd.read_parquet(f"data/{ticker}.parquet"),
            pd.read_parquet("data/13060.parquet") if os.path.exists("data/13060.parquet") else None
        )
        self.df['date_str'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')
        self.us_market = USMarketCache()

    def run(self) -> Dict[str, Any]:
        cash, pos, entry_p, high_p = self.cash, 0.0, 0.0, 0.0
        atr_mult = 3.0 if "中長期" in self.attr else (2.0 if self.attr == "押し目" else 2.5)
        trades = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            curr_p = float(row['close'])
            n_chg, vix = self.us_market.get_market_state(row['date_str'])

            if pos == 0:
                if AdvancedStrategyAnalyzer.evaluate_entry(row, self.attr, n_chg, vix):
                    pos, entry_p, high_p = cash // curr_p, curr_p, curr_p
                    cash -= pos * curr_p
            else:
                high_p = max(high_p, curr_p)
                # エグジットスコア判定
                ch = max(high_p - (float(row['atr']) * atr_mult), entry_p - (float(row['atr']) * atr_mult))
                
                # exit_strategyロジック: スコア80以上で撤退
                score = 0
                if bool(row.get('bb_3_reversal', False)): score += 40
                if curr_p < ch: score += 100
                if float(row.get('ma5', 0)) < float(row.get('ma25', 0)) and float(row.get('vol_ratio', 0)) >= 1.0: score += 15

                if score >= 80:
                    cash += pos * curr_p
                    trades.append((curr_p - entry_p) / entry_p)
                    pos = 0

        final = cash + (pos * self.df.iloc[-1]['close'] if pos > 0 else 0)
        return {"Ticker": self.ticker, "Return": f"{(final-self.cash)/self.cash:.2%}", "Trades": len(trades)}

# ==========================================
# 3. 堅牢性テスト
# ==========================================
def test_integrity():
    print("--- Running Robustness Tests ---")
    # 空データテスト
    assert AdvancedStrategyAnalyzer.calculate_indicators(pd.DataFrame()).empty
    # エントリーガードレール（NASDAQ -3%）
    mock = pd.Series({'rsi': 50, 'dev25': 0, 'rs_21': 5})
    assert AdvancedStrategyAnalyzer.evaluate_entry(mock, "スイング", -3.0, 15.0) == False, "Guardrail failed"
    print("--- Tests Passed ---")

if __name__ == "__main__":
    test_integrity()
    try:
        res = IntegratedBacktester("7203").run()
        print(f"Final Result: {res}")
        pd.DataFrame([res]).to_csv("backtest_report.csv", index=False)
    except Exception as e:
        print(f"Error: {e}")
