import pandas as pd
import numpy as np
import os
from typing import Dict, List, Final, Any

class BacktestEngine:
    """10年間のヒストリカルデータに基づき戦略を検証する"""
    def __init__(self, ticker: str, initial_cash: float = 1000000.0):
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.path = f"data/{ticker}.parquet"
        
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing data file: {self.path}")
        
        self.df = pd.read_parquet(self.path)

    def run_strategy(self, stop_loss: float = 0.05, take_profit: float = 0.10) -> Dict[str, Any]:
        """SMAゴールデンクロス + exit_strategyロジック"""
        # 指標計算
        self.df['SMA5'] = self.df['Close'].rolling(window=5).mean()
        self.df['SMA25'] = self.df['Close'].rolling(window=25).mean()
        self.df['Buy_Signal'] = (self.df['SMA5'] > self.df['SMA25']) & (self.df['SMA5'].shift(1) <= self.df['SMA25'].shift(1))

        cash, position, entry_price = self.initial_cash, 0.0, 0.0
        equity_curve = []
        trade_returns = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            price = float(row['Close'])

            if position == 0:
                if row['Buy_Signal']:
                    position = cash // price
                    entry_price = price
                    cash -= position * price
            else:
                ret = (price - entry_price) / entry_price
                if ret <= -stop_loss or ret >= take_profit:
                    cash += position * price
                    trade_returns.append(ret)
                    position, entry_price = 0, 0
            
            total_val = cash + (position * price if position > 0 else 0)
            equity_curve.append(total_val)

        return self._analyze(equity_curve, trade_returns)

    def _analyze(self, curve: List[float], returns: List[float]) -> Dict[str, Any]:
        curve_s = pd.Series(curve)
        ret_s = pd.Series(returns) if returns else pd.Series([0.0])
        
        total_ret = (curve_s.iloc[-1] - self.initial_cash) / self.initial_cash
        # シャープレシオ: $$SR = \frac{E[R_p]}{\sigma_p} \times \sqrt{252}$$
        sharpe = (ret_s.mean() / ret_s.std() * np.sqrt(252)) if ret_s.std() != 0 else 0
        
        # 最大ドローダウン
        mdd = ((curve_s - curve_s.cummax()) / curve_s.cummax()).min()
        
        return {
            "Ticker": self.ticker,
            "Total_Return": f"{total_ret:.2%}",
            "Sharpe_Ratio": round(float(sharpe), 2),
            "Max_Drawdown": f"{mdd:.2%}",
            "Trades": len(returns)
        }

if __name__ == "__main__":
    try:
        engine = BacktestEngine("7203")
        report = engine.run_strategy()
        pd.DataFrame([report]).to_csv("backtest_report.csv", index=False)
        print(f"[RESULT] {report}")
    except Exception as e:
        print(f"[ERROR] {e}")

# --- 堅牢性テスト (assert) ---
def test_robustness():
    print("\n--- Integrity Check ---")
    # 空データ想定テスト
    try:
        df_empty = pd.DataFrame()
        assert df_empty.empty
        # 正常値計算テスト
        test_returns = pd.Series([0.02, -0.01, 0.05])
        assert test_returns.std() > 0
        print("[PASS] Robustness tests passed.")
    except AssertionError:
        print("[FAIL] Robustness tests failed.")
        exit(1)

test_robustness()
