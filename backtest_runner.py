import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Final, Any
from datetime import datetime

# 共通設定
DATA_DIR: Final[str] = "data"
INITIAL_CASH: Final[float] = 1_000_000.0  # 初期資金100万円

class StrategyBacktester:
    """
    保存されたParquetデータを用いて10年間の戦略検証を行うクラス
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        
        # 物理的なファイル存在確認
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {self.file_path}")
        
        self.df: pd.DataFrame = pd.read_parquet(self.file_path)
        self._validate_and_prepare()

    def _validate_and_prepare(self) -> None:
        """データの整合性チェックとインジケータ計算"""
        if self.df.empty:
            raise ValueError("データが空です。")
            
        # 既存ツール(main.py)のロジックを継承したインジケータ計算
        # 例: 25日移動平均線 (SMA25)
        self.df['SMA25'] = self.df['Close'].rolling(window=25).mean()
        self.df['SMA5'] = self.df['Close'].rolling(window=5).mean()
        
        # 買いシグナルの定義 (例: ゴールデンクロス)
        self.df['Buy_Signal'] = (self.df['SMA5'] > self.df['SMA25']) & \
                                (self.df['SMA5'].shift(1) <= self.df['SMA25'].shift(1))

    def run(self, stop_loss: float = 0.05, take_profit: float = 0.10) -> Dict[str, Any]:
        """
        バックテストのメインループ
        """
        cash = INITIAL_CASH
        position = 0.0
        entry_price = 0.0
        history = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            current_price = float(row['Close'])
            date = row['Date']

            # ポジションを保有していない場合（エントリー判断）
            if position == 0:
                if row['Buy_Signal']:
                    position = cash // current_price
                    entry_price = current_price
                    cash -= position * current_price
                    history.append({"Date": date, "Action": "BUY", "Price": current_price, "Cash": cash})

            # ポジションを保有している場合（エグジット判断）
            else:
                profit_rate = (current_price - entry_price) / entry_price
                
                # エグジット条件（損切り・利確）
                if profit_rate <= -stop_loss or profit_rate >= take_profit:
                    cash += position * current_price
                    history.append({"Date": date, "Action": "SELL", "Price": current_price, "Cash": cash})
                    position = 0
                    entry_price = 0.0

        final_assets = cash + (position * self.df.iloc[-1]['Close'] if position > 0 else 0)
        return self._summarize(final_assets, history)

    def _summarize(self, final_assets: float, history: List[Dict]) -> Dict[str, Any]:
        """結果の要約"""
        total_return = (final_assets - INITIAL_CASH) / INITIAL_CASH
        return {
            "ticker": self.ticker,
            "final_assets": final_assets,
            "total_return": total_return,
            "trade_count": len(history) // 2
        }

def calculate_mdd(prices: pd.Series) -> float:
    r"""
    最大ドローダウンを算出
    $$MDD = \min\left(\frac{Value_t - Peak}{Peak}\right)$$
    """
    if prices.empty: return 0.0
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max
    return float(drawdown.min())

# --- 堅牢性テスト ---
def test_backtester_robustness():
    print("Running Backtester Safety Tests...")
    # ファイルがない場合のエラーハンドリング
    try:
        StrategyBacktester("9999")
    except FileNotFoundError:
        print("[OK] FileNotFoundError handled.")
    
    # MDD計算の検証
    test_prices = pd.Series([100, 110, 90, 120])
    mdd = calculate_mdd(test_prices)
    assert mdd < 0, "MDD should be negative."
    print("All Safety Tests Passed.")

if __name__ == "__main__":
    test_backtester_robustness()
    # 実際の運用例
    # runner = StrategyBacktester("7203")
    # result = runner.run()
    # print(result)
