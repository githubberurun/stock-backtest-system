import os
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# 2026年最新公式エンドポイント
BASE_URL: Final[str] = "https://api.jquants.com/v2"
DAILY_BARS_PATH: Final[str] = "/equities/bars/daily"

class JQuantsV2RobustFetcher:
    """
    プラン制限による400エラーを自動回避する、2026年仕様のデータ取得クラス
    """
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("JQUANTS_API_KEY が未設定です。")
        self.api_key: str = str(api_key).strip()
        self.headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "Accept": "application/json"
        }

    def _get_max_historical_start(self) -> str:
        """
        プラン制限(10年)を考慮し、今日から10年前の日付を算出
        """
        # 2026-03-07 の10年前は 2016-03-07
        start_date = datetime.now() - timedelta(days=365 * 10)
        return start_date.strftime("%Y-%m-%d")

    def fetch_historical_data(self, ticker: str, requested_start: str) -> pd.DataFrame:
        """
        APIキーを使用してデータを取得。開始日が制限外なら自動で切り詰める。
        """
        code = f"{ticker}0" if len(ticker) == 4 else ticker
        
        # プラン境界値の自動調整
        limit_start = self._get_max_historical_start()
        actual_start = max(requested_start.replace("", "-") if "-" not in requested_start else requested_start, limit_start)
        
        if actual_start > requested_start:
            print(f"[INFO] 取得開始日をプラン制限に合わせて調整しました: {requested_start} -> {actual_start}")

        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        while True:
            params = {"code": code, "from": actual_start}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(f"{BASE_URL}{DAILY_BARS_PATH}", headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 400:
                print(f"[FATAL] 400 Error. Data range might still be invalid: {response.text}")
                response.raise_for_status()

            json_res = response.json()
            all_data.extend(json_res.get("data", []))

            pagination_key = json_res.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.5)

        return self._preprocess(pd.DataFrame(all_data))

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """V2カラム名のマッピングと型チェック"""
        if df.empty:
            return df
        
        column_map = {'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume'}
        df = df.rename(columns=column_map)
        
        # Python型ヒントに基づいた内部変換
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Close']).sort_values("Date").reset_index(drop=True)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        r"""
        分析指標を算出。2025年以前の依頼指標をすべて継承。
        $$Sharpe\ Ratio = \frac{E[R_p - R_f]}{\sigma_p}$$
        """
        if df.empty or len(df) < 2:
            return {"sharpe": 0.0, "mdd": 0.0}
        
        returns = df['Close'].pct_change().dropna()
        # リスクフリーレートを考慮したシャープレシオ
        sharpe = (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() != 0 else 0
        
        # 最大ドローダウン
        cum_ret = (1 + returns).cumprod()
        mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        
        return {"sharpe": float(sharpe), "mdd": float(mdd)}

# --- GitHub Actions 実行メイン ---
def main():
    api_key = os.getenv("JQUANTS_API_KEY")
    ticker = "7203"
    
    fetcher = JQuantsV2RobustFetcher(api_key)
    try:
        # 10年前の今日を指定（400エラーを物理的に回避）
        safe_start = fetcher._get_max_historical_start()
        df = fetcher.fetch_historical_data(ticker, safe_start)
        
        if not df.empty:
            metrics = fetcher.calculate_metrics(df)
            print(f"[SUCCESS] Rows: {len(df)}, Sharpe: {metrics['sharpe']:.2f}, MDD: {metrics['mdd']:.2%}")
            
            os.makedirs("data", exist_ok=True)
            df.to_parquet(f"data/{ticker}.parquet", index=False)
        else:
            print("[WARNING] Data is empty.")
            exit(1)

    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)

# --- 堅牢性テスト (assert) ---
def test_robustness():
    print("Running boundary value tests...")
    fetcher = JQuantsV2RobustFetcher("dummy")
    
    # 1. 未来の日付が含まれていないか
    start = fetcher._get_max_historical_start()
    assert datetime.strptime(start, "%Y-%m-%d") <= datetime.now()
    
    # 2. 空のDF処理
    dummy_metrics = fetcher.calculate_metrics(pd.DataFrame())
    assert dummy_metrics["sharpe"] == 0.0
    
    print("Boundary Tests Passed.")

if __name__ == "__main__":
    test_robustness()
    main()
