import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# 2026年最新エンドポイント
BASE_URL: Final[str] = "https://api.jquants.com/v2"
DAILY_BARS_PATH: Final[str] = "/equities/bars/daily"

class JQuantsV2Fetcher:
    """J-Quants API v2 (2026年仕様) 用のデータ取得クラス"""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("JQUANTS_API_KEY is not set.")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def _get_start_date_10y(self) -> str:
        """プラン制限の10年(3652日)を正確に計算"""
        # 400エラー回避のため、10年前の翌日から開始
        start = datetime.now() - timedelta(days=365 * 10 - 1)
        return start.strftime("%Y-%m-%d")

    def fetch(self, ticker: str) -> pd.DataFrame:
        """データを取得し、V2短縮カラムを正規化する"""
        code = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date = self._get_start_date_10y()
        
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[DEBUG] Fetching Code: {code}, From: {start_date}")

        while True:
            params = {"code": code, "from": start_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(f"{BASE_URL}{DAILY_BARS_PATH}", headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"[ERROR] API Response: {response.text}")
                response.raise_for_status()

            json_res = response.json()
            all_data.extend(json_res.get("data", []))

            pagination_key = json_res.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.5)

        return self._preprocess(pd.DataFrame(all_data))

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        # V2短縮名 (O, H, L, C, Vo) -> 標準名へのマッピング
        df = df.rename(columns={'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume'})
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.sort_values("Date").reset_index(drop=True)

if __name__ == "__main__":
    api_key = os.getenv("JQUANTS_API_KEY")
    fetcher = JQuantsV2Fetcher(api_key)
    try:
        data = fetcher.fetch("7203") # トヨタ
        if not data.empty:
            os.makedirs("data", exist_ok=True)
            data.to_parquet("data/7203.parquet", index=False)
            print(f"[SUCCESS] {len(data)} rows saved.")
    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)
