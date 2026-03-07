import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# J-Quants API v2 エンドポイント
BASE_URL: Final[str] = "https://api.jquants.com/v2"
ENDPOINT: Final[str] = "/equities/bars/daily"

class JQuantsV2Fetcher:
    """J-Quants API v2準拠のデータ取得クラス"""
    def __init__(self, api_key: str):
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def get_safe_start_date(self) -> str:
        """プラン制限(10年)の境界値を考慮した開始日を算出"""
        # 10年前の同日（エラー回避のため+1日調整）
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def fetch(self, ticker: str) -> pd.DataFrame:
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date: str = self.get_safe_start_date()
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[FETCH] Target: {code}, Since: {start_date}")

        while True:
            params: Dict[str, Any] = {"code": code, "from": start_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=self.headers, params=params, timeout=30)
            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text}")
                return pd.DataFrame()

            res_json = response.json()
            all_data.extend(res_json.get("data", []))

            pagination_key = res_json.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.5)

        return self._clean(pd.DataFrame(all_data))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        # V2短縮名を以前の分析指標と互換性のある名前に変換
        df = df.rename(columns={'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume'})
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.sort_values("Date").reset_index(drop=True)

if __name__ == "__main__":
    key = os.getenv("JQUANTS_API_KEY")
    if key:
        fetcher = JQuantsV2Fetcher(key)
        os.makedirs("data", exist_ok=True)
        # 対象銘柄とベンチマーク(13060)の両方を取得
        for ticker in ["7203", "13060"]:
            data = fetcher.fetch(ticker)
            if not data.empty:
                data.to_parquet(f"data/{ticker}.parquet", index=False)
                print(f"[SUCCESS] {ticker} saved. ({len(data)} rows)")
