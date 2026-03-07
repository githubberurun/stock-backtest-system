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
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def get_safe_start_date(self) -> str:
        """プラン制限(10年)の境界値を考慮した開始日を算出"""
        # 10年前の同日（エラー回避のため+1日調整）
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
            
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date: str = self.get_safe_start_date()
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[FETCH] Target: {code}, Since: {start_date}")

        while True:
            params: Dict[str, Any] = {"code": code, "from": start_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=self.headers, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error during fetch: {e}")
                return pd.DataFrame()

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
        if df.empty: 
            return df
            
        # 本番環境(exit_strategy.py)と同等のカラムマッピング（調整後株価を優先）
        col_map = {
            'Date': 'date', 
            'AdjClose': 'close', 'AdjC': 'close', 'C': 'close_raw', 'Close': 'close_raw',
            'AdjHigh': 'high', 'AdjH': 'high', 'H': 'high_raw', 'High': 'high_raw',
            'AdjLow': 'low', 'AdjL': 'low', 'L': 'low_raw', 'Low': 'low_raw',
            'AdjOpen': 'open', 'AdjO': 'open', 'O': 'open_raw', 'Open': 'open_raw',
            'AdjVolume': 'volume', 'AdjVo': 'volume', 'Vo': 'volume_raw', 'Volume': 'volume_raw'
        }
        df = df.rename(columns=col_map)
        
        # 調整後株価が存在しない場合のフォールバック処理
        if 'close' not in df.columns and 'close_raw' in df.columns:
            df = df.rename(columns={
                'close_raw': 'close', 
                'high_raw': 'high', 
                'low_raw': 'low', 
                'open_raw': 'open', 
                'volume_raw': 'volume'
            })

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'date' in df.columns:
            df = df.dropna(subset=['close']).sort_values("date").reset_index(drop=True)
            
        return df

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy_fetcher = JQuantsV2Fetcher("dummy_key")
    
    # テスト1: 調整後株価(Adj)のマッピングテスト
    df_mock_adj = pd.DataFrame({
        'Date': ['2026-01-01'], 'AdjC': [150.5], 'AdjH': [155.0], 
        'AdjL': [149.0], 'AdjO': [150.0], 'AdjVo': [5000]
    })
    cleaned_adj = dummy_fetcher._clean(df_mock_adj)
    assert 'close' in cleaned_adj.columns, "AdjC should be mapped to 'close'"
    assert cleaned_adj['close'].iloc[0] == 150.5, "Value matching failed for AdjC"
    assert 'open' in cleaned_adj.columns and 'high' in cleaned_adj.columns, "Other OHLC missing"

    # テスト2: 生株価(Raw)のフォールバックテスト
    df_mock_raw = pd.DataFrame({
        'Date': ['2026-01-01'], 'C': [50.0], 'H': [55.0], 
        'L': [49.0], 'O': [50.0], 'Vo': [100]
    })
    cleaned_raw = dummy_fetcher._clean(df_mock_raw)
    assert 'close' in cleaned_raw.columns, "Raw 'C' should fallback to 'close'"
    assert cleaned_raw['close'].iloc[0] == 50.0, "Value matching failed for raw 'C'"

    # テスト3: 空データのクラッシュ防止
    df_empty = pd.DataFrame()
    cleaned_empty = dummy_fetcher._clean(df_empty)
    assert cleaned_empty.empty, "Empty DataFrame should return empty DataFrame"

    print("[TEST] All integrity tests passed.")

if __name__ == "__main__":
    test_integrity()
    
    key = os.getenv("JQUANTS_API_KEY")
    if not key:
        raise ValueError("[FATAL] JQUANTS_API_KEY is not set in environment variables.")
        
    fetcher = JQuantsV2Fetcher(key)
    os.makedirs("data", exist_ok=True)
    
    # 対象銘柄とベンチマーク(13060)の両方を取得
    for target_ticker in ["7203", "13060"]:
        fetched_data = fetcher.fetch(target_ticker)
        if not fetched_data.empty:
            fetched_data.to_parquet(f"data/{target_ticker}.parquet", index=False)
            print(f"[SUCCESS] {target_ticker} saved. ({len(fetched_data)} rows)")
        else:
            print(f"[WARN] Failed to fetch or clean data for {target_ticker}.")
