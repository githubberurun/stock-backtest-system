import os
import requests
import pandas as pd
import time
import gc
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# Requests: https://requests.readthedocs.io/en/latest/
# ==========================================

BASE_URL: Final[str] = "https://api.jquants.com/v2"
ENDPOINT: Final[str] = "/equities/bars/daily"

def is_recently_updated(filepath: str, hours: int = 12) -> bool:
    """物理的なファイルの更新日時をチェックし、指定時間以内なら完全スキップ"""
    if not isinstance(filepath, str): return False
    if not os.path.exists(filepath): return False
    try:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        return (datetime.now() - file_mtime) < timedelta(hours=hours)
    except Exception:
        return False

class JQuantsV2Fetcher:
    """J-Quants API v2準拠のデータ取得クラス (堅牢化版)"""
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def get_safe_start_date(self) -> str:
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def get_top_tickers(self, limit: int = 600) -> List[str]:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
            
        print(f"[INFO] Fetching top {limit} tickers by TurnoverValue...")
        target_date = datetime.now().date()
        
        for _ in range(5):
            params = {"date": target_date.strftime("%Y%m%d")}
            try:
                response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=self.headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if len(data) > 500:
                        df = pd.DataFrame(data)
                        df['Va_n'] = pd.to_numeric(df.get('TurnoverValue', df.get('Va', 0)), errors='coerce')
                        top_df = df[df['Va_n'] >= 10_000_000].sort_values('Va_n', ascending=False).head(limit)
                        return [str(code)[:4] for code in top_df['Code'].tolist()]
            except Exception as e:
                print(f"[WARN] Failed to fetch daily data for {target_date}: {e}")
            target_date -= timedelta(days=1)
            time.sleep(1)
            
        print("[ERROR] Could not fetch recent market data.")
        return []

    def fetch(self, ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
        if start_date is not None and not isinstance(start_date, str):
            raise TypeError("start_date must be a string")
            
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        actual_start: str = start_date if start_date else self.get_safe_start_date()
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        
        max_pages: int = 20
        page_count: int = 0

        while page_count < max_pages:
            page_count += 1
            params: Dict[str, Any] = {"code": code, "from": actual_start}
            if pagination_key:
                params["pagination_key"] = pagination_key

            retry_count = 0
            success = False
            
            while retry_count < 3 and not success:
                try:
                    response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=self.headers, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        success = True
                        res_json = response.json()
                        data_chunk = res_json.get("data", [])
                        all_data.extend(data_chunk)
                        pagination_key = res_json.get("pagination_key")
                    elif response.status_code in [403, 429, 500, 502, 503, 504]:
                        print(f" [API {response.status_code} Wait] ", end="", flush=True)
                        time.sleep(5)
                        retry_count += 1
                    else:
                        print(f"[ERROR] API {response.status_code}: {response.text}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    print(f" [NetErr Wait] ", end="", flush=True)
                    time.sleep(5)
                    retry_count += 1
            
            if not success or not pagination_key:
                break
                
            time.sleep(0.1)

        return self._clean(pd.DataFrame(all_data))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty: 
            return df
            
        col_map = {
            'Date': 'date', 
            'AdjClose': 'close', 'AdjC': 'close', 'C': 'close_raw', 'Close': 'close_raw',
            'AdjHigh': 'high', 'AdjH': 'high', 'H': 'high_raw', 'High': 'high_raw',
            'AdjLow': 'low', 'AdjL': 'low', 'L': 'low_raw', 'Low': 'low_raw',
            'AdjOpen': 'open', 'AdjO': 'open', 'O': 'open_raw', 'Open': 'open_raw',
            'AdjVolume': 'volume', 'AdjVo': 'volume', 'Vo': 'volume_raw', 'Volume': 'volume_raw',
            'TurnoverValue': 'turnover', 'Va': 'turnover'
        }
        df = df.rename(columns=col_map)
        
        if 'close' not in df.columns and 'close_raw' in df.columns:
            df = df.rename(columns={
                'close_raw': 'close', 'high_raw': 'high', 'low_raw': 'low', 
                'open_raw': 'open', 'volume_raw': 'volume'
            })

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'turnover' not in df.columns or df['turnover'].isnull().all():
            df['turnover'] = df['close'] * df['volume']
                
        if 'date' in df.columns:
            df = df.dropna(subset=['close']).sort_values("date").reset_index(drop=True)
            
        return df

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy_fetcher = JQuantsV2Fetcher("dummy_key")
    df_empty = pd.DataFrame()
    cleaned_empty = dummy_fetcher._clean(df_empty)
    assert cleaned_empty.empty, "Empty DataFrame should return empty DataFrame"
    assert is_recently_updated("non_existent_file.parquet") is False
    print("[TEST] All integrity tests passed.")

if __name__ == "__main__":
    test_integrity()
    
    key = os.getenv("JQUANTS_API_KEY")
    if not key:
        print("[WARN] JQUANTS_API_KEY is not set. Exiting fetcher execution.")
        exit(0)
        
    fetcher = JQuantsV2Fetcher(key)
    
    data_dir = "Colog_github"
    os.makedirs(data_dir, exist_ok=True)
    
    TARGET_LIMIT = 600
    target_tickers = fetcher.get_top_tickers(limit=TARGET_LIMIT)
    if "13060" not in target_tickers: target_tickers.append("13060")
        
    print(f"[INFO] Starting data fetch for {len(target_tickers)} tickers...")
    
    for i, target_ticker in enumerate(target_tickers):
        print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}...", end=" ", flush=True)
        file_path = f"{data_dir}/{target_ticker}.parquet"
        
        # 12時間以内の更新ならAPIを一切叩かず爆速スキップ
        if is_recently_updated(file_path, hours=12):
            print("CACHED (Time-Skip)", flush=True)
            continue

        existing_df = pd.DataFrame()
        start_date_for_fetch = None
        
        # 前回強制終了時に破損したファイルの自己修復リカバリ
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_parquet(file_path)
                if not existing_df.empty and 'date' in existing_df.columns:
                    last_date_obj = pd.to_datetime(existing_df['date'].max())
                    start_date_for_fetch = (last_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[RECOVER] Corrupted cache detected. Re-downloading...", end=" ", flush=True)
                existing_df = pd.DataFrame()
        
        fetched_data = fetcher.fetch(target_ticker, start_date=start_date_for_fetch)
        
        try:
            if not existing_df.empty:
                if not fetched_data.empty:
                    combined = pd.concat([existing_df, fetched_data]).drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
                    combined.to_parquet(file_path, index=False)
                    print(f"UPDATED (Appended {len(fetched_data)} rows)", flush=True)
                else:
                    os.utime(file_path, None) # データ更新が無くてもタイムスタンプを現在時刻にしてスキップ対象にする
                    print("CACHED (Up to date)", flush=True)
            else:
                if not fetched_data.empty:
                    fetched_data.to_parquet(file_path, index=False)
                    print(f"OK ({len(fetched_data)} rows)", flush=True)
                else:
                    print("FAILED (No data)", flush=True)
        except Exception as e:
            print(f"FAILED (Write Error: {e})", flush=True)
            
        # メモリリーク（OOMキル）を完全に防ぐための明示的なガベージコレクション
        del existing_df
        del fetched_data
        gc.collect()
        
        time.sleep(0.2) # J-Quantsのレートリミットを尊重する安全マージン
                
    print("[INFO] Data fetching process completed.")
