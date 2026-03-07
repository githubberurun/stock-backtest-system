import os
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime

# 2026年最新公式エンドポイント
# https://jpx-jquants.com/ja/spec/eq-bars-daily
BASE_URL: Final[str] = "https://api.jquants.com/v2"
DAILY_BARS_PATH: Final[str] = "/equities/bars/daily"

class JQuantsV2Fetcher:
    """
    J-Quants API v2 (2026年仕様) に完全準拠したデータ取得クラス
    """
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("JQUANTS_API_KEY が未設定です。")
        self.api_key: str = str(api_key).strip()
        self.headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "Accept": "application/json"
        }

    def _format_ticker(self, ticker: str) -> str:
        """4桁コードをV2推奨の5桁に変換 (例: 7203 -> 72030)"""
        return f"{ticker}0" if len(ticker) == 4 else ticker

    def _format_date(self, date_str: str) -> str:
        """YYYYMMDD を ISO形式 (YYYY-MM-DD) に変換"""
        if len(date_str) == 8 and "-" not in date_str:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    def fetch_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """V2エンドポイントからヒストリカルデータを取得"""
        code = self._format_ticker(ticker)
        d_from = self._format_date(start)
        d_to = self._format_date(end)

        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[DEBUG] Fetching via V2: {BASE_URL}{DAILY_BARS_PATH} (Code: {code})")

        while True:
            params = {"code": code, "from": d_from, "to": d_to}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(f"{BASE_URL}{DAILY_BARS_PATH}", headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"[ERROR] API Response: {response.status_code} - {response.text}")
                response.raise_for_status()

            # V2では "data" キーの下に配列が入る
            json_res = response.json()
            batch = json_res.get("data", [])
            all_data.extend(batch)

            pagination_key = json_res.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.5)

        return self._preprocess(pd.DataFrame(all_data))

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """V2の短縮カラム名を正規化し、型変換を行う"""
        if df.empty:
            return df
        
        # V2短縮名から以前のロジックとの互換名へのマッピング
        # O: Open, H: High, L: Low, C: Close, Vo: Volume
        column_map = {
            'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume', 'Date': 'Date'
        }
        df = df.rename(columns=column_map)
        
        # 数値型へ変換
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Close']).sort_values("Date").reset_index(drop=True)
        return df

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        r"""分析指標を算出 (LaTeX: $$SR = \frac{E[R_p - R_f]}{\sigma_p}$$)"""
        if df.empty or len(df) < 2:
            return {"sharpe": 0.0, "mdd": 0.0}
        
        returns = df['Close'].pct_change().dropna()
        sharpe = (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() != 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        
        return {"sharpe": float(sharpe), "mdd": float(mdd)}

def main():
    api_key = os.getenv("JQUANTS_API_KEY")
    ticker = "7203"
    
    fetcher = JQuantsV2Fetcher(api_key)
    try:
        # 過去10年のデータを取得（V2 Premiumなら20年、Standardなら10年程度可能）
        df = fetcher.fetch_historical_data(ticker, "20160101", "20260307")
        
        if not df.empty:
            metrics = fetcher.calculate_metrics(df)
            print(f"[SUCCESS] Rows: {len(df)}, Sharpe: {metrics['sharpe']:.2f}, MDD: {metrics['mdd']:.2%}")
            
            os.makedirs("data", exist_ok=True)
            df.to_parquet(f"data/{ticker}.parquet", index=False)
        else:
            print("[WARNING] No data fetched for the specified period.")

    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)

if __name__ == "__main__":
    main()
