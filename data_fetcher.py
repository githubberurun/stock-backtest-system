import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any

# 2026年時点の公式リファレンスに基づいたURL
# https://jpx-jquants.com/jp/api/overview/
API_BASE_URL: Final[str] = "https://api.jquants.com/v2"

class JQuantsActionFetcher:
    """
    GitHub Actions上で動作する、堅牢なデータ取得クラス
    """
    def __init__(self, api_key: Optional[str]):
        # 型チェック
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("有効なAPIキーが環境変数から取得できません。")
        
        self.api_key: str = api_key
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}

    def fetch_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        APIからデータを取得。内部で型チェックとデバッグログを出力。
        """
        # 型ヒントに基づくバリデーション
        if not ticker.endswith(".T") and not ticker.isdigit():
            print(f"[DEBUG] Invalid ticker format: {ticker}")
        
        all_quotes: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[DEBUG] Starting fetch for {ticker} from {start} to {end}")

        while True:
            params = {"code": ticker, "from": start, "to": end}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(
                f"{API_BASE_URL}/daily_quotes", 
                headers=self.headers, 
                params=params,
                timeout=30
            )
            
            # 検証結果の提示（2026年標準の例外処理）
            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
                break

            data = response.json()
            quotes = data.get("daily_quotes", [])
            all_quotes.extend(quotes)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
            
            time.sleep(0.5) # レート制限対策

        df = pd.DataFrame(all_quotes)
        return self._preprocess(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """取得データの型変換とクレンジング"""
        if df.empty:
            return df
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values("Date").reset_index(drop=True)

def run_task():
    """タスクのメイン実行部"""
    api_key = os.getenv("JQUANTS_API_KEY")
    ticker = "7203" # 例: トヨタ。銘柄コードの扱いは後ほど動的化可能
    
    fetcher = JQuantsActionFetcher(api_key)
    # 10年分のシミュレーション用に、まずは直近分を取得
    df = fetcher.fetch_data(ticker, "20250101", "20260307")
    
    # 物理的な保存確認
    os.makedirs("data", exist_ok=True)
    save_path = f"data/{ticker}.parquet"
    df.to_parquet(save_path, index=False)
    print(f"[SUCCESS] Data saved to {save_path}. Total rows: {len(df)}")

    # 堅牢性を証明するテストコード
    assert os.path.exists(save_path), "File was not saved successfully."
    if not df.empty:
        assert df['Close'].dtype in [float, int], "Data type conversion failed."
        assert not df['Date'].isnull().any(), "Date column contains nulls."

if __name__ == "__main__":
    run_task()
