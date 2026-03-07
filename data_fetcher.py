import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any, Union

# 2026年最新公式ドキュメントに基づくエンドポイント
# https://jpx-jquants.com/ja/spec/quickstart
DATA_URL: Final[str] = "https://api.jquants.com/v2/daily_quotes"

class JQuantsV2Fetcher:
    """
    J-Quants API v2 (APIキー認証) を使用したデータ取得クラス
    """
    def __init__(self, api_key: Optional[str]):
        # 型チェックおよび未定義変数の防止
        if not api_key:
            raise ValueError("環境変数 JQUANTS_API_KEY が未設定です。")
        
        self.api_key: str = str(api_key)
        # V2仕様: x-api-keyヘッダーを使用
        self.headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "Accept": "application/json"
        }

    def fetch_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        APIキーを直接使用してヒストリカルデータを取得。
        """
        all_quotes: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[DEBUG] Fetching {ticker} from {start} to {end} using V2 API Key.")

        while True:
            params: Dict[str, Any] = {
                "code": ticker,
                "from": start,
                "to": end
            }
            if pagination_key:
                params["pagination_key"] = pagination_key

            # リクエスト実行とエラーハンドリング
            try:
                response = requests.get(
                    DATA_URL, 
                    headers=self.headers, 
                    params=params, 
                    timeout=30
                )
                
                # 2026年標準のステータスコード判定
                if response.status_code == 403:
                    raise PermissionError("403 Forbidden: APIキーが無効、またはプランの権限外です。")
                elif response.status_code == 429:
                    print("[WARNING] Rate limit reached. Waiting 60s...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] API Request Failed: {e}")
                break

            quotes = data.get("daily_quotes", [])
            all_quotes.extend(quotes)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
            
            time.sleep(0.5) # レート制限（1秒2リクエスト程度）に配慮

        df = pd.DataFrame(all_quotes)
        return self._preprocess_and_validate(df)

    def _preprocess_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """型変換とデータの整合性チェック"""
        if df.empty:
            return pd.DataFrame()

        # 数値型への変換（J-Quants V2のデータ型を保証）
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 日付順にソート
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 内部での型チェックと異常値除外
        df = df.dropna(subset=['Close'])
        return df

def run_github_action():
    """GitHub Actions 実行用メイン関数"""
    api_key = os.getenv("JQUANTS_API_KEY")
    ticker = "7203" # トヨタ自動車（例）
    
    fetcher = JQuantsV2Fetcher(api_key)
    
    try:
        # 過去10年のシミュレーションを見据え、指定範囲で取得
        df = fetcher.fetch_historical_data(ticker, "20160101", "20260307")
        
        if df.empty:
            print("[CRITICAL] No data fetched. Job Failed.")
            exit(1)

        # 物理的な保存パスの確認と作成
        save_dir = "data"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{ticker}.parquet")
        
        # Parquet形式で保存（バックテストの高速化用）
        df.to_parquet(save_path, index=False)
        
        # 堅牢性を証明するテストコード (assert文)
        assert len(df) > 0, "Error: Dataframe length is 0."
        assert 'Close' in df.columns, "Error: Missing Close price column."
        assert not df['Close'].isnull().any(), "Error: Data contains NaN in Close column."
        assert os.path.exists(save_path), f"Error: File not found at {save_path}"
        
        print(f"[SUCCESS] {len(df)} rows of data saved to {save_path}.")

    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")
        exit(1)

if __name__ == "__main__":
    run_github_action()
