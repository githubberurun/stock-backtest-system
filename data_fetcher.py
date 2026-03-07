import os
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Final, Any, Union
from datetime import datetime

# 最新公式ドキュメント: https://jpx-jquants.com/ja/spec/quickstart
DATA_URL: Final[str] = "https://api.jquants.com/v2/daily_quotes"

class JQuantsV2Fetcher:
    """
    J-Quants API v2 (APIキー認証) 対応の堅牢なデータ取得・分析クラス
    """
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("JQUANTS_API_KEY が未設定です。GitHub Secretsを確認してください。")
        
        # 前後の空白や改行を除去（認証エラー防止）
        self.api_key: str = str(api_key).strip()
        self.headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "Accept": "application/json"
        }

    def _format_ticker(self, ticker: str) -> str:
        """4桁コードをV2仕様の5桁に変換 (例: 7203 -> 72030)"""
        if len(ticker) == 4 and ticker.isdigit():
            return f"{ticker}0"
        return ticker

    def _format_date(self, date_str: str) -> str:
        """YYYYMMDD形式をハイフン付き形式に変換 (例: 20160101 -> 2016-01-01)"""
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    def fetch_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """APIキーを使用して10年分のデータを取得"""
        code = self._format_ticker(ticker)
        d_from = self._format_date(start)
        d_to = self._format_date(end)

        all_quotes: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        print(f"[DEBUG] Requesting: Code={code}, From={d_from}, To={d_to}")

        while True:
            params = {"code": code, "from": d_from, "to": d_to}
            if pagination_key:
                params["pagination_key"] = pagination_key

            response = requests.get(DATA_URL, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 403:
                # 2026年仕様の403は「キーの有効性」か「パラメータ形式不備」が主因
                detail = response.json().get("message", "Unknown Permission Error")
                print(f"[ERROR] 403 Forbidden: {detail}")
                print("[HINT] J-Quants管理画面で APIキーに 'Daily Quotes' 権限があるか確認してください。")
                raise PermissionError(f"認証エラー: {detail}")
            
            response.raise_for_status()
            data = response.json()
            quotes = data.get("daily_quotes", [])
            all_quotes.extend(quotes)

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.5)

        df = pd.DataFrame(all_quotes)
        return self._preprocess(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """型変換とクレンジング（デバッグ機能付）"""
        if df.empty:
            return pd.DataFrame()
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"[DEBUG] Processed {len(df)} rows. Columns: {df.columns.tolist()}")
        return df

    # --- 過去の分析指標の継承 ---
    def calculate_backtest_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """シャープレシオ等の分析指標を算出 (LaTeX: $$SR = \frac{E[R_p - R_f]}{\sigma_p}$$)"""
        if df.empty or 'Close' not in df.columns:
            return {"sharpe": 0.0, "mdd": 0.0}
        
        returns = df['Close'].pct_change().dropna()
        excess_returns = returns - (0.01 / 252) # リスクフリーレート1%
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        mdd = ((cum_ret - peak) / peak).min()
        
        return {"sharpe": float(sharpe), "mdd": float(mdd)}

def run_main():
    api_key = os.getenv("JQUANTS_API_KEY")
    ticker = "7203" # トヨタ自動車
    
    fetcher = JQuantsV2Fetcher(api_key)
    try:
        # 10年分の取得を試行
        df = fetcher.fetch_historical_data(ticker, "20160101", "20260307")
        
        if not df.empty:
            metrics = fetcher.calculate_backtest_metrics(df)
            print(f"[SUCCESS] Sharpe Ratio: {metrics['sharpe']:.2f}, MDD: {metrics['mdd']:.2%}")
            
            os.makedirs("data", exist_ok=True)
            df.to_parquet(f"data/{ticker}.parquet", index=False)
        else:
            print("[WARNING] Data is empty.")
            exit(1)
            
    except Exception as e:
        print(f"[FATAL] {e}")
        exit(1)

# --- 堅牢性テスト（assert文による証明） ---
def test_robustness():
    print("\n--- Running Safety Tests ---")
    fetcher = JQuantsV2Fetcher("dummy_key")
    
    # 1. コード変換テスト
    assert fetcher._format_ticker("7203") == "72030", "5桁変換失敗"
    # 2. 日付変換テスト
    assert fetcher._format_date("20260307") == "2026-03-07", "日付ハイフン化失敗"
    # 3. 空データ処理
    assert fetcher._preprocess(pd.DataFrame()).empty, "空DFの処理失敗"
    
    print("All Safety Tests Passed.")

if __name__ == "__main__":
    test_robustness()
    run_main()
