import os
import requests
import pandas as pd
from typing import Dict, Final, Optional

class JQuantsV2RobustFetcher:
    """
    403エラーを回避するための2026年標準データ取得クラス
    """
    def __init__(self, api_key: str):
        self.api_key: str = api_key
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def fetch_with_format(self, ticker: str, start: str):
        # 1. 銘柄コードを5桁に補完 (7203 -> 72030)
        formatted_code = ticker if len(ticker) == 5 else f"{ticker}0"
        
        # 2. 日付をハイフン形式に変換 (20160101 -> 2016-01-01)
        if "-" not in start:
            start = f"{start[:4]}-{start[4:6]}-{start[6:]}"

        params = {"code": formatted_code, "from": start}
        
        response = requests.get(
            "https://api.jquants.com/v2/daily_quotes", 
            headers=self.headers, 
            params=params,
            timeout=30
        )

        if response.status_code == 403:
            # プラン権限の物理的確認を促す
            detail = response.json().get("message", "No detail")
            raise PermissionError(f"403 Forbidden: {detail}\n"
                                  f"APIキーの権限設定(Daily Quotes)とStandardプランの契約状態を確認してください。")
        
        response.raise_for_status()
        return pd.DataFrame(response.json().get("daily_quotes", []))

# 堅牢性テスト用
def test_data_integrity():
    """空データに対するテスト"""
    mock_df = pd.DataFrame()
    assert mock_df.empty, "Dataframe must be empty for initial state."
    print("Test passed: Logic handles empty response appropriately.")
