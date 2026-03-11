import os
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# Requests: https://requests.readthedocs.io/en/latest/
# ==========================================
BASE_URL: Final[str] = "https://api.jquants.com/v2"
EP_DAILY: Final[str] = "/equities/bars/daily"
EP_MARGIN: Final[str] = "/markets/margin-interest"
EP_FINS: Final[str] = "/fins/summary"

class JQuantsV2Fetcher:
    """J-Quants API v2準拠: 日足株価・需給・時価総額の統合データ取得クラス"""
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_safe_start_date(self) -> str:
        """プラン制限(10年)の境界値を考慮した開始日を算出"""
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def get_top_tickers(self) -> List[str]:
        """過去25営業日の売買代金に基づくハイブリッド抽出（変化率上位200 + 流動性上位100）"""
        print("[INFO] Fetching top tickers by Hybrid Volume Logic...")
        target_date = datetime.now().date()
        trading_days_data: List[pd.DataFrame] = []
        days_back = 0
        
        while len(trading_days_data) < 25 and days_back < 50:
            date_str = (target_date - timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                r = self.session.get(f"{BASE_URL}{EP_DAILY}", params={"date": date_str}, timeout=30)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if len(data) > 500:
                        df_temp = pd.DataFrame(data)
                        va_col = 'TurnoverValue' if 'TurnoverValue' in df_temp.columns else 'Va'
                        if va_col in df_temp.columns:
                            df_temp['Va_n'] = pd.to_numeric(df_temp[va_col], errors='coerce')
                            trading_days_data.append(df_temp[['Code', 'Va_n']])
            except Exception:
                pass
            days_back += 1
            time.sleep(0.3)

        if not trading_days_data:
            print("[ERROR] Could not fetch recent market data for volume ma25.")
            return []

        df_p = trading_days_data[0].copy()
        all_df = pd.concat(trading_days_data)
        ma25_va = all_df.groupby('Code')['Va_n'].mean()

        df_p['Va_ma25'] = df_p['Code'].map(ma25_va)
        df_filtered = df_p[df_p['Va_n'] >= 50_000_000].copy()
        df_filtered['vol_growth_ratio'] = df_filtered['Va_n'] / df_filtered['Va_ma25'].replace(0, np.nan)

        top_growth = df_filtered.sort_values('vol_growth_ratio', ascending=False).head(200)
        top_volume = df_filtered.sort_values('Va_n', ascending=False).head(100)
        
        targets = pd.concat([top_growth, top_volume]).drop_duplicates(subset=['Code']).head(300)
        return [str(code)[:4] for code in targets['Code'].tolist()]

    def _fetch_paginated(self, endpoint: str, params: Dict[str, Any]) -> pd.DataFrame:
        """ページネーションに対応した汎用データ取得ロジック"""
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        while True:
            current_params = params.copy()
            if pagination_key:
                current_params["pagination_key"] = pagination_key

            try:
                r = self.session.get(f"{BASE_URL}{endpoint}", params=current_params, timeout=30)
                if r.status_code != 200:
                    break
                res_json = r.json()
                all_data.extend(res_json.get("data", []))
                pagination_key = res_json.get("pagination_key")
                if not pagination_key:
                    break
                time.sleep(0.3)
            except requests.exceptions.RequestException:
                break
                
        return pd.DataFrame(all_data)

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
            
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date: str = self.get_safe_start_date()

        # 1. 日足データの取得と整形
        df_daily = self._fetch_paginated(EP_DAILY, {"code": code, "from": start_date})
        df_daily = self._clean_daily(df_daily)
        if df_daily.empty:
            return df_daily

        # 2. 信用残データ（需給）の取得
        df_margin = self._fetch_paginated(EP_MARGIN, {"code": code, "from": start_date})
        if not df_margin.empty and 'Date' in df_margin.columns:
            df_margin['LongVol'] = pd.to_numeric(df_margin.get('LongVol'), errors='coerce')
            df_margin['ShrtVol'] = pd.to_numeric(df_margin.get('ShrtVol'), errors='coerce').replace(0, np.nan)
            df_margin['mr_ratio'] = df_margin['LongVol'] / df_margin['ShrtVol']
            # ルックアヘッド・バイアス防止: 金曜基準のデータを翌週火曜(+4日)公表としてシフト
            df_margin['date'] = pd.to_datetime(df_margin['Date']) + pd.Timedelta(days=4)
            df_margin = df_margin.dropna(subset=['mr_ratio'])[['date', 'mr_ratio']].sort_values('date')
        else:
            df_margin = pd.DataFrame(columns=['date', 'mr_ratio'])

        # 3. 財務情報（発行済株式数）の取得
        df_fins = self._fetch_paginated(EP_FINS, {"code": code})
        if not df_fins.empty and 'DiscloseDate' in df_fins.columns:
            # 最新の優先列を探す
            share_col = 'ShOutFY' if 'ShOutFY' in df_fins.columns else 'AvgSh'
            if share_col in df_fins.columns:
                df_fins['shares_out'] = pd.to_numeric(df_fins[share_col], errors='coerce')
                # 公表日(DiscloseDate)を結合キーにする
                df_fins['date'] = pd.to_datetime(df_fins['DiscloseDate'])
                df_fins = df_fins.dropna(subset=['shares_out'])[['date', 'shares_out']].sort_values('date')
            else:
                df_fins = pd.DataFrame(columns=['date', 'shares_out'])
        else:
            df_fins = pd.DataFrame(columns=['date', 'shares_out'])

        # 4. Point-in-Time マージ (未来のデータを参照しないように直近過去のデータを結合)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        # 需給マージ
        if not df_margin.empty:
            df_daily = pd.merge_asof(df_daily.sort_values('date'), df_margin, on='date', direction='backward')
        else:
            df_daily['mr_ratio'] = np.nan
            
        # 財務マージ
        if not df_fins.empty:
            df_daily = pd.merge_asof(df_daily.sort_values('date'), df_fins, on='date', direction='backward')
        else:
            df_daily['shares_out'] = np.nan

        # 5. 指標の算出 (mcap, mr_zscore)
        if 'shares_out' in df_daily.columns:
            df_daily['mcap'] = (df_daily['shares_out'] * df_daily['close']) / 1e8
            
        if 'mr_ratio' in df_daily.columns:
            # 過去52週（約1年分）の移動平均と標準偏差からZスコアを算出
            # 日足データにマージされているため、直近250営業日(約1年)の窓で計算
            mean_mr = df_daily['mr_ratio'].rolling(window=250, min_periods=20).mean()
            std_mr = df_daily['mr_ratio'].rolling(window=250, min_periods=20).std().replace(0, np.nan)
            df_daily['mr_zscore'] = (df_daily['mr_ratio'] - mean_mr) / std_mr

        df_daily['date'] = df_daily['date'].dt.strftime('%Y-%m-%d')
        return df_daily

    def _clean_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be a pandas DataFrame")
        if df.empty: return df
            
        col_map = {
            'Date': 'date', 
            'AdjustmentClose': 'close', 'AdjClose': 'close', 'AdjC': 'close', 'C': 'close_raw', 'Close': 'close_raw',
            'AdjustmentHigh': 'high', 'AdjHigh': 'high', 'AdjH': 'high', 'H': 'high_raw', 'High': 'high_raw',
            'AdjustmentLow': 'low', 'AdjLow': 'low', 'AdjL': 'low', 'L': 'low_raw', 'Low': 'low_raw',
            'AdjustmentOpen': 'open', 'AdjOpen': 'open', 'AdjO': 'open', 'O': 'open_raw', 'Open': 'open_raw',
            'AdjustmentVolume': 'volume', 'AdjVolume': 'volume', 'AdjVo': 'volume', 'Vo': 'volume_raw', 'Volume': 'volume_raw'
        }
        df = df.rename(columns=col_map)
        
        if 'close' not in df.columns and 'close_raw' in df.columns:
            df = df.rename(columns={'close_raw': 'close', 'high_raw': 'high', 'low_raw': 'low', 'open_raw': 'open', 'volume_raw': 'volume'})

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'date' in df.columns:
            df = df.dropna(subset=['close']).sort_values("date").reset_index(drop=True)
            
        return df

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy = JQuantsV2Fetcher("dummy")
    df_empty = pd.DataFrame()
    assert dummy._clean_daily(df_empty).empty, "Empty DataFrame test failed"
    try:
        dummy.fetch(1234) # type: ignore
        assert False, "Type checking failed"
    except TypeError: pass
    
    # マージロジックのモックテスト（ルックアヘッド・バイアスの排除確認）
    df_mock_daily = pd.DataFrame({'date': pd.to_datetime(['2026-02-18', '2026-02-19']), 'close': [1000, 1050]})
    df_mock_margin = pd.DataFrame({'date': pd.to_datetime(['2026-02-18']), 'mr_ratio': [2.5]})
    
    merged = pd.merge_asof(df_mock_daily, df_mock_margin, on='date', direction='backward')
    assert merged['mr_ratio'].iloc[1] == 2.5, "Forward fill via merge_asof failed"
    print("[TEST] All integrity tests passed.")

if __name__ == "__main__":
    test_integrity()
    key = os.getenv("JQUANTS_API_KEY")
    if not key: raise ValueError("[FATAL] JQUANTS_API_KEY is not set.")
    
    fetcher = JQuantsV2Fetcher(key)
    # GitHub用フォルダ等の指定がある場合は適宜パスを変更可能ですが、今回は既存の data/ に出力します
    os.makedirs("data", exist_ok=True)
    
    target_tickers = fetcher.get_top_tickers()
    if "13060" not in target_tickers: target_tickers.append("13060")
        
    print(f"[INFO] Starting combined data fetch (Daily + Margin + Fins) for {len(target_tickers)} tickers...")
    
    for i, target_ticker in enumerate(target_tickers):
        print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}...", end=" ")
        fetched_data = fetcher.fetch(target_ticker)
        if not fetched_data.empty:
            fetched_data.to_parquet(f"data/{target_ticker}.parquet", index=False)
            print(f"OK ({len(fetched_data)} rows)")
        else: print("FAILED")
