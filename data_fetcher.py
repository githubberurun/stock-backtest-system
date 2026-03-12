import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional, Final
from datetime import datetime, timedelta

# ==========================================
# 0. 環境設定・定数定義
# ==========================================
JQUANTS_API_KEY: Final[str] = os.environ.get('JQUANTS_API_KEY', '').strip()
DATA_DIR: Final[str] = "data"
BENCHMARK_TICKER: Final[str] = "1306"
TARGET_UNIVERSE_SIZE: Final[int] = 300  # 売買代金上位300銘柄（流動性フィルター）

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. ユニバース自動選定エンジン (J-Quants API V2)
# ==========================================
def fetch_jquants_data(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not isinstance(url, str): raise TypeError("url must be string")
    if not isinstance(headers, dict): raise TypeError("headers must be dict")
    
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return data if isinstance(data, list) else []
            elif r.status_code == 429:
                time.sleep(10)
        except Exception as e:
            debug_log(f"J-Quants API Request Error: {e}")
            time.sleep(2)
    return []

def get_liquid_prime_tickers(api_key: str, limit: int = 300) -> List[str]:
    """プライム市場の中から、直近の売買代金上位銘柄を自動抽出する"""
    if not isinstance(api_key, str): raise TypeError("api_key must be string")
    if not isinstance(limit, int): raise TypeError("limit must be int")
    
    # APIキーがない場合のフェイルセーフ（TOPIX Core30などの代表銘柄群で代用）
    if not api_key:
        debug_log("⚠️ JQUANTS_API_KEYが設定されていません。フォールバックとして代表的な大型株リストを返します。")
        return ["7203", "8306", "9984", "6861", "8035", "9432", "6758", "4063", "8058", "8316"]

    debug_log("🔍 J-Quants APIから市場マスタを取得し、ユニバースを構築中...")
    headers = {"x-api-key": api_key}
    base_url = "https://api.jquants.com/v2"
    
    # 1. 市場マスタの取得とプライム銘柄の抽出
    master_data = fetch_jquants_data(f"{base_url}/equities/master", headers)
    if not master_data:
        debug_log("⚠️ マスタ取得失敗。フォールバックリストを返します。")
        return ["7203", "8306", "9984"]
        
    df_master = pd.DataFrame(master_data)
    prime_codes = [str(row['Code'])[:4] for _, row in df_master.iterrows() if row.get('MktNm') == 'プライム']
    
    # 2. 直近日足データから売買代金を取得
    target_date = datetime.now()
    df_bars = pd.DataFrame()
    for _ in range(5):
        date_str = target_date.strftime("%Y%m%d")
        bar_data = fetch_jquants_data(f"{base_url}/equities/bars/daily", headers, {"date": date_str})
        if bar_data and len(bar_data) > 500:
            df_bars = pd.DataFrame(bar_data)
            break
        target_date -= timedelta(days=1)
        
    if df_bars.empty:
        debug_log("⚠️ 日足データ取得失敗。プライムマスタの先頭から抽出します。")
        return prime_codes[:limit]

    # 3. 売買代金（TurnoverValue）でソートし、上位銘柄を抽出
    df_bars['Code4'] = df_bars['Code'].astype(str).str[:4]
    df_bars['TurnoverValue'] = pd.to_numeric(df_bars.get('TurnoverValue', 0), errors='coerce').fillna(0)
    
    # プライム市場のみに絞り込み
    df_filtered = df_bars[df_bars['Code4'].isin(prime_codes)].sort_values('TurnoverValue', ascending=False)
    top_codes = df_filtered['Code4'].head(limit).tolist()
    
    debug_log(f"✅ 動的ユニバース構築完了: 上位 {len(top_codes)} 銘柄を選定しました。")
    return [str(c) for c in top_codes]

# ==========================================
# 2. ヒストリカルデータ取得エンジン (yfinance)
# ==========================================
def fetch_and_save_data(ticker: str, is_benchmark: bool = False) -> bool:
    """指定された銘柄の10年分のヒストリカルデータを取得し保存する"""
    if not isinstance(ticker, str): raise TypeError("ticker must be a string")
    if not isinstance(is_benchmark, bool): raise TypeError("is_benchmark must be a bool")
    
    yf_ticker = f"{ticker}.T" if ticker.isdigit() else ticker
    save_name = f"{ticker}0" if ticker.isdigit() and len(ticker) == 4 else ticker
    
    try:
        t = yf.Ticker(yf_ticker)
        df = t.history(period="10y")
        
        if df.empty:
            debug_log(f"⚠️ No data found for {yf_ticker}")
            return False
            
        # インデックスのタイムゾーンを削除し、列名を小文字に統一
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        df.columns = [str(c).lower() for c in df.columns]
        
        # Datetime列が生成された場合のフォールバック
        if 'date' not in df.columns and 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
            
        # 必要な列が存在するか確認
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            debug_log(f"⚠️ Missing required columns for {yf_ticker}")
            return False
            
        # Parquet形式で高速ロード用に保存
        file_path = os.path.join(DATA_DIR, f"{save_name}.parquet")
        df.to_parquet(file_path, index=False)
        return True
        
    except Exception as e:
        debug_log(f"❌ Error fetching {yf_ticker}: {e}")
        return False

# ==========================================
# 3. メイン処理
# ==========================================
def main() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        debug_log(f"Created directory: {DATA_DIR}")
        
    print("\n==================================================")
    print(" 🚀 STARTING FULL-AUTO UNIVERSE DATA FETCHER")
    print("==================================================")
        
    # 1. ユニバースの自動抽出（流動性の高いプライム銘柄）
    target_tickers = get_liquid_prime_tickers(JQUANTS_API_KEY, limit=TARGET_UNIVERSE_SIZE)
    
    if not target_tickers:
        print("❌ 対象銘柄のリスト生成に失敗しました。処理を中断します。")
        return
        
    # 2. ベンチマークの取得
    fetch_and_save_data(BENCHMARK_TICKER, is_benchmark=True)
    
    # 3. 個別銘柄の取得（yfinanceのレートリミット回避のため微小なSleepを挟む）
    success_count = 0
    total = len(target_tickers)
    
    for i, ticker in enumerate(target_tickers):
        if i % 10 == 0:
            debug_log(f"Downloading progress: [{i}/{total}] ...")
            
        if fetch_and_save_data(ticker):
            success_count += 1
            
        # API制限（HTTP 429 Error）を回避するための安全な待機時間
        time.sleep(0.3)
            
    print("==================================================")
    print(f"✅ Data fetching complete. Successfully downloaded {success_count}/{total} tickers.")
    print("==================================================")

# ==========================================
# 4. 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_tests() -> None:
    debug_log("🧪 堅牢性テストを実行中...")
    
    # 1. 型チェックテスト
    try:
        fetch_and_save_data(1234) # type: ignore
        assert False, "TypeError should be raised for non-string ticker"
    except TypeError:
        pass
        
    # 2. 無効なティッカーの処理テスト
    assert fetch_and_save_data("INVALID_TICKER") is False, "Should return False for invalid ticker"
    
    # 3. APIキーなしでのフェイルセーフ動作テスト
    empty_res = get_liquid_prime_tickers("", limit=5)
    assert isinstance(empty_res, list) and len(empty_res) > 0, "Fallback list should be provided if API key is missing"
    
    debug_log("✅ 全てのテストを通過しました。")

if __name__ == "__main__":
    run_tests()
    main()
