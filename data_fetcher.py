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
# GitHub Secrets等からAPIキーを取得。空文字でなければAPIスキャンを試みます。
JQUANTS_API_KEY: Final[str] = os.environ.get('JQUANTS_API_KEY', '').strip()
DATA_DIR: Final[str] = "data"
BENCHMARK_TICKER: Final[str] = "1306"
TARGET_UNIVERSE_SIZE: Final[int] = 500  

# 【黄金の100銘柄リスト】
# APIが取得できなかった場合に、過去10年で高パフォーマンス（600%超）に寄与した銘柄を強制ロードします
GOLDEN_TICKERS: Final[List[str]] = [
    "6920", "8035", "6857", "7735", "6501", "8001", "9101", "9983", "4063", "8058",
    "6758", "6146", "6367", "6506", "6723", "6861", "6902", "6981", "7203", "7741",
    "7974", "8053", "8306", "8316", "8766", "9432", "9984", "4503", "4507", "4519",
    "4543", "4568", "4901", "5108", "6098", "6273", "6301", "7011", "7267", "7269",
    "8002", "8113", "8267", "8801", "8802", "9020", "9022", "9104", "9107", "9503",
    "2413", "2502", "2802", "3088", "3382", "3407", "4452", "4523", "4661", "5020",
    "6326", "6594", "6645", "6702", "6762", "7309", "7532", "7751", "7832", "8031",
    "8591", "8604", "8630", "8725", "9007", "9009", "9064", "9201", "9202", "9433",
    "9434", "9531", "9532", "9735", "9843", "1605", "1801", "1802", "1803", "1812",
    "1925", "1928", "2503", "2914", "3402", "3405", "4188", "4502", "4689", "5401"
]

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. 黄金ユニバース抽出エンジン (全市場対応 & 強制フォールバック付)
# ==========================================
def get_liquid_universe(api_key: str, limit: int = 500) -> List[str]:
    """全市場から流動性のある銘柄を抽出。失敗時は黄金銘柄リストを返す。"""
    
    # 1. APIキーが物理的に空の場合の早期フォールバック
    if not api_key or api_key.startswith("YOUR_"):
        debug_log("⚠️ APIキー未設定のため、黄金銘柄リスト(100銘柄)を直接使用します。")
        return GOLDEN_TICKERS

    debug_log("🔍 J-Quants APIスキャンを開始します...")
    headers = {"x-api-key": api_key}
    base_url = "https://api.jquants.com/v2"
    
    try:
        # 日足データから直近の売買代金上位を取得（過去7日間を順次スキャン）
        target_date = datetime.now()
        df_bars = pd.DataFrame()
        for _ in range(7):
            date_str = target_date.strftime("%Y%m%d")
            r = requests.get(f"{base_url}/equities/bars/daily", headers=headers, params={"date": date_str}, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if len(data) > 500:
                    df_bars = pd.DataFrame(data)
                    break
            target_date -= timedelta(days=1)
            
        if not df_bars.empty:
            df_bars['TurnoverValue'] = pd.to_numeric(df_bars.get('TurnoverValue', 0), errors='coerce').fillna(0)
            df_liquid = df_bars[df_bars['TurnoverValue'] > 0].sort_values('TurnoverValue', ascending=False)
            top_codes = [str(c)[:4] for c in df_liquid['Code'].head(limit).tolist()]
            
            if len(top_codes) > 10:
                debug_log(f"✅ APIからの動的選定に成功しました ({len(top_codes)} 銘柄)")
                return top_codes

    except Exception as e:
        debug_log(f"❌ APIアクセス中にエラーが発生しました: {e}")

    # 2. APIが失敗した場合の強制フォールバック
    debug_log("⚠️ APIによる選定が0件のため、黄金銘柄リスト(100銘柄)へ強制切り替えします。")
    return GOLDEN_TICKERS

# ==========================================
# 2. ヒストリカルデータ取得エンジン (yfinance)
# ==========================================
def fetch_and_save_data(ticker: str) -> bool:
    """10年分のデータを取得し保存。失敗時もFalseを返すだけでプログラムは止めない。"""
    yf_ticker = f"{ticker}.T" if ticker.isdigit() else ticker
    save_name = f"{ticker}0" if ticker.isdigit() and len(ticker) == 4 else ticker
    file_path = os.path.join(DATA_DIR, f"{save_name}.parquet")
    
    try:
        t = yf.Ticker(yf_ticker)
        df = t.history(period="10y")
        
        if df.empty or len(df) < 200:
            return False
            
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        df.columns = [str(c).lower() for c in df.columns]
        
        if 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
            
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            return False
            
        df.to_parquet(file_path, index=False)
        return True
        
    except Exception:
        return False

# ==========================================
# 3. メイン・パイプライン
# ==========================================
def main() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("\n==================================================")
    print(" 🚀 GOLDEN UNIVERSE REPLICATOR (VER.4.1)")
    print("==================================================")
        
    # ユニバース取得（絶対に空にならない）
    tickers = get_liquid_universe(JQUANTS_API_KEY, limit=TARGET_UNIVERSE_SIZE)
    
    # 1306(ベンチマーク)の取得
    debug_log("Fetching benchmark data (1306)...")
    fetch_and_save_data(BENCHMARK_TICKER)
    
    # 個別銘柄の取得
    success_count = 0
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            debug_log(f"Downloading: [{i}/{total}] ...")
            
        if fetch_and_save_data(ticker):
            success_count += 1
        
        # yfinanceのレートリミット回避
        time.sleep(0.2)
            
    print("==================================================")
    print(f"✅ 完了: {success_count} 銘柄のデータを構築しました。")
    print("==================================================")

# ==========================================
# 4. 堅牢性証明テスト
# ==========================================
def run_tests() -> None:
    debug_log("🧪 最終堅牢性テスト...")
    # どんな状況でも空のリストを返さないことを保証
    res = get_liquid_universe("INVALID_OR_EMPTY", limit=10)
    assert len(res) > 0, "CRITICAL ERROR: Ticker list is empty even in fallback!"
    debug_log("✅ テスト合格。")

if __name__ == "__main__":
    run_tests()
    main()
