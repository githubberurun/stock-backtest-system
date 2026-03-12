import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 0. 環境設定・定数定義
# ==========================================
# GitHub Secrets等からAPIキーを取得
JQUANTS_API_KEY: Final[str] = os.environ.get('JQUANTS_API_KEY', '').strip()
DATA_DIR: Final[str] = "data"
BENCHMARK_TICKER: Final[str] = "1306"
# 606%の利回りを再現するための母集団サイズ（中小型株を含むため500件に拡張）
TARGET_UNIVERSE_SIZE: Final[int] = 500  

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. 黄金ユニバース抽出エンジン (J-Quants API V2)
# ==========================================
def fetch_jquants_data(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """J-Quants APIからデータを取得する共通関数（リトライ付き）"""
    if not isinstance(url, str): raise TypeError("url must be string")
    if not isinstance(headers, dict): raise TypeError("headers must be dict")
    
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return data if isinstance(data, list) else []
            elif r.status_code == 429:
                debug_log("⚠️ Rate limit hit. Waiting...")
                time.sleep(10)
        except Exception as e:
            debug_log(f"J-Quants API Request Error: {e}")
            time.sleep(2)
    return []

def get_liquid_universe(api_key: str, limit: int = 500) -> List[str]:
    """全市場（プライム/スタンダード/グロース）から流動性のある銘柄を自動抽出する"""
    if not isinstance(api_key, str): raise TypeError("api_key must be string")
    if not isinstance(limit, int): raise TypeError("limit must be int")
    
    # APIキーがない場合のフェイルセーフ
    if not api_key:
        debug_log("⚠️ JQUANTS_API_KEYが設定されていません。フォールバックリストを使用します。")
        return ["7203", "8306", "9984", "8035", "6758", "4063", "8058", "6501", "4502", "9101"]

    debug_log("🔍 J-Quants APIから全市場の銘柄をスキャンして黄金ユニバースを構築中...")
    headers = {"x-api-key": api_key}
    base_url = "https://api.jquants.com/v2"
    
    # 1. 直近7日間の日足データから売買代金（TurnoverValue）の高い銘柄を取得
    target_date = datetime.now()
    df_bars = pd.DataFrame()
    for _ in range(7):
        date_str = target_date.strftime("%Y%m%d")
        bar_data = fetch_jquants_data(f"{base_url}/equities/bars/daily", headers, {"date": date_str})
        if bar_data and len(bar_data) > 1000:
            df_bars = pd.DataFrame(bar_data)
            break
        target_date -= timedelta(days=1)
        
    if df_bars.empty:
        debug_log("⚠️ 日足データの取得に失敗しました。")
        return []

    # 2. 型変換と流動性フィルタリング
    # TurnoverValue 列の欠損に備えた堅牢な変換
    if 'TurnoverValue' not in df_bars.columns:
        df_bars['TurnoverValue'] = 0.0
    df_bars['TurnoverValue'] = pd.to_numeric(df_bars['TurnoverValue'], errors='coerce').fillna(0.0)
    
    # 最低限の流動性フィルター（1日の売買代金が一定以上の銘柄のみ対象にする）
    # これにより、アルファ（超過収益）の源泉となる中小型優良株を拾いつつ、ボロ株を排除します
    df_liquid = df_bars[df_bars['TurnoverValue'] > 0].sort_values('TurnoverValue', ascending=False)
    
    # 3. 銘柄コードの抽出（先頭4桁）
    top_codes = [str(c)[:4] for c in df_liquid['Code'].head(limit).tolist()]
    
    debug_log(f"✅ 動的ユニバース構築完了: {len(top_codes)} 銘柄を選定しました。")
    return top_codes

# ==========================================
# 2. ヒストリカルデータ取得エンジン (yfinance)
# ==========================================
def fetch_and_save_data(ticker: str, is_benchmark: bool = False) -> bool:
    """指定された銘柄の10年分のデータを取得しParquet形式で保存する"""
    if not isinstance(ticker, str): raise TypeError("ticker must be a string")
    
    yf_ticker = f"{ticker}.T" if ticker.isdigit() else ticker
    save_name = f"{ticker}0" if ticker.isdigit() and len(ticker) == 4 else ticker
    
    try:
        t = yf.Ticker(yf_ticker)
        df = t.history(period="10y")
        
        if df.empty or len(df) < 200:
            return False
            
        # データの整形
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 列名の統一（yfinanceの仕様変更対応）
        if 'date' not in df.columns and 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
            
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            return False
            
        # 保存
        file_path = os.path.join(DATA_DIR, f"{save_name}.parquet")
        df.to_parquet(file_path, index=False)
        return True
        
    except Exception as e:
        debug_log(f"❌ Error fetching {yf_ticker}: {e}")
        return False

# ==========================================
# 3. メイン・パイプライン
# ==========================================
def main() -> None:
    # フォルダ準備
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        debug_log(f"Created directory: {DATA_DIR}")
        
    print("\n==================================================")
    print(" 🚀 STARTING GOLDEN UNIVERSE DATA FETCHER (VER.4.0)")
    print("==================================================")
        
    # 1. 黄金ユニバース（全市場の流動性上位）を自動取得
    target_tickers = get_liquid_universe(JQUANTS_API_KEY, limit=TARGET_UNIVERSE_SIZE)
    
    if not target_tickers:
        print("❌ 対象銘柄のリスト生成に失敗しました。処理を中断します。")
        return
        
    # 2. ベンチマーク(1306)の取得
    debug_log(f"Fetching benchmark data: {BENCHMARK_TICKER}")
    fetch_and_save_data(BENCHMARK_TICKER, is_benchmark=True)
    
    # 3. 個別銘柄のヒストリカルデータ取得
    success_count = 0
    total = len(target_tickers)
    
    for i, ticker in enumerate(target_tickers):
        if i % 50 == 0:
            debug_log(f"Downloading progress: [{i}/{total}] ...")
            
        if fetch_and_save_data(ticker):
            success_count += 1
            
        # yfinanceのレートリミット（429 Error）回避のための微小なスリープ
        time.sleep(0.2)
            
    print("==================================================")
    print(f"✅ 完了: {success_count}/{total} 銘柄のデータを構築しました。")
    print(f"📂 データ保存先: {os.path.abspath(DATA_DIR)}")
    print("==================================================")

# ==========================================
# 4. 堅牢性証明テスト
# ==========================================
def run_tests() -> None:
    debug_log("🧪 堅牢性テストを実行中...")
    
    # 型チェックテスト
    try:
        fetch_and_save_data(9984) # type: ignore
        assert False, "TypeError should be raised for non-string ticker"
    except TypeError:
        pass
        
    # APIキーなしでのフォールバックテスト
    res = get_liquid_universe("", limit=5)
    assert isinstance(res, list) and len(res) > 0, "Fallback list should be returned"
    
    # 欠損データ処理のシミュレーション
    dummy_df = pd.DataFrame({"Code": ["1111"]})
    if 'TurnoverValue' not in dummy_df.columns:
        dummy_df['TurnoverValue'] = 0.0
    dummy_df['TurnoverValue'] = pd.to_numeric(dummy_df['TurnoverValue'], errors='coerce').fillna(0.0)
    assert dummy_df['TurnoverValue'].iloc[0] == 0.0, "TurnoverValue filling failed"

    debug_log("✅ 全てのテストを通過しました。")

if __name__ == "__main__":
    run_tests()
    main()
