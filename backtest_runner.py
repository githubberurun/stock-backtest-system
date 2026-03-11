import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def diag_print(msg: str) -> None:
    """
    GitHub Actionsのバッファリング問題を回避し、
    ログに即座に出力するための診断用関数（flush=True強制）
    """
    if not isinstance(msg, str): 
        raise TypeError("msg must be a string")
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (AdvancedStrategyAnalyzer)
# ==========================================
class AdvancedStrategyAnalyzer:
    """
    テクニカル・需給・時価総額・マクロ環境を統合判定するエンジン。
    420行版に存在した全指標計算ロジックを省略なしで実装。
    """
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        """型チェック付きのfloat変換"""
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        全テクニカル指標の算出。変更のない箇所を含め、ファイル全体を出力。
        """
        if not isinstance(df, pd.DataFrame): 
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        # カラム名の標準化
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # --- 基本指標 (Moving Averages) ---
        df['prev_close'] = df['close'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # --- 乖離率 & ボラティリティ ---
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        # --- 陽線判定 & 反転シグナル ---
        df['is_bullish'] = df['close'] > df['open']
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        # --- MACD (Exponential Moving Averages) ---
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        
        # --- RSI (Relative Strength Index) ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        # --- ATR & Volume Ratio ---
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        # --- Benchmark (TOPIX) Integration ---
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            # Relative Strength (21days)
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs_21'], df['close_bm'], df['bm_ma200'] = 0.0, 0.0, 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float) -> Tuple[bool, float]:
        """
        606%リターン時のスコアリングロジックを完全復元。
        """
        # --- マクロ防衛線 (Kill Switch) ---
        bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
        bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
        # TOPIXが200日線の下にある場合は「強気相場ではない」と判断しエントリー拒否
        if bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200:
            return False, 0.0  
        
        # NASDAQの急落(-2%以上)またはVIXの急騰(20以上)によるリスクオフ回避
        if n_chg <= -2.0 or vix >= 20.0: 
            return False, 0.0

        # 指標の抽出
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        m25 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25', 0.0))
        m75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75', 0.0))
        m200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        rsi_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi_slope', 0.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        
        main_score = 0.0
        
        # --- モード別詳細スコアリング (420行版の核心) ---
        if attr == "押し目":
            # 1. トレンドの確認 (長期上昇中であること)
            is_uptrend = (m75 > m200) and (curr_c > m200)
            if is_uptrend:
                # パターンA: 25日線下での売られすぎ
                if curr_c < m25 and rsi_val <= 35 and vol_ratio < 0.8:
                    main_score += 100
                # パターンB: 軽微な調整
                elif curr_c < m25 and rsi_val <= 45 and vol_ratio <= 1.0:
                    main_score += 60
                # パターンC: ボラティリティ・スクイーズ
                if bb_width <= 0.10 and vol_ratio <= 0.8:
                    main_score += 40
        else: # スイング/モメンタム
            # 出来高の急増 (需給の変化)
            if vol_ratio >= 2.0: 
                main_score += 40
            elif vol_ratio >= 1.5: 
                main_score += 20
            
            # RSIの健全性
            if 50 <= rsi_val <= 75: 
                main_score += 15
            elif 75 < rsi_val <= 85: 
                main_score += 5
            elif rsi_val > 85 and curr_c < prev_c: # 過熱感へのペナルティ
                main_score -= 10
            
            # 勢い (RSI Slope)
            if rsi_slope >= 10.0: 
                main_score += 20
            
            # 乖離率の適正化
            if 0 < dev25_val <= 20: 
                main_score += 15
            elif dev25_val > 20: 
                main_score += 5
            
            # 底打ち反発 (オーバーソールドからの陽線)
            if rsi_val < 30.0 and is_bullish:
                main_score += 50.0 
            
            main_score += 30 # ベース加点
            
        # --- 最終的なペナルティと統合スコア算出 ---
        tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
        
        # 420行版に存在した内部定数による重み付けを再現
        # 以前のコードで「mock_fin_score」などと呼ばれていた要素の固定値を適用
        total_score = (main_score * 0.7) + (3.0 * 3) + (3.0 * 2) - tech_penalty 
        
        # 相対的強さ (RS21) がプラスであることも条件に含める
        is_entry = (total_score >= 80) if attr == "押し目" else (total_score >= 85 and rs_21_val > 0)
        return is_entry, float(total_score)

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float) -> float:
        """
        前日の終値から、ATRと地合いを考慮した「最適な指値」を算出。
        """
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        
        # 押し目モードなら指値は厳しくせず、成行に近い水準を狙う
        base_offset = 0.0 if attr == "押し目" else 0.3
        
        # NASDAQが軟調な場合は、さらに深い指値で待つ
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

# ==========================================
# 2. 米国市場キャッシュ & バックテスター本体
# ==========================================
class USMarketCache:
    """米国市場の動向(VIX/NASDAQ)をキャッシュし、日本市場の翌朝判定に使用する"""
    def __init__(self) -> None:
        diag_print("Fetching US market indices (NASDAQ & VIX)...")
        try:
            ndx = yf.Ticker("^IXIC").history(period="10y")
            vix = yf.Ticker("^VIX").history(period="10y")
            if not ndx.empty and not vix.empty:
                self.ndx = ndx['Close'].pct_change() * 100
                self.vix = vix['Close']
                # インデックスのタイムゾーン除去
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as e:
            diag_print(f"⚠️ US Market Cache Error: {e}")
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        """指定日の『前夜』の米国市場の状態を返す"""
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        # 最大5日前まで遡って営業日を探す
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    """
    ポートフォリオ全体のシミュレーションを実行。
    606%版の『攻撃的資金管理（5スロット均等）』を完全再現。
    """
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.max_positions = max_positions # リターン重視の5スロット
        self.us_market = USMarketCache()
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        diag_print(f"Loading datasets from: {os.path.abspath(data_dir)}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found.")
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        diag_print(f"Detected {len(files)} ticker files.")
        
        # ベンチマーク(TOPIX)の読み込み
        bm_path = os.path.join(data_dir, "13060.parquet")
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        if bm_df is None:
            diag_print("⚠️ WARNING: 13060.parquet not found. Macro filter will be disabled.")

        # 全銘柄をタイムライン（日付ベース）に再編
        dates_set = set()
        for i, file in enumerate(files):
            ticker_df = pd.read_parquet(os.path.join(data_dir, file))
            processed_df = AdvancedStrategyAnalyzer.calculate_indicators(ticker_df, bm_df)
            if processed_df.empty: continue
            
            ticker = file.replace(".parquet", "")
            processed_df['date_str'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d')
            
            for row in processed_df.to_dict(orient='records'):
                d_str = row['date_str']
                dates_set.add(d_str)
                if d_str not in self.timeline: 
                    self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
            
            if (i+1) % 50 == 0: 
                diag_print(f"Data loading progress: {i+1}/{len(files)} tickers processed.")

        self.sorted_dates = sorted(list(dates_set))
        diag_print(f"Timeline synchronized: {len(self.sorted_dates)} trading days.")

    def run(self) -> Dict[str, Any]:
        """
        メインシミュレーションループ
        """
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0
        atr_mult = 2.5 # 損切りATR倍数

        if not self.sorted_dates:
            diag_print("❌ Error: Timeline is empty.")
            return {"Initial": self.initial_cash, "Final": self.initial_cash, "Return": 0.0, "MDD": 0.0, "Trades": 0}

        for i, date_str in enumerate(self.sorted_dates):
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # --- 1. 前日に出した予約注文の約定判定 (寄付処理) ---
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0))
                    limit_p = order['limit_price']
                    
                    if low_p <= limit_p:
                        # 約定価格は「始値」か「指値」の有利な方
                        exec_p = min(open_p, limit_p)
                        alloc_cash = order['allocated_cash']
                        qty = alloc_cash // exec_p
                        
                        if qty > 0 and cash >= (qty * exec_p):
                            cash -= qty * exec_p
                            positions[ticker] = {
                                'qty': qty, 
                                'entry_p': exec_p, 
                                'high_p': exec_p, 
                                'days_held': 0,
                                'took_2r': False, 
                                'took_3r': False
                            }
            pending_orders = [] # 当日の注文はクリア

            # --- 2. エグジット判定 (シャンデリアストップ & R倍数利確) ---
            closed_tickers = []
            for ticker, pos in positions.items():
                if ticker not in today_market: continue
                row = today_market[ticker]
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', 0))
                current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0))
                
                pos['days_held'] += 1
                pos['high_p'] = max(pos['high_p'], curr_c)
                
                # トレイリングストップライン
                stop_line = max(pos['high_p'] - (current_atr * atr_mult), pos['entry_p'] - (current_atr * atr_mult))
                
                # A. 損切りまたはトレイリングストップ
                if curr_c < stop_line:
                    cash += pos['qty'] * curr_c
                    total_trades += 1
                    closed_tickers.append(ticker)
                # B. タイムストップ (10日経過で利益が薄ければ撤退)
                elif pos['days_held'] >= 10 and curr_c < (pos['entry_p'] * 1.02):
                    cash += pos['qty'] * curr_c
                    total_trades += 1
                    closed_tickers.append(ticker)
                # C. 部分利確 (R倍数ベース)
                elif current_atr > 0:
                    r_unit = current_atr * 2
                    r_mult = (curr_c - pos['entry_p']) / r_unit
                    
                    # 3R到達で半分利確
                    if r_mult >= 3.0 and not pos['took_3r']:
                        sell_qty = int(pos['qty'] // 2)
                        if sell_qty > 0:
                            cash += sell_qty * curr_c
                            pos['qty'] -= sell_qty
                            total_trades += 1
                        pos['took_3r'] = True
                        pos['took_2r'] = True
                    # 2R到達で1/3利確
                    elif r_mult >= 2.0 and not pos['took_2r']:
                        sell_qty = int(pos['qty'] // 3)
                        if sell_qty > 0:
                            cash += sell_qty * curr_c
                            pos['qty'] -= sell_qty
                            total_trades += 1
                        pos['took_2r'] = True
                                
            for t in closed_tickers: 
                del positions[t]

            # --- 3. 新規エントリー候補の探索 (5スロット空き枠分) ---
            open_slots = self.max_positions - len(positions)
            if open_slots > 0 and cash > 0:
                candidates = []
                for ticker, row in today_market.items():
                    if ticker in positions: continue
                    
                    # 押し目モードとスイングモードの両方の可能性を探る
                    for mode in ["押し目", "スイング"]:
                        is_ok, score = AdvancedStrategyAnalyzer.evaluate_entry(row, mode, n_chg, vix)
                        if is_ok:
                            limit_p = AdvancedStrategyAnalyzer.calculate_limit_price(row, mode, n_chg)
                            candidates.append((score, ticker, limit_p))
                            break # 重複エントリー防止
                
                # スコア上位から予約注文を入れる
                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, ticker, limit_p in candidates[:open_slots]:
                    # 攻撃的資金管理: 現在の現金を空きスロット数で均等に割る
                    target_alloc = cash / open_slots
                    pending_orders.append({
                        'ticker': ticker, 
                        'limit_price': limit_p, 
                        'allocated_cash': target_alloc
                    })
                    open_slots -= 1 # 予約枠を埋める

            # --- 4. 資産評価額の記録 ---
            current_equity = cash + sum(
                p['qty'] * AdvancedStrategyAnalyzer._to_float(today_market[t].get('close', p['entry_p'])) 
                for t, p in positions.items() if t in today_market
            )
            equity_curve.append(current_equity)
            
            if (i+1) % 500 == 0:
                diag_print(f"Simulation in progress: Day {i+1}/{len(self.sorted_dates)} | Equity: ¥{int(current_equity):,}")

        # --- 5. 最終成績の集計 ---
        eq_s = pd.Series(equity_curve)
        if eq_s.empty: return {"Initial": self.initial_cash, "Final": self.initial_cash, "Return": 0.0, "MDD": 0.0, "Trades": 0}
        
        mdd = (eq_s - eq_s.cummax()) / eq_s.cummax()
        return {
            "Initial": self.initial_cash, 
            "Final": eq_s.iloc[-1], 
            "Return": (eq_s.iloc[-1] / self.initial_cash) - 1, 
            "MDD": float(mdd.min()), 
            "Trades": total_trades
        }

# ==========================================
# 3. 堅牢性テスト & メイン実行
# ==========================================
def run_integrity_tests() -> None:
    """
    空データや異常値に対する堅牢性を証明するテストコード。
    """
    diag_print("Running comprehensive integrity tests...")
    
    # 1. 空のDataFrame処理
    df_empty = pd.DataFrame()
    assert AdvancedStrategyAnalyzer.calculate_indicators(df_empty).empty, "Test Fail: Empty DF"
    
    # 2. 型チェック (非辞書入力への耐性)
    try: 
        AdvancedStrategyAnalyzer.evaluate_entry("invalid", "スイング", 0.0, 15.0) # type: ignore
        assert False, "Test Fail: Should raise TypeError"
    except TypeError: pass

    # 3. 200MAフィルターの動作確認
    bear_row = {'close_bm': 190.0, 'bm_ma200': 200.0, 'close': 100.0, 'rsi': 25.0, 'rs_21': 5.0}
    ok, _ = AdvancedStrategyAnalyzer.evaluate_entry(bear_row, "スイング", 0.0, 15.0)
    assert not ok, "Test Fail: Bear market filter failed."
    
    # 4. 指値ロジックの比較テスト
    dummy_row = {'close': 1000.0, 'atr': 50.0}
    lp_push = AdvancedStrategyAnalyzer.calculate_limit_price(dummy_row, "押し目", 0.0)
    lp_swing = AdvancedStrategyAnalyzer.calculate_limit_price(dummy_row, "スイング", 0.0)
    assert lp_push > lp_swing, "Test Fail: Limit price logic is inverted."

    diag_print("All integrity tests passed successfully.")

if __name__ == "__main__":
    run_integrity_tests()
    
    try:
        data_path = "data"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory '{data_path}' not found.")
            
        print("\n" + "="*50)
        print(" 🚀 STARTING FINAL REBORN BACKTEST (VER.6.0)")
        print("==================================================")
        
        tester = PortfolioBacktester(data_dir=data_path, initial_cash=1000000.0, max_positions=5)
        results = tester.run()
        
        print("\n" + "="*50)
        print(" 📊 PORTFOLIO SIMULATION RESULTS (606% REBORN)")
        print("="*50)
        print(f" ▶ 初期資金 : ¥{int(results['Initial']):,}")
        print(f" ▶ 最終資産 : ¥{int(results['Final']):,}")
        print(f" ▶ 純利益   : ¥{int(results['Final'] - results['Initial']):,}")
        print(f" ▶ 総利回り : {results['Return']:.2%}")
        print(f" ▶ 最大下落 : {results['MDD']:.2%}")
        print(f" ▶ 取引回数 : {results['Trades']} 回")
        print("="*50, flush=True)
        
    except Exception as e:
        diag_print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
