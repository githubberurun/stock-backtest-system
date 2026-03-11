import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Final, Tuple
from datetime import datetime, timedelta

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. 統合分析エンジン (Entry & Exit & Limit Price)
# ==========================================
class AdvancedStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError): return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
        if df.empty or len(df) < 200: return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        df['prev_close'] = df['close'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        df['is_bullish'] = df['close'] > df['open']
        
        bull_condition = df['close'] > df['open']
        df['consecutive_bull_days'] = bull_condition.groupby((~bull_condition).cumsum()).cumsum()

        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        df['is_macd_improving'] = df['macd_hist'] > df['macd_hist'].shift(1)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            df = df.merge(benchmark_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['rs'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any]) -> Tuple[bool, float]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        
        curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        ma5 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma5', 0.0))
        rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
        vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        is_macd_improving = bool(row_dict.get('is_macd_improving', False))
        consecutive_bull_days = int(row_dict.get('consecutive_bull_days', 0))
        
        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        mr_zscore = AdvancedStrategyAnalyzer._to_float(row_dict.get('mr_zscore', 0.0))
        dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))

        is_small_cap = 0.0 < mcap < 500.0 if mcap > 0 else False

        if is_small_cap and (mr_zscore >= 1.0 or dev25_val <= -20.0):
            return False, 0.0

        if curr_c >= ma5 or (is_small_cap and consecutive_bull_days >= 2):
            return False, 0.0

        main_score = 50.0 
        if vol_ratio >= 2.0: main_score += 40
        elif vol_ratio >= 1.5: main_score += 20
        
        if 50 <= rsi_val <= 75: main_score += 15
        elif 75 < rsi_val <= 85: main_score += 5

        if rsi_val < 30.0 and is_bullish:
            if is_small_cap:
                if is_macd_improving and vol_ratio >= 1.5: main_score += 50.0
            else:
                main_score += 50.0 
                
        is_entry = (main_score >= 80)
        return is_entry, float(main_score)

    @staticmethod
    def get_order_params(row_dict: Dict[str, Any]) -> Tuple[float, float]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        
        is_small_cap = 0.0 < mcap < 500.0 if mcap > 0 else False
        
        if is_small_cap:
            limit_price = curr_price - (atr * 1.5)
            atr_mult = 3.0 
        else:
            limit_price = curr_price - (atr * 0.3)
            atr_mult = 2.5
            
        return float(max(1.0, limit_price)), float(atr_mult)

# ==========================================
# 2. 米国市場キャッシュ & ポートフォリオバックテスター
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        debug_log("Caching US market data...")
        try:
            ndx_data = yf.Ticker("^IXIC").history(period="10y")
            vix_data = yf.Ticker("^VIX").history(period="10y")
            if not ndx_data.empty and not vix_data.empty:
                self.ndx = ndx_data['Close'].pct_change() * 100
                self.vix = vix_data['Close']
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception:
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if not isinstance(date_str, str): raise TypeError("date_str must be string")
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.vix.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class PortfolioBacktester:
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.us_market = USMarketCache()
        
        debug_log("Loading and calculating indicators for all tickers...")
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        dates_set = set()
        
        bm_path = f"{data_dir}/13060.parquet"
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        
        for file in files:
            ticker = file.replace(".parquet", "")
            df = pd.read_parquet(f"{data_dir}/{file}")
            df = AdvancedStrategyAnalyzer.calculate_indicators(df, bm_df)
            if df.empty: continue
            
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            records = df.to_dict(orient='records')
            for row in records:
                d_str = row['date_str']
                dates_set.add(d_str)
                if d_str not in self.timeline: self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
                
        self.sorted_dates = sorted(list(dates_set))
        debug_log(f"Timeline built. Total trading days: {len(self.sorted_dates)}")

    def _calculate_total_equity(self, today_market: Dict[str, Dict[str, Any]], positions: Dict[str, Dict[str, Any]]) -> float:
        """現在の口座総資産（現金＋建玉の評価額）を計算"""
        equity = self.cash
        for ticker, pos in positions.items():
            if ticker in today_market:
                curr_c = AdvancedStrategyAnalyzer._to_float(today_market[ticker].get('close', pos['entry_p']))
                equity += pos['qty'] * curr_c
            else:
                equity += pos['qty'] * pos['entry_p']
        return equity

    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # 1. 前日の予約注文の約定処理
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0.0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
                    limit_p = order['limit_price']
                    target_qty = order['qty']
                    
                    if low_p <= limit_p:
                        exec_price = min(open_p, limit_p)
                        required_cash = target_qty * exec_price
                        
                        if target_qty > 0 and cash >= required_cash:
                            cash -= required_cash
                            if ticker in positions:
                                old_qty = positions[ticker]['qty']
                                old_entry = positions[ticker]['entry_p']
                                new_qty = old_qty + target_qty
                                new_entry = ((old_qty * old_entry) + (target_qty * exec_price)) / new_qty
                                positions[ticker]['qty'] = new_qty
                                positions[ticker]['entry_p'] = new_entry
                            else:
                                positions[ticker] = {
                                    'qty': target_qty, 'entry_p': exec_price, 
                                    'highest_close': exec_price, 
                                    'atr_mult': order['atr_mult'],
                                    'took_2r': False, 'took_3r': False, 'days_held': 0
                                }
            pending_orders = new_pending

            # 2. エグジット判定
            closed_tickers = []
            for ticker, pos in positions.items():
                if ticker not in today_market: continue
                row = today_market[ticker]
                
                curr_c = AdvancedStrategyAnalyzer._to_float(row.get('close', 0.0))
                current_atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
                dev25_val = AdvancedStrategyAnalyzer._to_float(row.get('dev25', 0.0))
                rsi_val = AdvancedStrategyAnalyzer._to_float(row.get('rsi', 50.0))
                vol_ratio = AdvancedStrategyAnalyzer._to_float(row.get('vol_ratio', 1.0))
                
                pos['days_held'] += 1
                pos['highest_close'] = max(pos['highest_close'], curr_c)
                
                ch_stop = max(pos['highest_close'] - (current_atr * pos['atr_mult']), pos['entry_p'] - (current_atr * pos['atr_mult']))
                
                exit_score = 0
                if bool(row.get('bb_3_reversal', False)): exit_score += 40
                if curr_c < ch_stop: exit_score += 100 
                if dev25_val > 20.0 and rsi_val > 85.0 and vol_ratio >= 2.0: exit_score += 100 
                if pos['days_held'] >= 10 and curr_c < (pos['entry_p'] * 1.02): exit_score += 100 
                if bool(row.get('bb_p1_cross_down', False)): exit_score += 20
                if AdvancedStrategyAnalyzer._to_float(row.get('ma5', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('ma25', 0)) and vol_ratio >= 1.0: exit_score += 15
                if AdvancedStrategyAnalyzer._to_float(row.get('macd', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('sig', 0)) and vol_ratio >= 1.0: exit_score += 15
                
                if exit_score >= 80:
                    cash += pos['qty'] * curr_c
                    total_trades += 1
                    closed_tickers.append(ticker)
                else:
                    if current_atr > 0 and curr_c > pos['entry_p']:
                        r_mult = (curr_c - pos['entry_p']) / (current_atr * 2)
                        if r_mult >= 3.0 and not pos['took_3r']:
                            sell_qty = int(pos['qty'] // 2)
                            if sell_qty > 0:
                                cash += sell_qty * curr_c
                                pos['qty'] -= sell_qty
                                total_trades += 1
                                pos['took_3r'] = True
                                pos['took_2r'] = True
                        elif r_mult >= 2.0 and not pos['took_2r']:
                            sell_qty = int(pos['qty'] // 3)
                            if sell_qty > 0:
                                cash += sell_qty * curr_c
                                pos['qty'] -= sell_qty
                                total_trades += 1
                                pos['took_2r'] = True
                                
            for ct in closed_tickers: del positions[ct]

            # 3. エントリー候補の探索と動的資金管理
            current_equity = self._calculate_total_equity(today_market, positions)
            available_cash = cash
            
            if vix >= 30.0:
                available_cash = 0.0

            if available_cash > 0:
                candidates = []
                for ticker, row in today_market.items():
                    if ticker in positions: continue 
                    is_entry, score = AdvancedStrategyAnalyzer.evaluate_entry(row)
                    if is_entry:
                        limit_p, atr_mult = AdvancedStrategyAnalyzer.get_order_params(row)
                        candidates.append((score, ticker, limit_p, atr_mult, row))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                for score, ticker, limit_p, atr_mult, row in candidates:
                    if available_cash <= 0: break
                    
                    atr = AdvancedStrategyAnalyzer._to_float(row.get('atr', 0.0))
                    if atr <= 0: continue
                        
                    stop_loss = limit_p - (atr * atr_mult)
                    diff = max(0.1, limit_p - stop_loss) 
                    
                    risk_fund = current_equity * 0.01
                    raw_shares = int((risk_fund / diff) // 100 * 100)
                    
                    max_alloc = current_equity * 0.20
                    max_shares = int((max_alloc / limit_p) // 100 * 100)
                    
                    target_qty = min(raw_shares, max_shares)
                    
                    if target_qty >= 100:
                        required_cash = target_qty * limit_p
                        if available_cash >= required_cash:
                            pending_orders.append({
                                'ticker': ticker, 
                                'limit_price': limit_p, 
                                'atr_mult': atr_mult,
                                'qty': target_qty
                            })
                            available_cash -= required_cash

            self.cash = cash 
            equity_curve.append(current_equity)

        final_equity = equity_curve[-1] if equity_curve else self.initial_cash
        
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            cummax = eq_series.cummax()
            mdd_series = (eq_series - cummax) / cummax
            mdd = float(mdd_series.min()) if not pd.isna(mdd_series.min()) else 0.0
        else:
            mdd = 0.0
            
        return {
            "Initial_Cash": self.initial_cash, 
            "Final_Cash": final_equity, 
            "Net_Profit": final_equity - self.initial_cash, 
            "Return": f"{(final_equity - self.initial_cash) / self.initial_cash:.2%}", 
            "MDD": f"{mdd:.2%}", 
            "Total_Trades": total_trades
        }

# ==========================================
# 3. 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running edge-case logic tests...")
    
    df_empty = pd.DataFrame()
    assert AdvancedStrategyAnalyzer.calculate_indicators(df_empty).empty, "Empty DataFrame should return empty"
    
    try: AdvancedStrategyAnalyzer.evaluate_entry("invalid_type") # type: ignore
    except TypeError: pass
    else: assert False, "Type checking failed on evaluate_entry"

    dummy_large = {'close': 1000.0, 'atr': 50.0, 'mcap': 1000.0}
    dummy_small = {'close': 1000.0, 'atr': 50.0, 'mcap': 300.0}
    
    limit_l, atr_l = AdvancedStrategyAnalyzer.get_order_params(dummy_large)
    limit_s, atr_s = AdvancedStrategyAnalyzer.get_order_params(dummy_small)
    
    assert limit_s < limit_l, "Small cap limit price must be deeper"
    assert atr_s > atr_l, "Small cap ATR multiplier must be wider"

    # [改修] FileNotFoundError回避のため一時ディレクトリを作成して検証
    test_dir = "dummy_test_dir_for_integrity"
    os.makedirs(test_dir, exist_ok=True)
    try:
        tester = PortfolioBacktester(data_dir=test_dir, initial_cash=1000000.0)
        mock_market = {'9999': {'close': 1500.0}}
        mock_positions = {'9999': {'qty': 100, 'entry_p': 1000.0}}
        equity = tester._calculate_total_equity(mock_market, mock_positions)
        assert equity == 1000000.0 + (100 * 1500.0), "Equity calculation failed"
    finally:
        os.rmdir(test_dir)

    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory '{data_dir}' not found. Please run data_fetcher.py first.")
            
        print("\n==================================================")
        print(" 🚀 STARTING FULL UNIVERSE CROSS-SECTIONAL BACKTEST (VER.3.1)")
        print("==================================================")
        
        tester = PortfolioBacktester(data_dir=data_dir, initial_cash=1000000.0)
        res = tester.run()
        
        print(f"\n==================================================")
        print(f" 📊 PORTFOLIO SIMULATION RESULTS (FULL UNIVERSE)")
        print(f"==================================================")
        print(f" ▶ 初期資金 : ¥{int(res['Initial_Cash']):,}")
        print(f" ▶ 最終資産 : ¥{int(res['Final_Cash']):,}")
        print(f" ▶ 純利益   : ¥{int(res['Net_Profit']):,}")
        print(f" ▶ 総利回り : {res['Return']}")
        print(f" ▶ 最大下落 : {res['MDD']}")
        print(f" ▶ 取引回数 : {res['Total_Trades']} 回")
        print(f"==================================================")
        
    except Exception as e:
        print(f"[FATAL] {e}")
