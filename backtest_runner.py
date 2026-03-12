import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用の即時出力関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 1. 統合分析エンジン (Entry & Exit & Limit Price)
# ==========================================
class AdvancedStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrameに必須列が不足しています: {missing}")

        # === 600%実績コードと完全同一の指標群 ===
        df['prev_close'] = df['close'].shift(1)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_up_3'] = df['ma20'] + (df['std20'] * 3)
        df['bb_p1'] = df['ma20'] + df['std20'] 
        df['prev_low'] = df['low'].shift(1)
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['bb_width'] = np.where(df['ma20'] > 0, (df['std20'] * 4) / df['ma20'], 0)
        
        df['is_bullish'] = df['close'] > df['open']
        
        df['was_above_bb_p1'] = (df['high'] >= df['bb_p1']).rolling(window=5).max() > 0
        df['bb_p1_cross_down'] = df['was_above_bb_p1'] & (df['close'] < df['bb_p1']) & (df['close'] < df['prev_low'])
        df['was_above_bb_up_3'] = (df['high'] >= df['bb_up_3']).rolling(window=3).max() > 0
        df['bb_3_reversal'] = df['was_above_bb_up_3'] & ((df['close'] < df['prev_low']) | (df['close'] < df['open']))

        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean() 
        df['vol_ratio'] = (df['volume'] / df['volume'].rolling(25).mean().replace(0, np.nan)).fillna(0)

        df['turnover'] = df['close'] * df['volume']
        df['ma25_turnover'] = df['turnover'].rolling(window=25).mean()

        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            df = df.merge(benchmark_df[['date', 'close', 'bm_ma200']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['bm_ma200'] = df['bm_ma200'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
            df['rs'] = df['rs_21']
        else:
            df['rs_21'] = 0.0
            df['rs'] = 0.0
            df['close_bm'] = 0.0
            df['bm_ma200'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], attr: str, n_chg: float, vix: float) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        
        mcap = AdvancedStrategyAnalyzer._to_float(row_dict.get('mcap', 0.0))
        ma25_turnover = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25_turnover', 0.0))
        is_small_cap = (0 < mcap < 500.0) or (mcap == 0.0 and 0 < ma25_turnover < 500_000_000)

        # ==========================================
        # [A] 大型株ロジック (600%コード完全一致・聖域)
        # ==========================================
        if not is_small_cap:
            bm_close = AdvancedStrategyAnalyzer._to_float(row_dict.get('close_bm', 0.0))
            bm_ma200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('bm_ma200', 0.0))
            if bm_close > 0 and bm_ma200 > 0 and bm_close < bm_ma200:
                return False, 0.0, is_small_cap  
            
            if n_chg <= -2.0 or vix >= 20.0: return False, 0.0, is_small_cap
            
            curr_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
            prev_c = AdvancedStrategyAnalyzer._to_float(row_dict.get('prev_close', 0.0))
            rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0), 50.0)
            dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0), 0.0)
            rs_21_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0), 0.0)
            vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0), 1.0)
            
            m25 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma25', 0.0))
            m75 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma75', 0.0))
            m200 = AdvancedStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
            bb_width = AdvancedStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
            rsi_slope = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi_slope', 0.0))
            is_bullish = bool(row_dict.get('is_bullish', False))
            
            main_score = 0.0
            if attr == "押し目":
                is_uptrend = (m75 > m200) and (curr_c > m200)
                if is_uptrend:
                    if curr_c < m25 and rsi_val <= 35 and vol_ratio < 0.8: main_score += 100
                    elif curr_c < m25 and rsi_val <= 45 and vol_ratio <= 1.0: main_score += 60
                    if bb_width <= 0.10 and vol_ratio <= 0.8: main_score += 40
            else:
                if vol_ratio >= 2.0: main_score += 40
                elif vol_ratio >= 1.5: main_score += 20
                if 50 <= rsi_val <= 75: main_score += 15
                elif 75 < rsi_val <= 85: main_score += 5
                elif rsi_val > 85 and curr_c < prev_c: main_score -= 10
                if rsi_slope >= 10.0: main_score += 20
                if 0 < dev25_val <= 20: main_score += 15
                elif dev25_val > 20: main_score += 5
                if bb_width <= 0.10 and vol_ratio <= 0.8: main_score += 20
                
                if rsi_val < 30.0 and is_bullish:
                    main_score += 50.0 
                    
                main_score += 30 
                
            surrogate_base = min(100.0, main_score)
            mock_fin_score, mock_appear_count = 3.0, 3.0
            tech_penalty = (20.0 if rsi_val > 80 else 0) + (15.0 if dev25_val > 20 else 0)
            
            total_score = (surrogate_base * 0.7) + (mock_fin_score * 3) + (mock_appear_count * 2) - tech_penalty 
            is_entry = (total_score >= 80) if attr == "押し目" else (total_score >= 85 and rs_21_val > 0)
            
            return is_entry, float(total_score), is_small_cap

        # ==========================================
        # [B] 小型株専用ロジック (354%版完全復元)
        # ==========================================
        else:
            if n_chg <= -2.0 or vix >= 20.0: return False, 0.0, is_small_cap
            
            rsi_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0))
            dev25_val = AdvancedStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
            vol_ratio = AdvancedStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
            is_bullish = bool(row_dict.get('is_bullish', False))
            
            is_entry = False
            score = 0.0
            
            # 354%を叩き出した「RS無視のゲリラエントリー」
            if is_bullish and vol_ratio >= 1.5:
                if rsi_val < 35.0 or dev25_val < -15.0:
                    is_entry = True
                    score = 90.0
            
            return is_entry, score, is_small_cap

    @staticmethod
    def calculate_limit_price(row_dict: Dict[str, Any], attr: str, n_chg: float, is_small_cap: bool) -> float:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        curr_price = AdvancedStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        atr = AdvancedStrategyAnalyzer._to_float(row_dict.get('atr', 0.0))
        
        nasdaq_drop_ratio = abs(n_chg) / 100.0 if n_chg <= -0.8 else 0.0
        
        # 354%実績の指値 (大型0.3 / 小型0.6) へ完全復帰
        if not is_small_cap:
            base_offset = 0.5 if "中長期" in attr else (0.0 if attr == "押し目" else 0.3)
        else:
            base_offset = 0.6 
            
        limit_price = curr_price - (atr * base_offset) - (curr_price * nasdaq_drop_ratio)
        return float(max(1.0, limit_price))

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
                # 600% / 354%実績の「1日間変化率」へ完全復帰
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
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.max_positions: int = max_positions
        
        # 【中核機構】小型株の同時保有枠を「最大1枠」に制限（機会損失の防止）
        self.max_small_cap_positions: int = 1 
        
        self.attr: str = "スイング" 
        self.us_market = USMarketCache()
        
        self.stats = {
            'limit_placed_small': 0, 'limit_exec_small': 0,
            'limit_placed_large': 0, 'limit_exec_large': 0,
            'ts_small': 0, 'ts_small_win': 0,
            'ts_large': 0, 'ts_large_win': 0,
            'sc_quota_blocked': 0
        }
        
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

    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            # 1. 注文の約定処理
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                
                # 約定時にも小型株のクオータ（保有枠上限）を厳格にチェック
                current_sc_count = sum(1 for p in positions.values() if p['is_sc'])
                if order['is_sc'] and current_sc_count >= self.max_small_cap_positions:
                    self.stats['sc_quota_blocked'] += 1
                    continue
                    
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    low_p = AdvancedStrategyAnalyzer._to_float(row.get('low', 0.0))
                    open_p = AdvancedStrategyAnalyzer._to_float(row.get('open', 0.0))
                    limit_p = order['limit_price']
                    
                    if order['is_sc']: self.stats['limit_placed_small'] += 1
                    else: self.stats['limit_placed_large'] += 1
                    
                    if low_p <= limit_p:
                        exec_price = min(open_p, limit_p)
                        alloc_cash = order['allocated_cash']
                        qty = alloc_cash // exec_price
                        
                        if qty > 0 and cash >= (qty * exec_price):
                            cash -= qty * exec_price
                            positions[ticker] = {
                                'qty': qty, 'entry_p': exec_price, 'high_p': exec_price, 
                                'took_2r': False, 'took_3r': False,
                                'days_held': 0, 'is_sc': order['is_sc']
                            }
                            if order['is_sc']: self.stats['limit_exec_small'] += 1
                            else: self.stats['limit_exec_large'] += 1
            pending_orders = [] # 当日約定しなかった指値は破棄（翌日再計算）

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
                pos['high_p'] = max(pos['high_p'], curr_c)
                
                # ==========================================
                # [A] 大型株エグジット (600%コード完全一致)
                # ==========================================
                if not pos['is_sc']:
                    atr_mult = 2.5 
                    ch_stop = max(pos['high_p'] - (current_atr * atr_mult), pos['entry_p'] - (current_atr * atr_mult))
                    
                    exit_score = 0
                    if bool(row.get('bb_3_reversal', False)): exit_score += 40
                    if curr_c < ch_stop: exit_score += 100 
                    
                    if dev25_val > 20.0 and rsi_val > 85.0 and vol_ratio > 2.0:
                        exit_score += 100 
                    
                    if pos['days_held'] >= 10 and curr_c < (pos['entry_p'] * 1.02):
                        exit_score += 100
                    
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
                            
                    if bool(row.get('bb_p1_cross_down', False)): exit_score += 20
                    if AdvancedStrategyAnalyzer._to_float(row.get('ma5', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('ma25', 0)) and vol_ratio >= 1.0: exit_score += 15
                    if AdvancedStrategyAnalyzer._to_float(row.get('macd', 0)) < AdvancedStrategyAnalyzer._to_float(row.get('sig', 0)) and vol_ratio >= 1.0: exit_score += 15
                    if AdvancedStrategyAnalyzer._to_float(row.get('rs', 0)) < -5: exit_score += 5
                    
                    if exit_score >= 80:
                        cash += pos['qty'] * curr_c
                        total_trades += 1
                        closed_tickers.append(ticker)
                        if pos['days_held'] >= 10 and curr_c < (pos['entry_p'] * 1.02):
                            self.stats['ts_large'] += 1
                            if curr_c > pos['entry_p']: self.stats['ts_large_win'] += 1

                # ==========================================
                # [B] 小型株エグジット (354%版への完全復元: 3.0ATRストップ)
                # ==========================================
                else:
                    atr_mult = 3.0 
                    ch_stop = max(pos['high_p'] - (current_atr * atr_mult), pos['entry_p'] - (current_atr * atr_mult))
                    
                    is_exit = False
                    
                    if curr_c < ch_stop:
                        is_exit = True
                    elif pos['days_held'] >= 7 and curr_c < (pos['entry_p'] * 1.02):
                        is_exit = True
                        self.stats['ts_small'] += 1
                        if curr_c > pos['entry_p']: self.stats['ts_small_win'] += 1
                    elif dev25_val > 30.0 and rsi_val > 80.0:
                        is_exit = True
                    
                    if current_atr > 0 and curr_c > pos['entry_p'] and not is_exit:
                        r_mult = (curr_c - pos['entry_p']) / (current_atr * 2)
                        if r_mult >= 2.5 and not pos['took_3r']:
                            sell_qty = int(pos['qty'] // 2)
                            if sell_qty > 0:
                                cash += sell_qty * curr_c
                                pos['qty'] -= sell_qty
                                total_trades += 1
                            pos['took_3r'] = True
                            pos['took_2r'] = True
                        elif r_mult >= 1.5 and not pos['took_2r']:
                            sell_qty = int(pos['qty'] // 3)
                            if sell_qty > 0:
                                cash += sell_qty * curr_c
                                pos['qty'] -= sell_qty
                                total_trades += 1
                            pos['took_2r'] = True

                    if is_exit:
                        cash += pos['qty'] * curr_c
                        total_trades += 1
                        closed_tickers.append(ticker)

            for ct in closed_tickers:
                del positions[ct]

            # 3. 新規エントリー候補の探索
            open_slots = self.max_positions - len(positions)
            if open_slots > 0 and cash > 0:
                current_sc_count = sum(1 for p in positions.values() if p['is_sc'])
                candidates = []
                
                for ticker, row in today_market.items():
                    if ticker in positions: continue 
                    
                    is_entry, score, is_sc = AdvancedStrategyAnalyzer.evaluate_entry(row, self.attr, n_chg, vix)
                    
                    # 【重要】エントリー探索時点でも小型株のクオータ（上限1枠）をチェックし、超過時は無視する
                    if is_entry:
                        if is_sc and current_sc_count >= self.max_small_cap_positions:
                            continue
                        
                        limit_p = AdvancedStrategyAnalyzer.calculate_limit_price(row, self.attr, n_chg, is_sc)
                        candidates.append((score, ticker, limit_p, is_sc))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                for score, ticker, limit_p, is_sc in candidates[:open_slots]:
                    target_alloc = cash / open_slots
                    pending_orders.append({
                        'ticker': ticker,
                        'limit_price': limit_p,
                        'allocated_cash': target_alloc,
                        'is_sc': is_sc
                    })
                    if is_sc:
                        current_sc_count += 1 # 今回のループ内での上限超過を防ぐ
                    open_slots -= 1

            # 4. 日次資産の記録
            daily_equity = cash
            for ticker, pos in positions.items():
                if ticker in today_market:
                    curr_c = AdvancedStrategyAnalyzer._to_float(today_market[ticker].get('close', pos['entry_p']))
                    daily_equity += pos['qty'] * curr_c
            equity_curve.append(daily_equity)

        final_equity = equity_curve[-1] if equity_curve else self.initial_cash
        
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            cummax = eq_series.cummax()
            mdd_series = (eq_series - cummax) / cummax
            mdd = float(mdd_series.min()) if not pd.isna(mdd_series.min()) else 0.0
        else:
            mdd = 0.0
            
        ret_val = (final_equity - self.initial_cash) / self.initial_cash
        
        return {
            "Initial_Cash": self.initial_cash,
            "Final_Cash": final_equity,
            "Net_Profit": final_equity - self.initial_cash,
            "Return": f"{ret_val:.2%}",
            "MDD": f"{mdd:.2%}",
            "Total_Trades": total_trades,
            "Stats": self.stats
        }

# ==========================================
# 3. 堅牢性テスト & メイン実行
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running integrity and edge-case tests...")
    
    empty_df = pd.DataFrame()
    res_df = AdvancedStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    # 354%版 小型株ロジックテスト
    dummy_row_small = {'close_bm': 2100.0, 'bm_ma200': 2000.0, 'close': 100.0, 'open': 90.0, 'rsi': 25.0, 'is_bullish': True, 'mcap': 100.0, 'dev25': -20.0, 'vol_ratio': 2.0}
    is_entry, score, is_sc = AdvancedStrategyAnalyzer.evaluate_entry(dummy_row_small, "スイング", 0.0, 15.0)
    assert is_sc is True, "Small cap detection failed."
    assert is_entry is True, "Small cap isolated entry logic failed."

    dummy_row_err = {'rsi': np.nan, 'dev25': 'invalid', 'rs_21': None}
    try:
        is_entry, score, is_sc = AdvancedStrategyAnalyzer.evaluate_entry(dummy_row_err, "スイング", 0.0, 15.0)
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data: {e}")
        
    try:
        AdvancedStrategyAnalyzer.evaluate_entry("invalid_type", "スイング", 0.0, 15.0) # type: ignore
        assert False, "evaluate_entry should raise TypeError for non-dict input"
    except TypeError:
        pass

    debug_log("All tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory '{data_dir}' not found. Please run data_fetcher.py first.")
            
        print("\n==================================================")
        print(" 🚀 STARTING PORTFOLIO CROSS-SECTIONAL BACKTEST")
        print("==================================================")
        
        STARTING_CAPITAL = 1000000.0
        MAX_CONCURRENT_POSITIONS = 5
        
        tester = PortfolioBacktester(data_dir=data_dir, initial_cash=STARTING_CAPITAL, max_positions=MAX_CONCURRENT_POSITIONS)
        res = tester.run()
        
        print(f"\n==================================================")
        print(f" 📊 PORTFOLIO SIMULATION RESULTS (354% Core + Quota Control)")
        print(f"==================================================")
        print(f" ▶ 初期資金 (Initial Cash) : ¥{int(res['Initial_Cash']):,}")
        print(f" ▶ 最終資産 (Final Cash)   : ¥{int(res['Final_Cash']):,}")
        print(f" ▶ 純利益 (Net Profit)     : ¥{int(res['Net_Profit']):,}")
        print(f" ▶ 総利回り (Return)       : {res['Return']}")
        print(f" ▶ 最大下落率 (MDD)        : {res['MDD']}")
        print(f" ▶ 総取引回数 (Trades)     : {res['Total_Trades']} 回")
        
        st = res['Stats']
        sm_rate = (st['limit_exec_small'] / st['limit_placed_small']) * 100 if st['limit_placed_small'] > 0 else 0
        lg_rate = (st['limit_exec_large'] / st['limit_placed_large']) * 100 if st['limit_placed_large'] > 0 else 0
        ts_sm_win_rate = (st['ts_small_win'] / st['ts_small']) * 100 if st['ts_small'] > 0 else 0
        
        print(f"==================================================")
        print(f" 🔬 アドバイザリー検証レポート")
        print(f" [1] 指値の約定率 (大型:0.3ATR / 小型:0.6ATR)")
        print(f"     - 小型株(ゲリラ) : {st['limit_exec_small']}/{st['limit_placed_small']} ({sm_rate:.1f}%)")
        print(f"     - 大型株(600%版) : {st['limit_exec_large']}/{st['limit_placed_large']} ({lg_rate:.1f}%)")
        print(f" [2] タイムストップと小型株の握力 (小型7日/3.0ATR vs 大型10日/2.5ATR)")
        print(f"     - 小型特例発動回数 : {st['ts_small']} 回 (うち微益撤退: {ts_sm_win_rate:.1f}%)")
        print(f"     - 大型通常発動回数 : {st['ts_large']} 回")
        print(f" [3] ポートフォリオ保護機能 (クオータ制)")
        print(f"     - 小型株の同時保有上限ブロック : {st['sc_quota_blocked']} 回")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
