import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from loguru import logger
import sqlite3


class RiskManager:
    """إدارة المخاطر وحساب أحجام الصفقات"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        with open("config/pairs.json", 'r') as f:
            self.pairs_config = json.load(f)
        
        self.db_path = self.config["database"]["path"]
        
        logger.add("logs/risk_manager.log", rotation="1 day", retention="30 days")
        
        # Risk parameters
        self.max_risk_per_trade = self.config['trading']['risk_per_trade']
        self.max_daily_loss = self.config['trading']['max_daily_loss']
        self.max_positions = self.config['trading']['max_positions']
        self.sl_atr_multiplier = self.config['trading']['stop_loss_atr_multiplier']
        self.tp_ratio = self.config['trading']['take_profit_ratio']
        
        # Track daily performance
        self.daily_pnl = 0
        self.open_positions = {}
        
    def calculate_position_size(self, account_balance: float, stop_loss_pips: float, 
                              symbol: str) -> float:
        """حساب حجم الصفقة بناء على المخاطر"""
        # Get pair configuration
        pair_config = self.pairs_config.get(symbol, {})
        pip_value = pair_config.get('pip_value', 0.0001)
        min_lot = pair_config.get('min_lot', 0.01)
        max_lot = pair_config.get('max_lot', 100.0)
        lot_step = pair_config.get('lot_step', 0.01)
        
        # Calculate risk amount
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate position size
        # Formula: Position Size = Risk Amount / (Stop Loss in Pips × Pip Value × Contract Size)
        # Assuming standard contract size of 100,000
        contract_size = 100000
        
        if stop_loss_pips > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value * contract_size)
        else:
            position_size = min_lot
        
        # Round to lot step
        position_size = round(position_size / lot_step) * lot_step
        
        # Apply limits
        position_size = max(min_lot, min(position_size, max_lot))
        
        logger.info(f"Calculated position size for {symbol}: {position_size} lots")
        logger.info(f"Risk: ${risk_amount:.2f}, SL: {stop_loss_pips} pips")
        
        return position_size
    
    def calculate_stop_loss(self, current_price: float, atr: float, direction: str, 
                          symbol: str) -> Tuple[float, float]:
        """حساب مستوى وقف الخسارة"""
        pair_config = self.pairs_config.get(symbol, {})
        pip_value = pair_config.get('pip_value', 0.0001)
        
        # Calculate stop loss distance
        sl_distance = atr * self.sl_atr_multiplier
        
        if direction == "BUY":
            stop_loss = current_price - sl_distance
        else:  # SELL
            stop_loss = current_price + sl_distance
        
        # Calculate stop loss in pips
        sl_pips = abs(current_price - stop_loss) / pip_value
        
        return stop_loss, sl_pips
    
    def calculate_take_profit(self, current_price: float, stop_loss: float, 
                            direction: str) -> float:
        """حساب مستوى جني الأرباح"""
        sl_distance = abs(current_price - stop_loss)
        tp_distance = sl_distance * self.tp_ratio
        
        if direction == "BUY":
            take_profit = current_price + tp_distance
        else:  # SELL
            take_profit = current_price - tp_distance
        
        return take_profit
    
    def check_daily_loss_limit(self, account_balance: float) -> bool:
        """التحقق من حد الخسارة اليومية"""
        max_daily_loss_amount = account_balance * self.max_daily_loss
        
        if abs(self.daily_pnl) >= max_daily_loss_amount and self.daily_pnl < 0:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    def check_position_limit(self) -> bool:
        """التحقق من حد عدد الصفقات المفتوحة"""
        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Maximum positions limit reached: {len(self.open_positions)}")
            return False
        
        return True
    
    def check_correlation_risk(self, symbol: str) -> bool:
        """التحقق من مخاطر الارتباط بين الأزواج"""
        # Define correlated pairs
        correlations = {
            "EURUSD": ["GBPUSD", "EURGBP"],
            "GBPUSD": ["EURUSD", "EURGBP"],
            "XAUUSD": ["XAGUSD"],
            "USDJPY": ["EURJPY", "GBPJPY"]
        }
        
        correlated_pairs = correlations.get(symbol, [])
        
        # Check if we already have positions in correlated pairs
        for position_symbol in self.open_positions:
            if position_symbol in correlated_pairs:
                logger.warning(f"Correlated position already exists: {position_symbol}")
                return False
        
        return True
    
    def validate_trade(self, symbol: str, direction: str, confidence: float, 
                      account_balance: float, current_price: float, atr: float) -> Dict:
        """التحقق من صحة الصفقة قبل التنفيذ"""
        validation_result = {
            'valid': True,
            'reasons': [],
            'position_size': 0,
            'stop_loss': 0,
            'take_profit': 0
        }
        
        # Check confidence level
        if confidence < self.config['trading']['min_confidence']:
            validation_result['valid'] = False
            validation_result['reasons'].append(f"Low confidence: {confidence:.2%}")
            return validation_result
        
        # Check daily loss limit
        if not self.check_daily_loss_limit(account_balance):
            validation_result['valid'] = False
            validation_result['reasons'].append("Daily loss limit reached")
            return validation_result
        
        # Check position limit
        if not self.check_position_limit():
            validation_result['valid'] = False
            validation_result['reasons'].append("Maximum positions limit reached")
            return validation_result
        
        # Check correlation risk
        if not self.check_correlation_risk(symbol):
            validation_result['valid'] = False
            validation_result['reasons'].append("Correlated position already exists")
            return validation_result
        
        # Calculate risk parameters
        stop_loss, sl_pips = self.calculate_stop_loss(current_price, atr, direction, symbol)
        take_profit = self.calculate_take_profit(current_price, stop_loss, direction)
        position_size = self.calculate_position_size(account_balance, sl_pips, symbol)
        
        # Check if position size is valid
        pair_config = self.pairs_config.get(symbol, {})
        if position_size < pair_config.get('min_lot', 0.01):
            validation_result['valid'] = False
            validation_result['reasons'].append("Position size too small")
            return validation_result
        
        # All checks passed
        validation_result['position_size'] = position_size
        validation_result['stop_loss'] = stop_loss
        validation_result['take_profit'] = take_profit
        
        logger.info(f"Trade validated for {symbol}: {direction}")
        logger.info(f"Position size: {position_size}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return validation_result
    
    def add_position(self, position_id: str, symbol: str, direction: str, 
                    volume: float, open_price: float, stop_loss: float, 
                    take_profit: float):
        """إضافة صفقة مفتوحة"""
        self.open_positions[position_id] = {
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'open_price': open_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'open_time': datetime.now()
        }
        
        logger.info(f"Added position {position_id} to tracking")
    
    def remove_position(self, position_id: str, close_price: float):
        """إزالة صفقة مغلقة وحساب الربح/الخسارة"""
        if position_id not in self.open_positions:
            logger.warning(f"Position {position_id} not found in tracking")
            return
        
        position = self.open_positions[position_id]
        
        # Calculate P&L
        if position['direction'] == "BUY":
            pnl_pips = close_price - position['open_price']
        else:
            pnl_pips = position['open_price'] - close_price
        
        # Convert to monetary value (simplified)
        pair_config = self.pairs_config.get(position['symbol'], {})
        pip_value = pair_config.get('pip_value', 0.0001)
        contract_size = 100000
        
        pnl = pnl_pips / pip_value * position['volume'] * contract_size * pip_value
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Remove from tracking
        del self.open_positions[position_id]
        
        logger.info(f"Closed position {position_id}: P&L = ${pnl:.2f}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        
        # Save to database
        self._save_trade_history(position_id, position, close_price, pnl)
    
    def update_trailing_stop(self, position_id: str, current_price: float, atr: float) -> Optional[float]:
        """تحديث وقف الخسارة المتحرك"""
        if position_id not in self.open_positions:
            return None
        
        position = self.open_positions[position_id]
        
        # Calculate new stop loss
        trail_distance = atr * 1.5  # Less than initial SL
        
        if position['direction'] == "BUY":
            new_sl = current_price - trail_distance
            # Only update if new SL is higher than current
            if new_sl > position['stop_loss']:
                position['stop_loss'] = new_sl
                logger.info(f"Updated trailing stop for {position_id}: {new_sl:.5f}")
                return new_sl
        else:  # SELL
            new_sl = current_price + trail_distance
            # Only update if new SL is lower than current
            if new_sl < position['stop_loss']:
                position['stop_loss'] = new_sl
                logger.info(f"Updated trailing stop for {position_id}: {new_sl:.5f}")
                return new_sl
        
        return None
    
    def get_exposure_summary(self) -> Dict:
        """الحصول على ملخص التعرض للمخاطر"""
        exposure = {
            'total_positions': len(self.open_positions),
            'by_symbol': {},
            'by_direction': {'BUY': 0, 'SELL': 0},
            'total_volume': 0
        }
        
        for position_id, position in self.open_positions.items():
            symbol = position['symbol']
            
            # By symbol
            if symbol not in exposure['by_symbol']:
                exposure['by_symbol'][symbol] = {
                    'count': 0,
                    'volume': 0,
                    'direction': []
                }
            
            exposure['by_symbol'][symbol]['count'] += 1
            exposure['by_symbol'][symbol]['volume'] += position['volume']
            exposure['by_symbol'][symbol]['direction'].append(position['direction'])
            
            # By direction
            exposure['by_direction'][position['direction']] += 1
            
            # Total volume
            exposure['total_volume'] += position['volume']
        
        return exposure
    
    def reset_daily_stats(self):
        """إعادة تعيين الإحصائيات اليومية"""
        self.daily_pnl = 0
        logger.info("Daily statistics reset")
    
    def _save_trade_history(self, trade_id: str, position: Dict, close_price: float, pnl: float):
        """حفظ تاريخ الصفقات في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                volume REAL NOT NULL,
                open_price REAL NOT NULL,
                close_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                open_time TIMESTAMP NOT NULL,
                close_time TIMESTAMP NOT NULL,
                pnl REAL NOT NULL
            )
        """)
        
        # Insert trade
        cursor.execute("""
            INSERT INTO trade_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            position['symbol'],
            position['direction'],
            position['volume'],
            position['open_price'],
            close_price,
            position['stop_loss'],
            position['take_profit'],
            position['open_time'],
            datetime.now(),
            pnl
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """الحصول على إحصائيات الأداء"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trades from last N days
        query = """
            SELECT * FROM trade_history 
            WHERE close_time > datetime('now', '-{} days')
            ORDER BY close_time DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'average_win': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        # Calculate statistics
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        stats = {
            'total_trades': len(df),
            'win_rate': len(winning_trades) / len(df) if len(df) > 0 else 0,
            'average_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'sharpe_ratio': self._calculate_sharpe_ratio(df['pnl'].values)
        }
        
        return stats
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """حساب نسبة شارب"""
        if len(returns) < 2:
            return 0
        
        # Assuming daily returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualize (252 trading days)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio