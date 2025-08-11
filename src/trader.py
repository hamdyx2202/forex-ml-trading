import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from loguru import logger
import time
from src.data_collector import MT5DataCollector
from src.feature_engineer import FeatureEngineer
from src.predictor import Predictor
from src.risk_manager import RiskManager
from src.advanced_learner import AdvancedLearner
from src.continuous_learner import ContinuousLearner


class Trader:
    """نظام التداول الآلي الرئيسي"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.data_collector = MT5DataCollector(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.predictor = Predictor(config_path)
        self.risk_manager = RiskManager(config_path)
        self.advanced_learner = AdvancedLearner(config_path)
        self.continuous_learner = ContinuousLearner(config_path)
        
        logger.add("logs/trader.log", rotation="1 day", retention="30 days")
        
        self.is_running = False
        self.last_signal_time = {}
        
    def connect(self) -> bool:
        """الاتصال بـ MetaTrader 5"""
        return self.data_collector.connect_mt5()
    
    def disconnect(self):
        """قطع الاتصال بـ MetaTrader 5"""
        self.data_collector.disconnect_mt5()
    
    def get_account_info(self) -> Dict:
        """الحصول على معلومات الحساب"""
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'profit': account_info.profit,
            'currency': account_info.currency
        }
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """الحصول على السعر الحالي"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': (tick.ask - tick.bid) / self._get_pip_value(symbol),
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   stop_loss: float, take_profit: float) -> Optional[int]:
        """وضع أمر تداول"""
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
        
        # Get current price
        price_info = self.get_current_price(symbol)
        if not price_info:
            return None
        
        # Prepare request
        if order_type == "BUY":
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = price_info['ask']
        else:  # SELL
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = price_info['bid']
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "ML Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order placed successfully: {result.order}")
        
        # Add to risk manager
        self.risk_manager.add_position(
            str(result.order),
            symbol,
            order_type,
            volume,
            price,
            stop_loss,
            take_profit
        )
        
        return result.order
    
    def close_position(self, position_id: int) -> bool:
        """إغلاق صفقة"""
        position = mt5.positions_get(ticket=position_id)
        
        if not position:
            logger.error(f"Position {position_id} not found")
            return False
        
        position = position[0]
        
        # Prepare close request
        symbol = position.symbol
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position_id,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "ML Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.retcode}")
            return False
        
        # Update risk manager
        self.risk_manager.remove_position(str(position_id), price)
        
        # التعلم من نتيجة الصفقة
        trade_result = {
            'trade_id': str(position_id),
            'symbol': position.symbol,
            'timeframe': 'H1',  # يمكن تحسينها
            'entry_time': datetime.fromtimestamp(position.time),
            'exit_time': datetime.now(),
            'direction': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
            'entry_price': position.price_open,
            'exit_price': price,
            'pnl_pips': (price - position.price_open) / 0.0001 if position.type == mt5.POSITION_TYPE_BUY else (position.price_open - price) / 0.0001,
            'volume': position.volume
        }
        
        # تمرير النتيجة للتعلم المستمر
        self.continuous_learner.learn_from_trade(trade_result)
        
        logger.info(f"Position {position_id} closed successfully")
        return True
    
    def get_open_positions(self) -> List[Dict]:
        """الحصول على الصفقات المفتوحة"""
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        positions_list = []
        for pos in positions:
            positions_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'open_time': datetime.fromtimestamp(pos.time)
            })
        
        return positions_list
    
    def check_and_update_positions(self):
        """فحص وتحديث الصفقات المفتوحة"""
        positions = self.get_open_positions()
        
        for position in positions:
            symbol = position['symbol']
            
            # Get latest data
            df = self.data_collector.get_latest_data(symbol, "M5", limit=100)
            if df.empty:
                continue
            
            # Calculate ATR for trailing stop
            df = self.feature_engineer.add_technical_indicators(df)
            current_atr = df['atr_14'].iloc[-1]
            
            # Update trailing stop
            new_sl = self.risk_manager.update_trailing_stop(
                str(position['ticket']),
                position['current_price'],
                current_atr
            )
            
            if new_sl:
                self._modify_position(position['ticket'], new_sl, position['tp'])
    
    def _modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """تعديل صفقة"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
            "magic": 234000,
            "comment": "ML Bot Modify"
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify position: {result.retcode}")
            return False
        
        logger.info(f"Position {ticket} modified: SL={sl:.5f}")
        return True
    
    def analyze_and_trade(self, symbol: str, timeframe: str):
        """تحليل السوق واتخاذ قرار التداول باستخدام التعلم المتقدم"""
        logger.info(f"Analyzing {symbol} {timeframe} with advanced learning")
        
        # Check if we recently processed this symbol
        key = f"{symbol}_{timeframe}"
        if key in self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time[key]).seconds
            if time_since_last < 300:  # 5 minutes cooldown
                return
        
        # Get account info
        account_info = self.get_account_info()
        if not account_info:
            logger.error("Failed to get account info")
            return
        
        # البحث عن فرص عالية الجودة باستخدام التعلم المتقدم
        opportunities = self.advanced_learner.find_high_quality_opportunities(symbol, timeframe)
        
        if not opportunities:
            # إذا لم توجد فرص من التعلم المتقدم، استخدم النموذج العادي
            prediction = self.predictor.predict_latest(symbol, timeframe)
            if not prediction:
                logger.warning(f"No opportunities found for {symbol} {timeframe}")
                return
        else:
            # استخدم أفضل فرصة من التعلم المتقدم
            best_opportunity = opportunities[0]
            prediction = {
                'recommendation': 'BUY' if best_opportunity['direction'] == 'BUY' else 'SELL',
                'confidence': best_opportunity['confidence'],
                'current_price': best_opportunity['price'],
                'reasons': best_opportunity['reasons']
            }
            logger.info(f"Found high quality opportunity: {best_opportunity['direction']} with {best_opportunity['confidence']:.1%} confidence")
        
        # Check if we should trade
        if prediction['recommendation'] in ['NO_TRADE', 'HOLD']:
            logger.info(f"No trade signal for {symbol}: {prediction['recommendation']}")
            return
        
        # Determine direction
        if prediction['recommendation'] in ['BUY', 'STRONG_BUY']:
            direction = 'BUY'
        else:
            direction = 'SELL'
        
        # Get current price and ATR
        current_price = prediction['current_price']
        
        # Get ATR from latest data
        df = self.data_collector.get_latest_data(symbol, timeframe, limit=100)
        df = self.feature_engineer.add_technical_indicators(df)
        current_atr = df['atr_14'].iloc[-1]
        
        # Validate trade
        validation = self.risk_manager.validate_trade(
            symbol,
            direction,
            prediction['confidence'],
            account_info['balance'],
            current_price,
            current_atr
        )
        
        if not validation['valid']:
            logger.warning(f"Trade validation failed: {validation['reasons']}")
            return
        
        # Place order
        order_id = self.place_order(
            symbol,
            direction,
            validation['position_size'],
            validation['stop_loss'],
            validation['take_profit']
        )
        
        if order_id:
            self.last_signal_time[key] = datetime.now()
            logger.info(f"Trade executed: {symbol} {direction} - Order ID: {order_id}")
    
    def run_trading_cycle(self):
        """تشغيل دورة تداول واحدة"""
        if not self.is_running:
            return
        
        logger.info("Starting trading cycle")
        
        # Update positions
        self.check_and_update_positions()
        
        # Analyze each pair
        for symbol in self.config['trading']['pairs']:
            for timeframe in self.config['trading']['timeframes']:
                try:
                    self.analyze_and_trade(symbol, timeframe)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} {timeframe}: {str(e)}")
        
        logger.info("Trading cycle completed")
    
    def start_trading(self):
        """بدء التداول الآلي"""
        logger.info("Starting automated trading")
        
        if not self.connect():
            logger.error("Failed to connect to MT5")
            return
        
        self.is_running = True
        
        while self.is_running:
            try:
                self.run_trading_cycle()
                
                # Wait for next cycle
                time.sleep(60)  # 1 minute between cycles
                
            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)
        
        self.stop_trading()
    
    def stop_trading(self):
        """إيقاف التداول الآلي"""
        logger.info("Stopping automated trading")
        self.is_running = False
        self.disconnect()
    
    def _get_pip_value(self, symbol: str) -> float:
        """الحصول على قيمة النقطة"""
        with open("config/pairs.json", 'r') as f:
            pairs_config = json.load(f)
        
        return pairs_config.get(symbol, {}).get('pip_value', 0.0001)
    
    def get_trading_summary(self) -> Dict:
        """الحصول على ملخص التداول"""
        account_info = self.get_account_info()
        positions = self.get_open_positions()
        exposure = self.risk_manager.get_exposure_summary()
        performance = self.risk_manager.get_performance_stats()
        
        return {
            'account': account_info,
            'open_positions': positions,
            'exposure': exposure,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # مثال على الاستخدام
    trader = Trader()
    
    # Start trading
    # trader.start_trading()
    
    # Or run single cycle
    # trader.connect()
    # trader.run_trading_cycle()
    # trader.disconnect()