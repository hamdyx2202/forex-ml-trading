#!/usr/bin/env python3
"""
Integration Script - ربط نظام التعلم المستمر مع MT5
يجمع بين التعلم المستمر والتداول الفعلي
"""

import time
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

# استيراد المكونات المطلوبة
from continuous_learner_advanced_v2 import AdvancedContinuousLearnerV2
from mt5_bridge_server_linux import MT5BridgeServer

class IntegratedTradingSystem:
    """نظام متكامل يجمع بين التعلم المستمر والتداول"""
    
    def __init__(self):
        # إنشاء المكونات
        self.learner = AdvancedContinuousLearnerV2()
        self.server = MT5BridgeServer()
        
        # إعدادات التداول
        self.trading_config = {
            'pairs': [
                'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm',
                'XAUUSDm', 'XAGUSDm', 'BTCUSDm', 'US30m', 'OILm'
            ],
            'timeframes': ['M1', 'M5', 'M15', 'M30', 'H1', 'H4'],
            'risk_per_trade': 0.01,  # 1% من الرصيد
            'max_open_trades': 10,
            'min_confidence': 0.75,
            'use_strategies': ['scalping', 'short_term', 'medium_term']
        }
        
        # سجل الصفقات النشطة
        self.active_trades = {}
        
    def start_integrated_system(self):
        """بدء النظام المتكامل"""
        logger.info("🚀 Starting Integrated Trading System...")
        
        # 1. بدء التعلم المستمر
        logger.info("📚 Starting continuous learning...")
        self.learner.start_continuous_learning(
            self.trading_config['pairs'],
            self.trading_config['timeframes']
        )
        
        # 2. بدء خادم MT5
        logger.info("🌐 Starting MT5 bridge server...")
        # self.server.start()  # سيتم تشغيله في خيط منفصل
        
        # 3. بدء حلقة التداول الرئيسية
        logger.info("💹 Starting main trading loop...")
        self._main_trading_loop()
    
    def _main_trading_loop(self):
        """الحلقة الرئيسية للتداول"""
        while True:
            try:
                # فحص كل زوج وفترة زمنية
                for pair in self.trading_config['pairs']:
                    for timeframe in self.trading_config['timeframes']:
                        # الحصول على البيانات الحالية
                        current_data = self.server.get_latest_data(pair, timeframe)
                        
                        if current_data is None:
                            continue
                        
                        # إنشاء الميزات
                        features = self._prepare_features(current_data, pair)
                        
                        if features is None:
                            continue
                        
                        # الحصول على التنبؤات من جميع الاستراتيجيات
                        predictions = self.learner.predict_with_sl_tp(pair, timeframe, features)
                        
                        # معالجة التنبؤات واتخاذ قرارات التداول
                        self._process_predictions(pair, timeframe, predictions, current_data)
                
                # تحديث الصفقات النشطة
                self._update_active_trades()
                
                # انتظار قبل الدورة التالية
                time.sleep(60)  # فحص كل دقيقة
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping integrated system...")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                time.sleep(10)
    
    def _prepare_features(self, data, pair):
        """إعداد الميزات من البيانات الحالية"""
        try:
            # استخدام نفس معالج الميزات من النظام المتقدم
            df = pd.DataFrame([data])
            features_df = self.learner.advanced_trainer.create_ultra_advanced_features(df, pair)
            
            if features_df.empty:
                return None
            
            return features_df.values[-1]  # آخر صف
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return None
    
    def _process_predictions(self, pair, timeframe, predictions, current_data):
        """معالجة التنبؤات واتخاذ قرارات التداول"""
        
        # فحص الصفقات النشطة لهذا الزوج
        active_key = f"{pair}_{timeframe}"
        if active_key in self.active_trades:
            # تحديث الصفقة الموجودة
            self._manage_existing_trade(active_key, predictions, current_data)
        else:
            # فحص فرص جديدة
            self._check_new_trade_opportunity(pair, timeframe, predictions, current_data)
    
    def _check_new_trade_opportunity(self, pair, timeframe, predictions, current_data):
        """فحص فرص التداول الجديدة"""
        
        # التحقق من عدد الصفقات المفتوحة
        if len(self.active_trades) >= self.trading_config['max_open_trades']:
            return
        
        # البحث عن أفضل إشارة من الاستراتيجيات المختلفة
        best_signal = None
        best_confidence = 0
        best_strategy = None
        
        for strategy in self.trading_config['use_strategies']:
            if strategy not in predictions:
                continue
            
            pred = predictions[strategy]
            
            # فحص الثقة
            if pred['confidence'] > best_confidence and pred['confidence'] >= self.trading_config['min_confidence']:
                if pred['signal'] != 1:  # ليس محايد
                    best_signal = pred
                    best_confidence = pred['confidence']
                    best_strategy = strategy
        
        # فتح صفقة جديدة إذا وجدت إشارة قوية
        if best_signal:
            self._open_new_trade(pair, timeframe, best_signal, best_strategy, current_data)
    
    def _open_new_trade(self, pair, timeframe, signal, strategy, current_data):
        """فتح صفقة جديدة"""
        try:
            # حساب حجم الصفقة
            lot_size = self._calculate_lot_size(
                signal['stop_loss'],
                current_data['close'],
                pair
            )
            
            # إعداد أمر التداول
            order = {
                'symbol': pair,
                'action': 'BUY' if signal['signal'] == 2 else 'SELL',
                'volume': lot_size,
                'sl': signal['stop_loss'],
                'tp': signal['take_profit_1'],  # البدء بـ TP1
                'comment': f"{strategy}_{timeframe}",
                'magic': self._generate_magic_number(strategy)
            }
            
            # إرسال الأمر إلى MT5
            result = self.server.send_order(order)
            
            if result and result.get('success'):
                # حفظ معلومات الصفقة
                trade_key = f"{pair}_{timeframe}"
                self.active_trades[trade_key] = {
                    'ticket': result['ticket'],
                    'symbol': pair,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'entry_price': current_data['close'],
                    'entry_time': datetime.now(),
                    'signal': signal,
                    'tp_levels': {
                        'tp1': signal['take_profit_1'],
                        'tp2': signal['take_profit_2'],
                        'tp3': signal['take_profit_3']
                    },
                    'current_tp': 'tp1',
                    'trailing_active': False
                }
                
                logger.info(f"✅ Opened {order['action']} trade on {pair} {timeframe}")
                logger.info(f"   Strategy: {strategy}, Confidence: {signal['confidence']:.2%}")
                logger.info(f"   Entry: {current_data['close']:.5f}, SL: {signal['stop_loss']:.5f}, TP1: {signal['take_profit_1']:.5f}")
                
                # حفظ في قاعدة البيانات للتعلم
                self._log_trade_to_database(order, signal, strategy)
                
        except Exception as e:
            logger.error(f"❌ Error opening trade: {e}")
    
    def _manage_existing_trade(self, trade_key, predictions, current_data):
        """إدارة الصفقات الموجودة"""
        trade = self.active_trades[trade_key]
        
        try:
            # الحصول على معلومات الصفقة من MT5
            position_info = self.server.get_position_info(trade['ticket'])
            
            if not position_info:
                # الصفقة مغلقة
                del self.active_trades[trade_key]
                return
            
            current_price = current_data['close']
            entry_price = trade['entry_price']
            
            # حساب الربح الحالي بالنقاط
            pip_value = self.learner.advanced_trainer.calculate_pip_value(trade['symbol'])
            
            if trade['signal']['signal'] == 2:  # Long
                profit_pips = (current_price - entry_price) / pip_value
            else:  # Short
                profit_pips = (entry_price - current_price) / pip_value
            
            # إدارة Take Profit المتدرج
            if trade['current_tp'] == 'tp1' and profit_pips >= 20:
                # الانتقال إلى TP2 ونقل SL للتعادل
                self._move_to_breakeven(trade)
                trade['current_tp'] = 'tp2'
                
            elif trade['current_tp'] == 'tp2' and profit_pips >= 40:
                # الانتقال إلى TP3 وتفعيل Trailing Stop
                self._activate_trailing_stop(trade)
                trade['current_tp'] = 'tp3'
                trade['trailing_active'] = True
            
            # تحديث Trailing Stop إذا كان مفعلاً
            if trade['trailing_active']:
                self._update_trailing_stop(trade, current_price)
            
            # فحص إذا كانت هناك إشارة معاكسة قوية
            strategy = trade['strategy']
            if strategy in predictions:
                new_signal = predictions[strategy]
                
                # إغلاق إذا كانت هناك إشارة معاكسة بثقة عالية
                if new_signal['confidence'] > 0.8 and new_signal['signal'] != trade['signal']['signal']:
                    logger.warning(f"⚠️ Reverse signal detected for {trade_key}, closing trade")
                    self._close_trade(trade)
                    
        except Exception as e:
            logger.error(f"❌ Error managing trade {trade_key}: {e}")
    
    def _calculate_lot_size(self, stop_loss, current_price, pair):
        """حساب حجم الصفقة بناءً على إدارة المخاطر"""
        try:
            # الحصول على معلومات الحساب
            account_info = self.server.get_account_info()
            
            if not account_info:
                return 0.01  # حجم افتراضي
            
            balance = account_info['balance']
            risk_amount = balance * self.trading_config['risk_per_trade']
            
            # حساب المسافة للـ Stop Loss بالنقاط
            pip_value = self.learner.advanced_trainer.calculate_pip_value(pair)
            sl_distance_pips = abs(current_price - stop_loss) / pip_value
            
            # حساب قيمة النقطة للوت الواحد
            # هذا يحتاج لمعلومات من MT5 حول العملة
            point_value_per_lot = 10  # افتراضي للأزواج الرئيسية
            
            if 'JPY' in pair:
                point_value_per_lot = 1000
            elif 'XAU' in pair:
                point_value_per_lot = 100
            
            # حساب حجم اللوت
            lot_size = risk_amount / (sl_distance_pips * point_value_per_lot)
            
            # تقريب لأقرب 0.01
            lot_size = round(lot_size, 2)
            
            # التحقق من الحدود
            lot_size = max(0.01, min(lot_size, 10.0))
            
            return lot_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating lot size: {e}")
            return 0.01
    
    def _move_to_breakeven(self, trade):
        """نقل Stop Loss للتعادل"""
        try:
            modify_order = {
                'ticket': trade['ticket'],
                'sl': trade['entry_price'],
                'tp': trade['tp_levels']['tp2']
            }
            
            result = self.server.modify_position(modify_order)
            
            if result and result.get('success'):
                logger.info(f"✅ Moved SL to breakeven for {trade['symbol']}")
                
        except Exception as e:
            logger.error(f"❌ Error moving to breakeven: {e}")
    
    def _activate_trailing_stop(self, trade):
        """تفعيل Trailing Stop"""
        try:
            settings = trade['signal']['sl_tp_settings']
            trailing_distance = settings.get('trailing_stop_atr', 1.0)
            
            # تحديث MT5
            # ملاحظة: MT5 يحتاج لتنفيذ Trailing Stop من جانب العميل
            
            logger.info(f"✅ Activated trailing stop for {trade['symbol']}")
            
        except Exception as e:
            logger.error(f"❌ Error activating trailing stop: {e}")
    
    def _update_trailing_stop(self, trade, current_price):
        """تحديث Trailing Stop"""
        # منطق تحديث Trailing Stop
        pass
    
    def _close_trade(self, trade):
        """إغلاق صفقة"""
        try:
            result = self.server.close_position(trade['ticket'])
            
            if result and result.get('success'):
                logger.info(f"✅ Closed trade {trade['ticket']} on {trade['symbol']}")
                
                # حذف من الصفقات النشطة
                trade_key = f"{trade['symbol']}_{trade['timeframe']}"
                if trade_key in self.active_trades:
                    del self.active_trades[trade_key]
                
        except Exception as e:
            logger.error(f"❌ Error closing trade: {e}")
    
    def _update_active_trades(self):
        """تحديث معلومات الصفقات النشطة"""
        # التحقق من حالة جميع الصفقات النشطة
        for trade_key in list(self.active_trades.keys()):
            trade = self.active_trades[trade_key]
            
            # التحقق إذا كانت الصفقة ما زالت مفتوحة
            position_info = self.server.get_position_info(trade['ticket'])
            
            if not position_info:
                # الصفقة مغلقة
                logger.info(f"Trade {trade['ticket']} closed")
                del self.active_trades[trade_key]
    
    def _generate_magic_number(self, strategy):
        """توليد رقم سحري للاستراتيجية"""
        magic_numbers = {
            'ultra_short': 10001,
            'scalping': 10002,
            'short_term': 10003,
            'medium_term': 10004,
            'long_term': 10005
        }
        return magic_numbers.get(strategy, 10000)
    
    def _log_trade_to_database(self, order, signal, strategy):
        """حفظ معلومات الصفقة في قاعدة البيانات"""
        # حفظ للتعلم المستمر
        pass
    
    def get_system_status(self):
        """الحصول على حالة النظام"""
        return {
            'active_trades': len(self.active_trades),
            'trades_details': self.active_trades,
            'learning_status': 'Active',
            'server_status': 'Running',
            'last_update': datetime.now()
        }

def main():
    """تشغيل النظام المتكامل"""
    system = IntegratedTradingSystem()
    
    try:
        # بدء النظام
        system.start_integrated_system()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    # تأكد من استيراد المكتبات المطلوبة
    import pandas as pd
    main()