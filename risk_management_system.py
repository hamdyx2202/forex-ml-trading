#!/usr/bin/env python3
"""
💰 Risk Management System - نظام إدارة المخاطر المتقدم
🛡️ يحمي رأس المال ويحسن الربحية
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class RiskManagementSystem:
    """نظام إدارة المخاطر الذكي"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = 0.01  # 1% افتراضي
        self.max_daily_risk = 0.03      # 3% يومي
        self.max_weekly_risk = 0.06     # 6% أسبوعي
        self.max_correlation_exposure = 0.5  # 50% في أزواج مترابطة
        
        # تتبع الأداء
        self.daily_loss = 0
        self.weekly_loss = 0
        self.open_trades = {}
        self.trade_history = []
        self.daily_trades = []
        
        # معاملات الارتباط بين الأزواج
        self.correlations = {
            'EURUSD': {'GBPUSD': 0.85, 'USDCHF': -0.95, 'USDJPY': -0.30},
            'GBPUSD': {'EURUSD': 0.85, 'USDCHF': -0.80, 'USDJPY': -0.25},
            'USDJPY': {'USDCHF': 0.75, 'EURUSD': -0.30, 'GBPUSD': -0.25},
            'USDCHF': {'EURUSD': -0.95, 'GBPUSD': -0.80, 'USDJPY': 0.75},
            'AUDUSD': {'NZDUSD': 0.90, 'USDCAD': -0.75},
            'NZDUSD': {'AUDUSD': 0.90, 'USDCAD': -0.70},
            'USDCAD': {'AUDUSD': -0.75, 'NZDUSD': -0.70}
        }
        
        # إعدادات متقدمة
        self.enable_dynamic_risk = True
        self.enable_correlation_filter = True
        self.enable_time_filter = True
        self.enable_drawdown_protection = True
        
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, 
                              market_context=None, confidence=0.5):
        """حساب حجم الصفقة الأمثل"""
        try:
            # 1. التحقق من القيود الأساسية
            if not self._check_trading_allowed():
                return 0, "Trading not allowed due to risk limits"
            
            # 2. حساب المخاطرة الأساسية
            base_risk = self._calculate_base_risk(market_context, confidence)
            
            # 3. تعديل حسب الأداء
            performance_multiplier = self._get_performance_multiplier()
            
            # 4. تعديل حسب التقلبات
            volatility_multiplier = self._get_volatility_multiplier(market_context)
            
            # 5. تعديل حسب الارتباط
            correlation_multiplier = self._get_correlation_multiplier(symbol)
            
            # 6. المخاطرة النهائية
            final_risk = base_risk * performance_multiplier * volatility_multiplier * correlation_multiplier
            final_risk = max(0.001, min(0.02, final_risk))  # بين 0.1% و 2%
            
            # 7. حساب حجم الصفقة
            risk_amount = self.current_balance * final_risk
            
            # pip value وstop loss
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            stop_loss_pips = abs(entry_price - stop_loss_price) / pip_value
            
            if stop_loss_pips == 0:
                return 0, "Invalid stop loss"
            
            # حساب lot size
            # للفوركس: 1 لوت = 100,000 وحدة
            # قيمة النقطة للوت الواحد = 10$ لمعظم الأزواج
            pip_value_per_lot = 10  # دولار لكل نقطة لكل لوت
            
            if 'JPY' in symbol:
                pip_value_per_lot = 1000 / self._get_usdjpy_rate()
            
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
            
            # تقريب لأقرب 0.01 لوت
            lot_size = round(lot_size, 2)
            
            # الحد الأدنى والأقصى
            lot_size = max(0.01, min(lot_size, self._get_max_lot_size()))
            
            # 8. التحقق النهائي
            actual_risk = (lot_size * stop_loss_pips * pip_value_per_lot) / self.current_balance
            
            if actual_risk > self.max_risk_per_trade * 1.1:  # تسامح 10%
                lot_size = lot_size * (self.max_risk_per_trade / actual_risk)
                lot_size = round(lot_size, 2)
            
            logger.info(f"Position size calculated: {lot_size} lots, Risk: {actual_risk:.2%}")
            
            return lot_size, f"Risk: {actual_risk:.2%}"
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, f"Error: {str(e)}"
    
    def _check_trading_allowed(self):
        """التحقق من السماح بالتداول"""
        # تحقق من الخسارة اليومية
        if abs(self.daily_loss) >= self.current_balance * self.max_daily_risk:
            logger.warning("Daily loss limit reached")
            return False
        
        # تحقق من الخسارة الأسبوعية
        if abs(self.weekly_loss) >= self.current_balance * self.max_weekly_risk:
            logger.warning("Weekly loss limit reached")
            return False
        
        # تحقق من الـ drawdown
        if self.enable_drawdown_protection:
            current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
            if current_drawdown > 0.20:  # 20% max drawdown
                logger.warning("Maximum drawdown reached")
                return False
        
        return True
    
    def _calculate_base_risk(self, market_context, confidence):
        """حساب المخاطرة الأساسية"""
        base_risk = self.max_risk_per_trade
        
        if market_context:
            # تعديل حسب قوة السوق
            market_score = market_context.get('score', 0)
            
            if abs(market_score) >= 70:
                base_risk *= 1.5  # زيادة المخاطرة للإشارات القوية
            elif abs(market_score) >= 50:
                base_risk *= 1.2
            elif abs(market_score) <= 20:
                base_risk *= 0.5  # تقليل المخاطرة للإشارات الضعيفة
            
            # تعديل حسب جودة الجلسة
            session = market_context.get('session', {})
            if session.get('session_quality') == 'EXCELLENT':
                base_risk *= 1.2
            elif session.get('session_quality') == 'LOW':
                base_risk *= 0.7
            
            # تقليل المخاطرة وقت الأخبار
            if session.get('is_news_time', False):
                base_risk *= 0.5
        
        # تعديل حسب الثقة
        if confidence >= 0.8:
            base_risk *= 1.3
        elif confidence >= 0.7:
            base_risk *= 1.1
        elif confidence <= 0.6:
            base_risk *= 0.8
        
        return base_risk
    
    def _get_performance_multiplier(self):
        """حساب معامل الأداء"""
        if len(self.trade_history) < 10:
            return 1.0
        
        # حساب معدل الربح آخر 20 صفقة
        recent_trades = self.trade_history[-20:]
        winning_trades = sum(1 for t in recent_trades if t['profit'] > 0)
        win_rate = winning_trades / len(recent_trades)
        
        # حساب profit factor
        total_profit = sum(t['profit'] for t in recent_trades if t['profit'] > 0)
        total_loss = abs(sum(t['profit'] for t in recent_trades if t['profit'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 1.0
        
        # معامل الأداء
        if win_rate >= 0.7 and profit_factor >= 2.0:
            return 1.5  # أداء ممتاز
        elif win_rate >= 0.6 and profit_factor >= 1.5:
            return 1.2  # أداء جيد
        elif win_rate <= 0.3 or profit_factor <= 0.8:
            return 0.5  # أداء ضعيف
        else:
            return 1.0
    
    def _get_volatility_multiplier(self, market_context):
        """حساب معامل التقلبات"""
        if not market_context:
            return 1.0
        
        volatility = market_context.get('volatility', {})
        volatility_level = volatility.get('volatility_level', 'NORMAL')
        
        if volatility_level == 'VERY_HIGH':
            return 0.5  # تقليل المخاطرة بنسبة 50%
        elif volatility_level == 'HIGH':
            return 0.7
        elif volatility_level == 'LOW':
            return 1.2
        elif volatility_level == 'VERY_LOW':
            return 0.8  # الأسواق الهادئة جداً قد تكون خادعة
        else:
            return 1.0
    
    def _get_correlation_multiplier(self, symbol):
        """حساب معامل الارتباط"""
        if not self.enable_correlation_filter:
            return 1.0
        
        # حساب التعرض الحالي للأزواج المترابطة
        correlated_exposure = 0
        base_currency = symbol[:3]
        
        for open_symbol, trade in self.open_trades.items():
            if open_symbol == symbol:
                continue
            
            # التحقق من الارتباط
            correlation = self._get_correlation(symbol, open_symbol)
            
            if abs(correlation) > 0.7:
                # نفس الاتجاه مع ارتباط إيجابي أو اتجاه معاكس مع ارتباط سلبي
                if (correlation > 0 and trade['direction'] == 'BUY') or \
                   (correlation < 0 and trade['direction'] == 'SELL'):
                    correlated_exposure += abs(trade['risk'])
        
        # تقليل المخاطرة إذا كان التعرض عالي
        total_exposure = sum(abs(t['risk']) for t in self.open_trades.values())
        
        if correlated_exposure > 0:
            correlation_ratio = correlated_exposure / self.current_balance
            
            if correlation_ratio > 0.03:  # أكثر من 3% في أزواج مترابطة
                return 0.3
            elif correlation_ratio > 0.02:
                return 0.5
            elif correlation_ratio > 0.01:
                return 0.7
        
        return 1.0
    
    def _get_correlation(self, symbol1, symbol2):
        """الحصول على معامل الارتباط بين زوجين"""
        # تنظيف الرموز
        clean_symbol1 = symbol1.replace('m', '').replace('.ecn', '')
        clean_symbol2 = symbol2.replace('m', '').replace('.ecn', '')
        
        if clean_symbol1 in self.correlations:
            if clean_symbol2 in self.correlations[clean_symbol1]:
                return self.correlations[clean_symbol1][clean_symbol2]
        
        # محاولة عكسية
        if clean_symbol2 in self.correlations:
            if clean_symbol1 in self.correlations[clean_symbol2]:
                return self.correlations[clean_symbol2][clean_symbol1]
        
        # ارتباط افتراضي بناءً على العملات المشتركة
        if clean_symbol1[:3] == clean_symbol2[:3] or clean_symbol1[3:] == clean_symbol2[3:]:
            return 0.5  # ارتباط متوسط للعملات المشتركة
        
        return 0.0
    
    def _get_max_lot_size(self):
        """الحصول على الحد الأقصى لحجم اللوت"""
        # حسب رصيد الحساب
        if self.current_balance < 1000:
            return 0.1
        elif self.current_balance < 5000:
            return 0.5
        elif self.current_balance < 10000:
            return 1.0
        elif self.current_balance < 50000:
            return 5.0
        else:
            return 10.0
    
    def _get_usdjpy_rate(self):
        """الحصول على سعر USDJPY (مؤقت)"""
        return 110.0  # يجب تحديثه من البيانات الحقيقية
    
    def validate_trade_setup(self, symbol, direction, entry_price, 
                           stop_loss_price, take_profit_price, 
                           lot_size, market_context=None):
        """التحقق من صحة إعداد الصفقة"""
        validations = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'risk_score': 0
        }
        
        # 1. التحقق من نسبة المخاطرة/المكافأة
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        
        if risk > 0:
            risk_reward_ratio = reward / risk
            
            if risk_reward_ratio < 1.5:
                validations['warnings'].append(f"Low risk/reward ratio: {risk_reward_ratio:.2f}")
                validations['risk_score'] += 20
            elif risk_reward_ratio < 2.0:
                validations['warnings'].append(f"Moderate risk/reward ratio: {risk_reward_ratio:.2f}")
                validations['risk_score'] += 10
        else:
            validations['errors'].append("Invalid stop loss")
            validations['is_valid'] = False
        
        # 2. التحقق من المسافة للدعم/المقاومة
        if market_context:
            sr_levels = market_context.get('support_resistance', {})
            
            # التحقق من وضع SL
            if direction == 'BUY' and sr_levels.get('nearest_support'):
                support_price = sr_levels['nearest_support']['price']
                if stop_loss_price > support_price * 0.9995:  # SL قريب جداً من الدعم
                    validations['warnings'].append("Stop loss too close to support")
                    validations['risk_score'] += 15
            
            elif direction == 'SELL' and sr_levels.get('nearest_resistance'):
                resistance_price = sr_levels['nearest_resistance']['price']
                if stop_loss_price < resistance_price * 1.0005:  # SL قريب جداً من المقاومة
                    validations['warnings'].append("Stop loss too close to resistance")
                    validations['risk_score'] += 15
        
        # 3. التحقق من التعرض الإجمالي
        total_risk = sum(abs(t['risk']) for t in self.open_trades.values())
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        stop_loss_pips = abs(entry_price - stop_loss_price) / pip_value
        new_risk = (lot_size * stop_loss_pips * 10) / self.current_balance
        
        if total_risk + new_risk > 0.05:  # أكثر من 5% مخاطرة إجمالية
            validations['errors'].append("Total exposure too high")
            validations['is_valid'] = False
        
        # 4. التحقق من عدد الصفقات المفتوحة
        if len(self.open_trades) >= 5:
            validations['warnings'].append("Too many open trades")
            validations['risk_score'] += 25
        
        # 5. التحقق من الارتباط
        correlation_check = self._check_correlation_risk(symbol, direction)
        if correlation_check['risk'] == 'HIGH':
            validations['warnings'].append(correlation_check['message'])
            validations['risk_score'] += 30
        
        # 6. التحقق من الوقت
        time_check = self._check_time_restrictions()
        if not time_check['allowed']:
            validations['errors'].append(time_check['message'])
            validations['is_valid'] = False
        
        # 7. النتيجة النهائية
        if validations['risk_score'] >= 50:
            validations['warnings'].append("High risk score - consider reducing position or waiting")
        
        if not validations['is_valid']:
            logger.warning(f"Trade validation failed: {validations['errors']}")
        
        return validations
    
    def _check_correlation_risk(self, symbol, direction):
        """التحقق من مخاطر الارتباط"""
        high_correlation_pairs = []
        
        for open_symbol, trade in self.open_trades.items():
            correlation = self._get_correlation(symbol, open_symbol)
            
            # نفس الاتجاه مع ارتباط إيجابي عالي
            if correlation > 0.8 and trade['direction'] == direction:
                high_correlation_pairs.append(open_symbol)
            # اتجاه معاكس مع ارتباط سلبي عالي
            elif correlation < -0.8 and trade['direction'] != direction:
                high_correlation_pairs.append(open_symbol)
        
        if len(high_correlation_pairs) >= 2:
            return {
                'risk': 'HIGH',
                'message': f"High correlation with {', '.join(high_correlation_pairs)}"
            }
        elif len(high_correlation_pairs) == 1:
            return {
                'risk': 'MEDIUM',
                'message': f"Correlated with {high_correlation_pairs[0]}"
            }
        
        return {'risk': 'LOW', 'message': ''}
    
    def _check_time_restrictions(self):
        """التحقق من قيود الوقت"""
        if not self.enable_time_filter:
            return {'allowed': True, 'message': ''}
        
        current_time = datetime.now()
        
        # منع التداول في نهاية الأسبوع
        if current_time.weekday() == 4 and current_time.hour >= 20:  # الجمعة بعد 8 مساءً
            return {'allowed': False, 'message': 'No trading on Friday evening'}
        
        # منع التداول في بداية الأسبوع
        if current_time.weekday() == 0 and current_time.hour < 2:  # الاثنين قبل 2 صباحاً
            return {'allowed': False, 'message': 'No trading on Monday opening'}
        
        # تحذير من الأوقات منخفضة السيولة
        if current_time.hour >= 22 or current_time.hour < 6:
            if len(self.open_trades) >= 3:
                return {'allowed': False, 'message': 'Too many trades during low liquidity'}
        
        return {'allowed': True, 'message': ''}
    
    def update_balance(self, new_balance):
        """تحديث الرصيد"""
        profit_loss = new_balance - self.current_balance
        self.current_balance = new_balance
        
        # تحديث الخسائر اليومية والأسبوعية
        if profit_loss < 0:
            self.daily_loss += profit_loss
            self.weekly_loss += profit_loss
        
        # إعادة تعيين الخسائر في بداية اليوم/الأسبوع
        self._check_period_reset()
    
    def _check_period_reset(self):
        """التحقق من إعادة تعيين الفترات"""
        current_time = datetime.now()
        
        # إعادة تعيين يومي
        if hasattr(self, 'last_daily_reset'):
            if current_time.date() > self.last_daily_reset.date():
                self.daily_loss = 0
                self.daily_trades = []
                self.last_daily_reset = current_time
        else:
            self.last_daily_reset = current_time
        
        # إعادة تعيين أسبوعي
        if hasattr(self, 'last_weekly_reset'):
            if current_time.isocalendar()[1] > self.last_weekly_reset.isocalendar()[1]:
                self.weekly_loss = 0
                self.last_weekly_reset = current_time
        else:
            self.last_weekly_reset = current_time
    
    def register_trade(self, trade_info):
        """تسجيل صفقة جديدة"""
        trade_id = trade_info['id']
        
        # إضافة للصفقات المفتوحة
        self.open_trades[trade_id] = {
            'symbol': trade_info['symbol'],
            'direction': trade_info['direction'],
            'entry_price': trade_info['entry_price'],
            'stop_loss': trade_info['stop_loss'],
            'take_profit': trade_info['take_profit'],
            'lot_size': trade_info['lot_size'],
            'risk': trade_info['risk'],
            'entry_time': datetime.now()
        }
        
        # إضافة للصفقات اليومية
        self.daily_trades.append(trade_id)
        
        logger.info(f"Trade registered: {trade_id}")
    
    def close_trade(self, trade_id, exit_price):
        """إغلاق صفقة"""
        if trade_id not in self.open_trades:
            return
        
        trade = self.open_trades[trade_id]
        
        # حساب الربح/الخسارة
        pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
        
        if trade['direction'] == 'BUY':
            profit_pips = (exit_price - trade['entry_price']) / pip_value
        else:
            profit_pips = (trade['entry_price'] - exit_price) / pip_value
        
        profit_loss = profit_pips * trade['lot_size'] * 10  # $10 per pip per lot
        
        # تسجيل في التاريخ
        self.trade_history.append({
            'id': trade_id,
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'profit': profit_loss,
            'profit_pips': profit_pips,
            'duration': (datetime.now() - trade['entry_time']).total_seconds() / 3600,
            'exit_time': datetime.now()
        })
        
        # تحديث الرصيد
        self.update_balance(self.current_balance + profit_loss)
        
        # إزالة من الصفقات المفتوحة
        del self.open_trades[trade_id]
        
        logger.info(f"Trade closed: {trade_id}, Profit: ${profit_loss:.2f}")
    
    def get_risk_report(self):
        """تقرير المخاطر الحالي"""
        total_exposure = sum(abs(t['risk']) for t in self.open_trades.values())
        
        report = {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_pl': self.current_balance - self.initial_balance,
            'total_pl_percentage': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'open_trades': len(self.open_trades),
            'total_exposure': total_exposure,
            'total_exposure_percentage': (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'average_win': self._calculate_average_win(),
            'average_loss': self._calculate_average_loss(),
            'risk_status': self._get_risk_status()
        }
        
        return report
    
    def _calculate_max_drawdown(self):
        """حساب أقصى انخفاض"""
        if not self.trade_history:
            return 0
        
        peak = self.initial_balance
        max_dd = 0
        balance = self.initial_balance
        
        for trade in self.trade_history:
            balance += trade['profit']
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd * 100
    
    def _calculate_win_rate(self):
        """حساب معدل الربح"""
        if not self.trade_history:
            return 0
        
        winning_trades = sum(1 for t in self.trade_history if t['profit'] > 0)
        return (winning_trades / len(self.trade_history)) * 100
    
    def _calculate_profit_factor(self):
        """حساب عامل الربح"""
        if not self.trade_history:
            return 0
        
        gross_profit = sum(t['profit'] for t in self.trade_history if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trade_history if t['profit'] < 0))
        
        if gross_loss == 0:
            return gross_profit / 1 if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_average_win(self):
        """حساب متوسط الربح"""
        winning_trades = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        
        if not winning_trades:
            return 0
        
        return sum(winning_trades) / len(winning_trades)
    
    def _calculate_average_loss(self):
        """حساب متوسط الخسارة"""
        losing_trades = [abs(t['profit']) for t in self.trade_history if t['profit'] < 0]
        
        if not losing_trades:
            return 0
        
        return sum(losing_trades) / len(losing_trades)
    
    def _get_risk_status(self):
        """تحديد حالة المخاطر"""
        # التحقق من الخسائر
        daily_loss_pct = abs(self.daily_loss / self.current_balance) if self.current_balance > 0 else 0
        weekly_loss_pct = abs(self.weekly_loss / self.current_balance) if self.current_balance > 0 else 0
        
        # التحقق من التعرض
        total_exposure = sum(abs(t['risk']) for t in self.open_trades.values())
        exposure_pct = (total_exposure / self.current_balance) if self.current_balance > 0 else 0
        
        # تحديد الحالة
        if daily_loss_pct > 0.025 or weekly_loss_pct > 0.05:
            return 'CRITICAL'
        elif daily_loss_pct > 0.02 or weekly_loss_pct > 0.04 or exposure_pct > 0.04:
            return 'HIGH'
        elif daily_loss_pct > 0.01 or weekly_loss_pct > 0.02 or exposure_pct > 0.02:
            return 'MEDIUM'
        else:
            return 'LOW'