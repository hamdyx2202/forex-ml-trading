#!/usr/bin/env python3
"""
نظام SL/TP الديناميكي المتقدم
Advanced Dynamic SL/TP System

يتضمن:
1. SL/TP بناءً على الدعم والمقاومة
2. SL/TP بناءً على ATR
3. Break Even (نقل SL للدخول عند الربح)
4. Trailing Stop (وقف الخسارة المتحرك)
5. دعم لجميع أنواع الأدوات المالية
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from loguru import logger
import json

# استيراد حاسب الدعم والمقاومة
from support_resistance import SupportResistanceCalculator

class DynamicSLTPSystem:
    """نظام SL/TP الديناميكي المتقدم"""
    
    def __init__(self):
        """تهيئة النظام"""
        self.sr_calculator = SupportResistanceCalculator()
        
        # إعدادات افتراضية لكل نوع أداة
        self.instrument_settings = {
            'forex_major': {
                'min_sl_pips': 10,
                'max_sl_pips': 100,
                'default_rr': 2.0,
                'atr_multiplier': 1.5,
                'be_trigger_pips': 20,
                'trail_start_pips': 30,
                'trail_step_pips': 10
            },
            'forex_minor': {
                'min_sl_pips': 15,
                'max_sl_pips': 150,
                'default_rr': 2.0,
                'atr_multiplier': 2.0,
                'be_trigger_pips': 25,
                'trail_start_pips': 40,
                'trail_step_pips': 15
            },
            'metals': {
                'min_sl_pips': 50,
                'max_sl_pips': 500,
                'default_rr': 2.5,
                'atr_multiplier': 2.0,
                'be_trigger_pips': 100,
                'trail_start_pips': 150,
                'trail_step_pips': 50
            },
            'energy': {
                'min_sl_pips': 30,
                'max_sl_pips': 300,
                'default_rr': 2.0,
                'atr_multiplier': 2.5,
                'be_trigger_pips': 60,
                'trail_start_pips': 100,
                'trail_step_pips': 30
            },
            'indices': {
                'min_sl_points': 20,
                'max_sl_points': 200,
                'default_rr': 2.0,
                'atr_multiplier': 1.5,
                'be_trigger_points': 40,
                'trail_start_points': 60,
                'trail_step_points': 20
            },
            'crypto': {
                'min_sl_percent': 1.0,
                'max_sl_percent': 10.0,
                'default_rr': 3.0,
                'atr_multiplier': 2.5,
                'be_trigger_percent': 2.0,
                'trail_start_percent': 3.0,
                'trail_step_percent': 1.0
            },
            'stocks': {
                'min_sl_percent': 0.5,
                'max_sl_percent': 5.0,
                'default_rr': 2.5,
                'atr_multiplier': 2.0,
                'be_trigger_percent': 1.0,
                'trail_start_percent': 1.5,
                'trail_step_percent': 0.5
            }
        }
        
    def calculate_dynamic_sl_tp(self,
                               signal: str,
                               entry_price: float,
                               df: pd.DataFrame,
                               symbol: str,
                               method: str = 'hybrid',
                               custom_rr: Optional[float] = None) -> Dict:
        """
        حساب SL/TP الديناميكي
        
        Args:
            signal: BUY/SELL/STRONG_BUY/STRONG_SELL
            entry_price: سعر الدخول
            df: DataFrame مع OHLCV
            symbol: رمز الأداة
            method: 'sr' (دعم/مقاومة) أو 'atr' أو 'hybrid'
            custom_rr: نسبة مخاطرة/ربح مخصصة
            
        Returns:
            dict: معلومات SL/TP كاملة
        """
        try:
            # تحديد نوع الأداة والإعدادات
            instrument_type = self._get_instrument_type(symbol)
            settings = self.instrument_settings[instrument_type]
            
            # نسبة المخاطرة للربح
            risk_reward = custom_rr if custom_rr else settings['default_rr']
            
            # حساب بالطريقة المطلوبة
            if method == 'sr':
                result = self._calculate_sr_based(signal, entry_price, df, symbol, risk_reward, settings)
            elif method == 'atr':
                result = self._calculate_atr_based(signal, entry_price, df, symbol, risk_reward, settings)
            else:  # hybrid
                result = self._calculate_hybrid(signal, entry_price, df, symbol, risk_reward, settings)
            
            # إضافة معلومات إضافية
            result['instrument_type'] = instrument_type
            result['symbol'] = symbol
            result['entry_price'] = entry_price
            result['signal'] = signal
            result['timestamp'] = datetime.now().isoformat()
            
            # حساب النقاط/النسبة المئوية
            result = self._add_pip_calculations(result, symbol)
            
            # إضافة إعدادات Break Even و Trailing Stop
            result['break_even'] = self._calculate_break_even_levels(result, settings, instrument_type)
            result['trailing_stop'] = self._calculate_trailing_stop_levels(result, settings, instrument_type)
            
            logger.info(f"✅ Calculated dynamic SL/TP for {symbol}: "
                       f"SL={result['sl']:.5f}, TP={result['tp']:.5f}, "
                       f"Method={result['method']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating dynamic SL/TP: {str(e)}")
            return self._default_sl_tp(signal, entry_price, symbol)
    
    def _calculate_sr_based(self, signal: str, entry_price: float, df: pd.DataFrame, 
                           symbol: str, risk_reward: float, settings: Dict) -> Dict:
        """حساب SL/TP بناءً على الدعم والمقاومة"""
        # حساب مستويات الدعم والمقاومة
        sr_levels = self.sr_calculator.calculate_all_levels(df, symbol)
        
        # استخدام طريقة الدعم والمقاومة من الحاسب
        result = self.sr_calculator.calculate_dynamic_sl_tp(
            signal, entry_price, sr_levels, symbol, risk_reward
        )
        
        # التحقق من الحدود
        result = self._validate_sl_tp_limits(result, entry_price, symbol, settings)
        
        result['method'] = 'support_resistance'
        return result
    
    def _calculate_atr_based(self, signal: str, entry_price: float, df: pd.DataFrame, 
                            symbol: str, risk_reward: float, settings: Dict) -> Dict:
        """حساب SL/TP بناءً على ATR"""
        result = {
            'sl': None,
            'tp': None,
            'sl_distance': None,
            'tp_distance': None,
            'method': 'atr_based',
            'risk_reward': risk_reward
        }
        
        # حساب ATR
        if len(df) >= 14:
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
            
            if atr is None or pd.isna(atr):
                # حساب ATR يدوياً
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
            
            if atr and not pd.isna(atr):
                # حساب المسافات
                sl_distance = atr * settings['atr_multiplier']
                tp_distance = sl_distance * risk_reward
                
                if signal in ['BUY', 'STRONG_BUY']:
                    result['sl'] = entry_price - sl_distance
                    result['tp'] = entry_price + tp_distance
                    result['sl_distance'] = sl_distance
                    result['tp_distance'] = tp_distance
                else:  # SELL
                    result['sl'] = entry_price + sl_distance
                    result['tp'] = entry_price - tp_distance
                    result['sl_distance'] = sl_distance
                    result['tp_distance'] = tp_distance
        
        # التحقق من الحدود
        result = self._validate_sl_tp_limits(result, entry_price, symbol, settings)
        
        return result
    
    def _calculate_hybrid(self, signal: str, entry_price: float, df: pd.DataFrame, 
                         symbol: str, risk_reward: float, settings: Dict) -> Dict:
        """حساب هجين يجمع بين الدعم/المقاومة و ATR"""
        # حساب بالطريقتين
        sr_result = self._calculate_sr_based(signal, entry_price, df, symbol, risk_reward, settings)
        atr_result = self._calculate_atr_based(signal, entry_price, df, symbol, risk_reward, settings)
        
        # اختيار الأفضل
        result = {
            'method': 'hybrid',
            'risk_reward': risk_reward
        }
        
        # للـ SL: اختر الأقرب (أكثر أماناً)
        if sr_result['sl'] and atr_result['sl']:
            if signal in ['BUY', 'STRONG_BUY']:
                # للشراء: SL الأعلى (الأقرب)
                if sr_result['sl'] > atr_result['sl']:
                    result['sl'] = sr_result['sl']
                    result['sl_method'] = 'sr_based'
                else:
                    result['sl'] = atr_result['sl']
                    result['sl_method'] = 'atr_based'
            else:
                # للبيع: SL الأقل (الأقرب)
                if sr_result['sl'] < atr_result['sl']:
                    result['sl'] = sr_result['sl']
                    result['sl_method'] = 'sr_based'
                else:
                    result['sl'] = atr_result['sl']
                    result['sl_method'] = 'atr_based'
        elif sr_result['sl']:
            result['sl'] = sr_result['sl']
            result['sl_method'] = 'sr_based'
        else:
            result['sl'] = atr_result['sl']
            result['sl_method'] = 'atr_based'
        
        # للـ TP: اختر بناءً على R:R
        if result['sl']:
            if signal in ['BUY', 'STRONG_BUY']:
                result['sl_distance'] = entry_price - result['sl']
                result['tp_distance'] = result['sl_distance'] * risk_reward
                result['tp'] = entry_price + result['tp_distance']
            else:
                result['sl_distance'] = result['sl'] - entry_price
                result['tp_distance'] = result['sl_distance'] * risk_reward
                result['tp'] = entry_price - result['tp_distance']
            
            # تحقق من وجود مستوى دعم/مقاومة قريب من TP
            if sr_result['tp']:
                if signal in ['BUY', 'STRONG_BUY']:
                    if sr_result['tp'] > entry_price and sr_result['tp'] < result['tp']:
                        result['tp'] = sr_result['tp']
                        result['tp_method'] = 'sr_limited'
                    else:
                        result['tp_method'] = 'rr_based'
                else:
                    if sr_result['tp'] < entry_price and sr_result['tp'] > result['tp']:
                        result['tp'] = sr_result['tp']
                        result['tp_method'] = 'sr_limited'
                    else:
                        result['tp_method'] = 'rr_based'
        
        return result
    
    def _get_instrument_type(self, symbol: str) -> str:
        """تحديد نوع الأداة المالية"""
        symbol_upper = symbol.upper()
        
        # Forex
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        if any(pair in symbol_upper for pair in major_pairs):
            return 'forex_major'
        
        forex_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        if any(curr in symbol_upper for curr in forex_currencies) and len(symbol_upper) <= 8:
            return 'forex_minor'
        
        # Metals
        if any(metal in symbol_upper for metal in ['XAU', 'GOLD', 'XAG', 'SILVER']):
            return 'metals'
        
        # Energy
        if any(energy in symbol_upper for energy in ['OIL', 'WTI', 'BRENT', 'GAS']):
            return 'energy'
        
        # Indices
        if any(idx in symbol_upper for idx in ['US30', 'NAS100', 'SP500', 'DAX', 'FTSE', 'NIKKEI']):
            return 'indices'
        
        # Crypto
        if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'DOT']):
            return 'crypto'
        
        # Stocks
        if any(stock in symbol_upper for stock in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB']):
            return 'stocks'
        
        # Default to forex minor
        return 'forex_minor'
    
    def _validate_sl_tp_limits(self, result: Dict, entry_price: float, 
                              symbol: str, settings: Dict) -> Dict:
        """التحقق من حدود SL/TP المعقولة"""
        instrument_type = self._get_instrument_type(symbol)
        
        if instrument_type in ['forex_major', 'forex_minor']:
            # تحويل للنقاط
            pip_value = self._get_pip_value(symbol)
            
            min_sl = settings.get('min_sl_pips', 10) * pip_value
            max_sl = settings.get('max_sl_pips', 100) * pip_value
            
            if result['sl_distance']:
                if result['sl_distance'] < min_sl:
                    # توسيع SL
                    if result['signal'] in ['BUY', 'STRONG_BUY']:
                        result['sl'] = entry_price - min_sl
                    else:
                        result['sl'] = entry_price + min_sl
                    result['sl_distance'] = min_sl
                    result['tp_distance'] = min_sl * result['risk_reward']
                    
                elif result['sl_distance'] > max_sl:
                    # تقليص SL
                    if result['signal'] in ['BUY', 'STRONG_BUY']:
                        result['sl'] = entry_price - max_sl
                    else:
                        result['sl'] = entry_price + max_sl
                    result['sl_distance'] = max_sl
                    result['tp_distance'] = max_sl * result['risk_reward']
        
        elif instrument_type == 'crypto':
            # للعملات الرقمية: نسبة مئوية
            min_sl_pct = settings.get('min_sl_percent', 1.0) / 100
            max_sl_pct = settings.get('max_sl_percent', 10.0) / 100
            
            if result['sl_distance']:
                sl_pct = result['sl_distance'] / entry_price
                
                if sl_pct < min_sl_pct:
                    result['sl_distance'] = entry_price * min_sl_pct
                    result['tp_distance'] = result['sl_distance'] * result['risk_reward']
                elif sl_pct > max_sl_pct:
                    result['sl_distance'] = entry_price * max_sl_pct
                    result['tp_distance'] = result['sl_distance'] * result['risk_reward']
        
        # إعادة حساب TP بناءً على SL المحدث
        if result['sl'] and result['sl_distance']:
            if result['signal'] in ['BUY', 'STRONG_BUY']:
                result['tp'] = entry_price + result['tp_distance']
            else:
                result['tp'] = entry_price - result['tp_distance']
        
        return result
    
    def _calculate_break_even_levels(self, sl_tp_result: Dict, settings: Dict, 
                                    instrument_type: str) -> Dict:
        """حساب مستويات Break Even"""
        be_levels = {
            'enabled': True,
            'trigger_price': None,
            'trigger_distance': None,
            'new_sl': None,
            'buffer': None
        }
        
        entry_price = sl_tp_result['entry_price']
        signal = sl_tp_result['signal']
        
        # الحصول على قيمة التفعيل
        if instrument_type in ['forex_major', 'forex_minor']:
            trigger_pips = settings.get('be_trigger_pips', 20)
            pip_value = self._get_pip_value(sl_tp_result['symbol'])
            trigger_distance = trigger_pips * pip_value
            buffer = 2 * pip_value  # 2 pips buffer
        elif instrument_type == 'crypto':
            trigger_pct = settings.get('be_trigger_percent', 2.0) / 100
            trigger_distance = entry_price * trigger_pct
            buffer = entry_price * 0.001  # 0.1% buffer
        else:
            trigger_points = settings.get('be_trigger_points', 40)
            trigger_distance = trigger_points * 0.01  # تحويل للسعر
            buffer = 1 * 0.01  # 1 point buffer
        
        # حساب مستويات التفعيل
        if signal in ['BUY', 'STRONG_BUY']:
            be_levels['trigger_price'] = entry_price + trigger_distance
            be_levels['new_sl'] = entry_price + buffer
        else:
            be_levels['trigger_price'] = entry_price - trigger_distance
            be_levels['new_sl'] = entry_price - buffer
        
        be_levels['trigger_distance'] = trigger_distance
        be_levels['buffer'] = buffer
        
        return be_levels
    
    def _calculate_trailing_stop_levels(self, sl_tp_result: Dict, settings: Dict, 
                                       instrument_type: str) -> Dict:
        """حساب مستويات Trailing Stop"""
        trail_levels = {
            'enabled': True,
            'start_price': None,
            'start_distance': None,
            'step_size': None,
            'min_distance': None
        }
        
        entry_price = sl_tp_result['entry_price']
        signal = sl_tp_result['signal']
        
        # الحصول على قيم التفعيل
        if instrument_type in ['forex_major', 'forex_minor']:
            start_pips = settings.get('trail_start_pips', 30)
            step_pips = settings.get('trail_step_pips', 10)
            pip_value = self._get_pip_value(sl_tp_result['symbol'])
            
            trail_levels['start_distance'] = start_pips * pip_value
            trail_levels['step_size'] = step_pips * pip_value
            trail_levels['min_distance'] = 10 * pip_value  # 10 pips minimum
            
        elif instrument_type == 'crypto':
            start_pct = settings.get('trail_start_percent', 3.0) / 100
            step_pct = settings.get('trail_step_percent', 1.0) / 100
            
            trail_levels['start_distance'] = entry_price * start_pct
            trail_levels['step_size'] = entry_price * step_pct
            trail_levels['min_distance'] = entry_price * 0.01  # 1% minimum
            
        else:
            start_points = settings.get('trail_start_points', 60)
            step_points = settings.get('trail_step_points', 20)
            
            trail_levels['start_distance'] = start_points * 0.01
            trail_levels['step_size'] = step_points * 0.01
            trail_levels['min_distance'] = 20 * 0.01  # 20 points minimum
        
        # حساب سعر البدء
        if signal in ['BUY', 'STRONG_BUY']:
            trail_levels['start_price'] = entry_price + trail_levels['start_distance']
        else:
            trail_levels['start_price'] = entry_price - trail_levels['start_distance']
        
        return trail_levels
    
    def update_trailing_stop(self, position: Dict, current_price: float) -> Dict:
        """
        تحديث Trailing Stop للصفقة المفتوحة
        
        Args:
            position: معلومات الصفقة الحالية
            current_price: السعر الحالي
            
        Returns:
            dict: معلومات SL المحدثة
        """
        result = {
            'should_update': False,
            'new_sl': position['current_sl'],
            'reason': None
        }
        
        try:
            trail_info = position.get('trailing_stop', {})
            if not trail_info.get('enabled', False):
                return result
            
            signal = position['signal']
            entry_price = position['entry_price']
            current_sl = position['current_sl']
            
            # التحقق من تفعيل Trailing Stop
            if signal in ['BUY', 'STRONG_BUY']:
                # للشراء
                if current_price >= trail_info['start_price']:
                    # حساب SL الجديد
                    distance_from_start = current_price - trail_info['start_price']
                    steps = int(distance_from_start / trail_info['step_size'])
                    
                    if steps > 0:
                        new_sl = entry_price + trail_info['min_distance'] + (steps * trail_info['step_size'])
                        
                        # التأكد من أن SL الجديد أفضل من الحالي
                        if new_sl > current_sl:
                            result['should_update'] = True
                            result['new_sl'] = new_sl
                            result['reason'] = f'Trailing stop: {steps} steps'
            else:
                # للبيع
                if current_price <= trail_info['start_price']:
                    # حساب SL الجديد
                    distance_from_start = trail_info['start_price'] - current_price
                    steps = int(distance_from_start / trail_info['step_size'])
                    
                    if steps > 0:
                        new_sl = entry_price - trail_info['min_distance'] - (steps * trail_info['step_size'])
                        
                        # التأكد من أن SL الجديد أفضل من الحالي
                        if new_sl < current_sl:
                            result['should_update'] = True
                            result['new_sl'] = new_sl
                            result['reason'] = f'Trailing stop: {steps} steps'
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
        
        return result
    
    def check_break_even(self, position: Dict, current_price: float) -> Dict:
        """
        التحقق من تفعيل Break Even
        
        Args:
            position: معلومات الصفقة
            current_price: السعر الحالي
            
        Returns:
            dict: معلومات Break Even
        """
        result = {
            'should_activate': False,
            'new_sl': position['current_sl'],
            'reason': None
        }
        
        try:
            be_info = position.get('break_even', {})
            if not be_info.get('enabled', False):
                return result
            
            # التحقق من عدم تفعيله مسبقاً
            if position.get('break_even_activated', False):
                return result
            
            signal = position['signal']
            
            # التحقق من الوصول لسعر التفعيل
            if signal in ['BUY', 'STRONG_BUY']:
                if current_price >= be_info['trigger_price']:
                    result['should_activate'] = True
                    result['new_sl'] = be_info['new_sl']
                    result['reason'] = 'Break even triggered'
            else:
                if current_price <= be_info['trigger_price']:
                    result['should_activate'] = True
                    result['new_sl'] = be_info['new_sl']
                    result['reason'] = 'Break even triggered'
            
        except Exception as e:
            logger.error(f"Error checking break even: {str(e)}")
        
        return result
    
    def _get_pip_value(self, symbol: str) -> float:
        """الحصول على قيمة النقطة"""
        if 'JPY' in symbol.upper():
            return 0.01
        else:
            return 0.0001
    
    def _add_pip_calculations(self, result: Dict, symbol: str) -> Dict:
        """إضافة حسابات النقاط والنسب المئوية"""
        if result['sl'] and result['tp']:
            entry_price = result['entry_price']
            
            # حساب النقاط
            pip_value = self._get_pip_value(symbol)
            result['sl_pips'] = abs(result['sl_distance']) / pip_value
            result['tp_pips'] = abs(result['tp_distance']) / pip_value
            
            # حساب النسب المئوية
            result['sl_percent'] = (abs(result['sl_distance']) / entry_price) * 100
            result['tp_percent'] = (abs(result['tp_distance']) / entry_price) * 100
            
            # نسبة المخاطرة الفعلية
            if result['sl_distance'] > 0:
                result['actual_risk_reward'] = result['tp_distance'] / result['sl_distance']
        
        return result
    
    def _default_sl_tp(self, signal: str, entry_price: float, symbol: str) -> Dict:
        """قيم SL/TP الافتراضية في حالة الخطأ"""
        pip_value = self._get_pip_value(symbol)
        default_sl_pips = 30
        default_tp_pips = 60
        
        result = {
            'sl': None,
            'tp': None,
            'sl_distance': default_sl_pips * pip_value,
            'tp_distance': default_tp_pips * pip_value,
            'method': 'default',
            'risk_reward': 2.0
        }
        
        if signal in ['BUY', 'STRONG_BUY']:
            result['sl'] = entry_price - result['sl_distance']
            result['tp'] = entry_price + result['tp_distance']
        else:
            result['sl'] = entry_price + result['sl_distance']
            result['tp'] = entry_price - result['tp_distance']
        
        return result
    
    def save_sl_tp_config(self, config: Dict, filename: str = "sl_tp_config.json"):
        """حفظ إعدادات SL/TP"""
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Saved SL/TP config to {filename}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def load_sl_tp_config(self, filename: str = "sl_tp_config.json") -> Dict:
        """تحميل إعدادات SL/TP"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}


# دالة مساعدة للاستخدام المباشر
def calculate_dynamic_sl_tp(signal: str, entry_price: float, df: pd.DataFrame, 
                           symbol: str, method: str = 'hybrid') -> Dict:
    """
    دالة سريعة لحساب SL/TP الديناميكي
    
    Args:
        signal: إشارة التداول
        entry_price: سعر الدخول
        df: بيانات السوق
        symbol: رمز الأداة
        method: طريقة الحساب
        
    Returns:
        dict: معلومات SL/TP
    """
    system = DynamicSLTPSystem()
    return system.calculate_dynamic_sl_tp(signal, entry_price, df, symbol, method)


if __name__ == "__main__":
    # مثال للاختبار
    print("🎯 Dynamic SL/TP System Test")
    
    # بيانات تجريبية
    test_data = {
        'open': np.random.rand(200) * 0.01 + 1.1000,
        'high': np.random.rand(200) * 0.01 + 1.1100,
        'low': np.random.rand(200) * 0.01 + 1.0900,
        'close': np.random.rand(200) * 0.01 + 1.1000,
        'volume': np.random.randint(1000, 10000, 200),
        'ATR': np.random.rand(200) * 0.001 + 0.0010
    }
    
    df = pd.DataFrame(test_data)
    
    # إنشاء النظام
    system = DynamicSLTPSystem()
    
    # اختبار أنواع مختلفة
    test_cases = [
        ('EURUSD', 'BUY', 1.1000),
        ('XAUUSD', 'SELL', 2000.00),
        ('US30', 'BUY', 35000),
        ('BTCUSD', 'BUY', 50000)
    ]
    
    for symbol, signal, entry_price in test_cases:
        print(f"\n📊 {symbol} - {signal} @ {entry_price}")
        
        # حساب SL/TP
        result = system.calculate_dynamic_sl_tp(signal, entry_price, df, symbol, 'hybrid')
        
        print(f"   SL: {result['sl']:.5f} ({result['sl_pips']:.1f} pips)")
        print(f"   TP: {result['tp']:.5f} ({result['tp_pips']:.1f} pips)")
        print(f"   Method: {result['method']}")
        print(f"   R:R = 1:{result['actual_risk_reward']:.2f}")
        print(f"   Break Even: {result['break_even']['trigger_price']:.5f}")
        print(f"   Trail Start: {result['trailing_stop']['start_price']:.5f}")