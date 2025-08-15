#!/usr/bin/env python3
"""
نظام إدارة الأدوات المالية المتعددة
Multi-Instrument Management System

يدعم:
1. أزواج الفوركس (Majors, Minors, Exotics)
2. المعادن (Gold, Silver, Platinum, Palladium)
3. الطاقة (Oil, Natural Gas)
4. المؤشرات (US30, NAS100, SP500, DAX, etc.)
5. العملات الرقمية (Bitcoin, Ethereum, etc.)
6. الأسهم (Apple, Google, Tesla, etc.)
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
import os

class InstrumentManager:
    """مدير الأدوات المالية المتعددة"""
    
    def __init__(self):
        """تهيئة مدير الأدوات"""
        self.instruments = self._initialize_instruments()
        self.active_instruments = []
        self.config_file = "instruments_config.json"
        self.load_config()
        
    def _initialize_instruments(self) -> Dict:
        """تهيئة قاعدة بيانات الأدوات المالية"""
        return {
            # Forex Major Pairs
            'EURUSD': {
                'type': 'forex_major',
                'name': 'Euro vs US Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.2,
                'daily_range_pips': 80,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'GBPUSD': {
                'type': 'forex_major',
                'name': 'British Pound vs US Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.5,
                'daily_range_pips': 100,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'USDJPY': {
                'type': 'forex_major',
                'name': 'US Dollar vs Japanese Yen',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.0,
                'daily_range_pips': 70,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'USDCHF': {
                'type': 'forex_major',
                'name': 'US Dollar vs Swiss Franc',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.8,
                'daily_range_pips': 60,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'AUDUSD': {
                'type': 'forex_major',
                'name': 'Australian Dollar vs US Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.3,
                'daily_range_pips': 70,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'USDCAD': {
                'type': 'forex_major',
                'name': 'US Dollar vs Canadian Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.5,
                'daily_range_pips': 75,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'NZDUSD': {
                'type': 'forex_major',
                'name': 'New Zealand Dollar vs US Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 2.0,
                'daily_range_pips': 65,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            
            # Forex Minor Pairs
            'EURJPY': {
                'type': 'forex_minor',
                'name': 'Euro vs Japanese Yen',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 2.0,
                'daily_range_pips': 100,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'GBPJPY': {
                'type': 'forex_minor',
                'name': 'British Pound vs Japanese Yen',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 2.5,
                'daily_range_pips': 150,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'EURGBP': {
                'type': 'forex_minor',
                'name': 'Euro vs British Pound',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 1.8,
                'daily_range_pips': 60,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': True
            },
            'AUDCAD': {
                'type': 'forex_minor',
                'name': 'Australian Dollar vs Canadian Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 2.2,
                'daily_range_pips': 70,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': False
            },
            'NZDCAD': {
                'type': 'forex_minor',
                'name': 'New Zealand Dollar vs Canadian Dollar',
                'pip_value': 0.0001,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'typical_spread': 2.5,
                'daily_range_pips': 65,
                'session': 'all',
                'suffix_variants': ['', 'm', '.fx', 'fx'],
                'enabled': False
            },
            
            # Metals
            'XAUUSD': {
                'type': 'metals',
                'name': 'Gold vs US Dollar',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 50.0,
                'lot_step': 0.01,
                'typical_spread': 30,
                'daily_range_pips': 2000,
                'session': 'all',
                'suffix_variants': ['', 'm', '.spot', 'spot'],
                'enabled': True
            },
            'XAGUSD': {
                'type': 'metals',
                'name': 'Silver vs US Dollar',
                'pip_value': 0.001,
                'min_lot': 0.01,
                'max_lot': 50.0,
                'lot_step': 0.01,
                'typical_spread': 3,
                'daily_range_pips': 50,
                'session': 'all',
                'suffix_variants': ['', 'm', '.spot', 'spot'],
                'enabled': True
            },
            'XPTUSD': {
                'type': 'metals',
                'name': 'Platinum vs US Dollar',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 20.0,
                'lot_step': 0.01,
                'typical_spread': 50,
                'daily_range_pips': 3000,
                'session': 'all',
                'suffix_variants': ['', 'm', '.spot', 'spot'],
                'enabled': False
            },
            'XPDUSD': {
                'type': 'metals',
                'name': 'Palladium vs US Dollar',
                'pip_value': 0.01,
                'min_lot': 0.01,
                'max_lot': 10.0,
                'lot_step': 0.01,
                'typical_spread': 100,
                'daily_range_pips': 5000,
                'session': 'all',
                'suffix_variants': ['', 'm', '.spot', 'spot'],
                'enabled': False
            },
            
            # Energy
            'USOIL': {
                'type': 'energy',
                'name': 'WTI Crude Oil',
                'pip_value': 0.01,
                'min_lot': 0.1,
                'max_lot': 100.0,
                'lot_step': 0.1,
                'typical_spread': 3,
                'daily_range_pips': 200,
                'session': 'us',
                'suffix_variants': ['', '.cash', 'cash', 'WTI'],
                'enabled': True
            },
            'UKOIL': {
                'type': 'energy',
                'name': 'Brent Crude Oil',
                'pip_value': 0.01,
                'min_lot': 0.1,
                'max_lot': 100.0,
                'lot_step': 0.1,
                'typical_spread': 3,
                'daily_range_pips': 200,
                'session': 'uk',
                'suffix_variants': ['', '.cash', 'cash', 'BRENT'],
                'enabled': True
            },
            'NGAS': {
                'type': 'energy',
                'name': 'Natural Gas',
                'pip_value': 0.001,
                'min_lot': 1.0,
                'max_lot': 100.0,
                'lot_step': 1.0,
                'typical_spread': 3,
                'daily_range_pips': 100,
                'session': 'us',
                'suffix_variants': ['', '.cash', 'cash', 'NATGAS'],
                'enabled': False
            },
            
            # Indices
            'US30': {
                'type': 'indices',
                'name': 'Dow Jones 30',
                'pip_value': 1.0,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 3,
                'daily_range_pips': 300,
                'session': 'us',
                'suffix_variants': ['', '.cash', 'cash', 'DJ30'],
                'enabled': True
            },
            'NAS100': {
                'type': 'indices',
                'name': 'NASDAQ 100',
                'pip_value': 0.1,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 2,
                'daily_range_pips': 200,
                'session': 'us',
                'suffix_variants': ['', '.cash', 'cash', 'USTEC'],
                'enabled': True
            },
            'SP500': {
                'type': 'indices',
                'name': 'S&P 500',
                'pip_value': 0.1,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 0.5,
                'daily_range_pips': 50,
                'session': 'us',
                'suffix_variants': ['', '.cash', 'cash', 'US500'],
                'enabled': True
            },
            'DAX': {
                'type': 'indices',
                'name': 'Germany 30',
                'pip_value': 0.1,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 2,
                'daily_range_pips': 150,
                'session': 'eu',
                'suffix_variants': ['', '.cash', 'cash', 'GER30'],
                'enabled': True
            },
            'FTSE100': {
                'type': 'indices',
                'name': 'UK 100',
                'pip_value': 0.1,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 2,
                'daily_range_pips': 100,
                'session': 'uk',
                'suffix_variants': ['', '.cash', 'cash', 'UK100'],
                'enabled': False
            },
            'NIKKEI': {
                'type': 'indices',
                'name': 'Japan 225',
                'pip_value': 1.0,
                'min_lot': 0.1,
                'max_lot': 50.0,
                'lot_step': 0.1,
                'typical_spread': 10,
                'daily_range_pips': 300,
                'session': 'asia',
                'suffix_variants': ['', '.cash', 'cash', 'JPN225'],
                'enabled': False
            },
            
            # Crypto
            'BTCUSD': {
                'type': 'crypto',
                'name': 'Bitcoin vs US Dollar',
                'pip_value': 1.0,
                'min_lot': 0.01,
                'max_lot': 5.0,
                'lot_step': 0.01,
                'typical_spread': 50,
                'daily_range_pips': 2000,
                'session': 'all',
                'suffix_variants': ['', '.crypto', 'crypto'],
                'enabled': True
            },
            'ETHUSD': {
                'type': 'crypto',
                'name': 'Ethereum vs US Dollar',
                'pip_value': 0.1,
                'min_lot': 0.01,
                'max_lot': 10.0,
                'lot_step': 0.01,
                'typical_spread': 5,
                'daily_range_pips': 150,
                'session': 'all',
                'suffix_variants': ['', '.crypto', 'crypto'],
                'enabled': True
            },
            'XRPUSD': {
                'type': 'crypto',
                'name': 'Ripple vs US Dollar',
                'pip_value': 0.0001,
                'min_lot': 10.0,
                'max_lot': 10000.0,
                'lot_step': 10.0,
                'typical_spread': 0.002,
                'daily_range_pips': 10,
                'session': 'all',
                'suffix_variants': ['', '.crypto', 'crypto'],
                'enabled': False
            },
            'LTCUSD': {
                'type': 'crypto',
                'name': 'Litecoin vs US Dollar',
                'pip_value': 0.01,
                'min_lot': 0.1,
                'max_lot': 100.0,
                'lot_step': 0.1,
                'typical_spread': 1,
                'daily_range_pips': 20,
                'session': 'all',
                'suffix_variants': ['', '.crypto', 'crypto'],
                'enabled': False
            },
            
            # Stocks
            'AAPL': {
                'type': 'stocks',
                'name': 'Apple Inc',
                'pip_value': 0.01,
                'min_lot': 1.0,
                'max_lot': 1000.0,
                'lot_step': 1.0,
                'typical_spread': 0.05,
                'daily_range_pips': 5,
                'session': 'us',
                'suffix_variants': ['', '.us', 'us'],
                'enabled': True
            },
            'GOOGL': {
                'type': 'stocks',
                'name': 'Alphabet Inc',
                'pip_value': 0.01,
                'min_lot': 1.0,
                'max_lot': 1000.0,
                'lot_step': 1.0,
                'typical_spread': 0.10,
                'daily_range_pips': 30,
                'session': 'us',
                'suffix_variants': ['', '.us', 'us'],
                'enabled': True
            },
            'MSFT': {
                'type': 'stocks',
                'name': 'Microsoft Corp',
                'pip_value': 0.01,
                'min_lot': 1.0,
                'max_lot': 1000.0,
                'lot_step': 1.0,
                'typical_spread': 0.05,
                'daily_range_pips': 10,
                'session': 'us',
                'suffix_variants': ['', '.us', 'us'],
                'enabled': True
            },
            'TSLA': {
                'type': 'stocks',
                'name': 'Tesla Inc',
                'pip_value': 0.01,
                'min_lot': 1.0,
                'max_lot': 1000.0,
                'lot_step': 1.0,
                'typical_spread': 0.20,
                'daily_range_pips': 20,
                'session': 'us',
                'suffix_variants': ['', '.us', 'us'],
                'enabled': True
            },
            'AMZN': {
                'type': 'stocks',
                'name': 'Amazon.com Inc',
                'pip_value': 0.01,
                'min_lot': 1.0,
                'max_lot': 1000.0,
                'lot_step': 1.0,
                'typical_spread': 0.15,
                'daily_range_pips': 50,
                'session': 'us',
                'suffix_variants': ['', '.us', 'us'],
                'enabled': False
            }
        }
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """
        الحصول على معلومات الأداة
        
        Args:
            symbol: رمز الأداة (مع أو بدون suffix)
            
        Returns:
            dict: معلومات الأداة أو None
        """
        # البحث المباشر
        if symbol in self.instruments:
            return self.instruments[symbol].copy()
        
        # البحث مع التعامل مع suffixes
        base_symbol = self._extract_base_symbol(symbol)
        if base_symbol in self.instruments:
            info = self.instruments[base_symbol].copy()
            info['actual_symbol'] = symbol
            return info
        
        return None
    
    def _extract_base_symbol(self, symbol: str) -> str:
        """استخراج الرمز الأساسي بدون suffix"""
        # إزالة suffixes الشائعة
        suffixes = ['m', '.fx', 'fx', '.cash', 'cash', '.spot', 'spot', 
                   '.crypto', 'crypto', '.us', 'us']
        
        symbol_upper = symbol.upper()
        for suffix in suffixes:
            if symbol_upper.endswith(suffix.upper()):
                return symbol_upper[:-len(suffix)]
        
        return symbol_upper
    
    def get_enabled_instruments(self, types: Optional[List[str]] = None) -> List[str]:
        """
        الحصول على قائمة الأدوات المفعلة
        
        Args:
            types: أنواع الأدوات المطلوبة (None = الكل)
            
        Returns:
            list: قائمة الرموز المفعلة
        """
        enabled = []
        
        for symbol, info in self.instruments.items():
            if info['enabled']:
                if types is None or info['type'] in types:
                    enabled.append(symbol)
        
        return enabled
    
    def get_instruments_by_type(self, instrument_type: str) -> List[str]:
        """الحصول على الأدوات حسب النوع"""
        instruments = []
        
        for symbol, info in self.instruments.items():
            if info['type'] == instrument_type and info['enabled']:
                instruments.append(symbol)
        
        return instruments
    
    def enable_instrument(self, symbol: str) -> bool:
        """تفعيل أداة"""
        base_symbol = self._extract_base_symbol(symbol)
        if base_symbol in self.instruments:
            self.instruments[base_symbol]['enabled'] = True
            self.save_config()
            logger.info(f"✅ Enabled {base_symbol}")
            return True
        return False
    
    def disable_instrument(self, symbol: str) -> bool:
        """تعطيل أداة"""
        base_symbol = self._extract_base_symbol(symbol)
        if base_symbol in self.instruments:
            self.instruments[base_symbol]['enabled'] = False
            self.save_config()
            logger.info(f"❌ Disabled {base_symbol}")
            return True
        return False
    
    def add_custom_instrument(self, symbol: str, info: Dict) -> bool:
        """إضافة أداة مخصصة"""
        try:
            # التحقق من المعلومات المطلوبة
            required_fields = ['type', 'name', 'pip_value', 'min_lot', 
                             'max_lot', 'lot_step', 'typical_spread']
            
            for field in required_fields:
                if field not in info:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # إضافة الحقول الافتراضية
            info.setdefault('daily_range_pips', 100)
            info.setdefault('session', 'all')
            info.setdefault('suffix_variants', [''])
            info.setdefault('enabled', True)
            
            self.instruments[symbol.upper()] = info
            self.save_config()
            logger.info(f"✅ Added custom instrument: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom instrument: {str(e)}")
            return False
    
    def get_pip_value(self, symbol: str, price: Optional[float] = None) -> float:
        """
        حساب قيمة النقطة
        
        Args:
            symbol: رمز الأداة
            price: السعر الحالي (مطلوب لبعض الحسابات)
            
        Returns:
            float: قيمة النقطة
        """
        info = self.get_instrument_info(symbol)
        if info:
            return info['pip_value']
        
        # قيمة افتراضية
        return 0.0001
    
    def calculate_position_size(self, symbol: str, account_balance: float, 
                               risk_percent: float, sl_pips: float) -> float:
        """
        حساب حجم المركز
        
        Args:
            symbol: رمز الأداة
            account_balance: رصيد الحساب
            risk_percent: نسبة المخاطرة (%)
            sl_pips: عدد نقاط وقف الخسارة
            
        Returns:
            float: حجم المركز (lots)
        """
        info = self.get_instrument_info(symbol)
        if not info:
            return 0.01
        
        # حساب المبلغ المخاطر به
        risk_amount = account_balance * (risk_percent / 100)
        
        # حساب قيمة النقطة للوت الواحد
        # هذا تبسيط، قد تحتاج لحساب أكثر دقة حسب نوع الحساب
        pip_value_per_lot = 10.0  # افتراضي للحساب القياسي
        
        # حساب حجم المركز
        if sl_pips > 0:
            position_size = risk_amount / (sl_pips * pip_value_per_lot)
        else:
            position_size = info['min_lot']
        
        # التقريب حسب lot_step
        lot_step = info['lot_step']
        position_size = round(position_size / lot_step) * lot_step
        
        # التحقق من الحدود
        position_size = max(info['min_lot'], min(info['max_lot'], position_size))
        
        return position_size
    
    def get_trading_session(self, symbol: str) -> str:
        """الحصول على جلسة التداول للأداة"""
        info = self.get_instrument_info(symbol)
        if info:
            return info['session']
        return 'all'
    
    def is_tradable_now(self, symbol: str, current_hour_utc: int) -> bool:
        """
        التحقق من إمكانية التداول الآن
        
        Args:
            symbol: رمز الأداة
            current_hour_utc: الساعة الحالية UTC
            
        Returns:
            bool: هل يمكن التداول الآن
        """
        session = self.get_trading_session(symbol)
        
        # جلسات التداول (UTC)
        sessions = {
            'all': (0, 24),  # 24/7
            'forex': (21, 21),  # Sunday 21:00 - Friday 21:00
            'us': (13, 20),  # 13:30 - 20:00 UTC
            'eu': (7, 16),   # 07:00 - 16:00 UTC
            'uk': (8, 16),   # 08:00 - 16:30 UTC
            'asia': (23, 6)  # 23:00 - 06:00 UTC (crosses midnight)
        }
        
        if session not in sessions:
            session = 'all'
        
        start, end = sessions[session]
        
        # معالجة الجلسات التي تعبر منتصف الليل
        if start > end:
            return current_hour_utc >= start or current_hour_utc < end
        else:
            return start <= current_hour_utc < end
    
    def get_statistics(self) -> Dict:
        """الحصول على إحصائيات الأدوات"""
        stats = {
            'total': len(self.instruments),
            'enabled': 0,
            'by_type': {}
        }
        
        for symbol, info in self.instruments.items():
            if info['enabled']:
                stats['enabled'] += 1
            
            instrument_type = info['type']
            if instrument_type not in stats['by_type']:
                stats['by_type'][instrument_type] = {'total': 0, 'enabled': 0}
            
            stats['by_type'][instrument_type]['total'] += 1
            if info['enabled']:
                stats['by_type'][instrument_type]['enabled'] += 1
        
        return stats
    
    def save_config(self):
        """حفظ إعدادات الأدوات"""
        try:
            config = {
                'instruments': self.instruments,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Saved instruments config")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def load_config(self):
        """تحميل إعدادات الأدوات"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                if 'instruments' in config:
                    # دمج مع الإعدادات الافتراضية
                    for symbol, info in config['instruments'].items():
                        if symbol in self.instruments:
                            self.instruments[symbol].update(info)
                        else:
                            self.instruments[symbol] = info
                
                logger.info(f"✅ Loaded instruments config")
                
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
    
    def export_for_metatrader(self) -> str:
        """تصدير قائمة الأدوات لـ MetaTrader"""
        enabled = self.get_enabled_instruments()
        
        # تجميع حسب النوع
        mt_list = []
        
        # Forex
        forex = [s for s in enabled if self.instruments[s]['type'].startswith('forex')]
        if forex:
            mt_list.append("// Forex Pairs")
            mt_list.extend(forex)
            mt_list.append("")
        
        # Metals
        metals = [s for s in enabled if self.instruments[s]['type'] == 'metals']
        if metals:
            mt_list.append("// Metals")
            mt_list.extend(metals)
            mt_list.append("")
        
        # Energy
        energy = [s for s in enabled if self.instruments[s]['type'] == 'energy']
        if energy:
            mt_list.append("// Energy")
            mt_list.extend(energy)
            mt_list.append("")
        
        # Indices
        indices = [s for s in enabled if self.instruments[s]['type'] == 'indices']
        if indices:
            mt_list.append("// Indices")
            mt_list.extend(indices)
            mt_list.append("")
        
        # Crypto
        crypto = [s for s in enabled if self.instruments[s]['type'] == 'crypto']
        if crypto:
            mt_list.append("// Cryptocurrencies")
            mt_list.extend(crypto)
            mt_list.append("")
        
        # Stocks
        stocks = [s for s in enabled if self.instruments[s]['type'] == 'stocks']
        if stocks:
            mt_list.append("// Stocks")
            mt_list.extend(stocks)
        
        return "\n".join(mt_list)


# دالة مساعدة للاستخدام المباشر
def get_all_trading_instruments() -> List[str]:
    """الحصول على جميع الأدوات المفعلة للتداول"""
    manager = InstrumentManager()
    return manager.get_enabled_instruments()


if __name__ == "__main__":
    # اختبار النظام
    print("🎯 Instrument Manager Test")
    
    manager = InstrumentManager()
    
    # عرض الإحصائيات
    stats = manager.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total instruments: {stats['total']}")
    print(f"   Enabled: {stats['enabled']}")
    print(f"\n   By Type:")
    for type_name, type_stats in stats['by_type'].items():
        print(f"   - {type_name}: {type_stats['enabled']}/{type_stats['total']}")
    
    # عرض الأدوات المفعلة
    print(f"\n✅ Enabled Instruments:")
    enabled = manager.get_enabled_instruments()
    for i, symbol in enumerate(enabled, 1):
        info = manager.get_instrument_info(symbol)
        print(f"   {i}. {symbol} ({info['name']}) - {info['type']}")
    
    # تصدير لـ MetaTrader
    print(f"\n📝 MetaTrader Export:")
    print(manager.export_for_metatrader())
    
    # اختبار حساب حجم المركز
    print(f"\n💰 Position Size Test:")
    test_cases = [
        ('EURUSD', 10000, 1.0, 20),  # $10k, 1% risk, 20 pips SL
        ('XAUUSD', 10000, 1.0, 100), # $10k, 1% risk, 100 pips SL
        ('BTCUSD', 10000, 2.0, 500), # $10k, 2% risk, 500 pips SL
    ]
    
    for symbol, balance, risk, sl_pips in test_cases:
        lot_size = manager.calculate_position_size(symbol, balance, risk, sl_pips)
        print(f"   {symbol}: {lot_size} lots (${balance}, {risk}% risk, {sl_pips} pips SL)")