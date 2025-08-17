#!/usr/bin/env python3
"""
Instrument Manager - إدارة أزواج العملات والأدوات المالية
"""

class InstrumentManager:
    """مدير الأدوات المالية"""
    
    def __init__(self):
        self.instruments = self._load_default_instruments()
    
    def _load_default_instruments(self):
        """تحميل الأدوات الافتراضية"""
        return [
            # Forex Majors - بنفس الأسماء في قاعدة البيانات
            {'symbol': 'EURUSDm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'GBPUSDm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'USDJPYm', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            {'symbol': 'USDCHFm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'AUDUSDm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'USDCADm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'NZDUSDm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            # Forex Minors
            {'symbol': 'EURGBPm', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'EURJPYm', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            {'symbol': 'GBPJPYm', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            # Commodities  
            {'symbol': 'XAUUSDm', 'type': 'commodity', 'active': True, 'pip_value': 0.01},
            {'symbol': 'XAGUSDm', 'type': 'commodity', 'active': True, 'pip_value': 0.001},
            {'symbol': 'WTIm', 'type': 'commodity', 'active': True, 'pip_value': 0.01},
            # Crypto
            {'symbol': 'BTCUSDm', 'type': 'crypto', 'active': True, 'pip_value': 1.0},
            {'symbol': 'ETHUSDm', 'type': 'crypto', 'active': True, 'pip_value': 0.01},
        ]
    
    def get_all_instruments(self):
        """الحصول على جميع الأدوات"""
        return self.instruments
    
    def get_active_instruments(self):
        """الحصول على الأدوات النشطة فقط"""
        return [inst for inst in self.instruments if inst.get('active', True)]
    
    def get_instrument(self, symbol):
        """الحصول على أداة محددة"""
        for inst in self.instruments:
            if inst['symbol'] == symbol:
                return inst
        return None
    
    def get_pip_value(self, symbol):
        """الحصول على قيمة النقطة"""
        inst = self.get_instrument(symbol)
        return inst['pip_value'] if inst else 0.0001