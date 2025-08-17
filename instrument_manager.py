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
            # Forex Majors
            {'symbol': 'EUR/USD', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'GBP/USD', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'USD/JPY', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            {'symbol': 'USD/CHF', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'AUD/USD', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'USD/CAD', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'NZD/USD', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            # Forex Minors
            {'symbol': 'EUR/GBP', 'type': 'forex', 'active': True, 'pip_value': 0.0001},
            {'symbol': 'EUR/JPY', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            {'symbol': 'GBP/JPY', 'type': 'forex', 'active': True, 'pip_value': 0.01},
            # Commodities
            {'symbol': 'XAU/USD', 'type': 'commodity', 'active': True, 'pip_value': 0.01},
            {'symbol': 'XAG/USD', 'type': 'commodity', 'active': True, 'pip_value': 0.001},
            {'symbol': 'WTI/USD', 'type': 'commodity', 'active': True, 'pip_value': 0.01},
            # Crypto
            {'symbol': 'BTC/USD', 'type': 'crypto', 'active': True, 'pip_value': 1.0},
            {'symbol': 'ETH/USD', 'type': 'crypto', 'active': True, 'pip_value': 0.01},
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