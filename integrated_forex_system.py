import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import joblib
import warnings
import MetaTrader5 as mt5
from ultimate_forex_system_v2 import AdvancedForexSystem
from advanced_hypothesis_system import HypothesisEngine
import json

warnings.filterwarnings('ignore')

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('integrated_forex_system.log'),
        logging.StreamHandler()
    ]
)

class IntegratedForexSystem:
    """النظام المتكامل الذي يجمع بين ML والفرضيات"""
    
    def __init__(self):
        # تهيئة النظامين
        self.ml_system = AdvancedForexSystem()
        self.hypothesis_engine = HypothesisEngine()
        
        # إعدادات التداول
        self.min_confidence = 0.65
        self.risk_per_trade = 0.02
        self.max_positions = 5
        
        # تتبع الأداء
        self.performance_history = []
        self.active_positions = {}
        
        logging.info("✅ Integrated Forex System initialized successfully")
        
    def analyze_pair(self, symbol, timeframe):
        """تحليل زوج عملات باستخدام النظام المتكامل"""
        try:
            # الحصول على البيانات
            df = self.ml_system.get_data(symbol, timeframe, 1000)
            
            if df is None or len(df) < 200:
                logging.warning(f"Insufficient data for {symbol}")
                return None
                
            # حساب الميزات
            features = self.ml_system.calculate_all_features(df)
            
            # الحصول على آخر صف من الميزات للتنبؤ
            latest_features = features.iloc[-1:].copy()
            
            # تنبؤات ML
            tf_name = self.ml_system.timeframe_names[timeframe]
            ml_prediction, ml_probabilities = self.ml_system.predict(symbol, tf_name, latest_features)
            
            if ml_prediction is None:
                logging.warning(f"No ML model available for {symbol} {tf_name}")
                ml_prediction = [2]  # Default to HOLD
                ml_probabilities = [[0.33, 0.33, 0.34]]
                
            # تقييم الفرضيات
            hypothesis_results = self.hypothesis_engine.evaluate_all(df, features)
            
            # دمج النتائج
            integrated_results = self.hypothesis_engine.integrate_with_ml_predictions(
                hypothesis_results, 
                ml_prediction[0], 
                ml_probabilities[0]
            )
            
            # إضافة معلومات إضافية
            analysis_result = {
                'symbol': symbol,
                'timeframe': tf_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': df['close'].iloc[-1],
                'ml_prediction': ['BUY', 'SELL', 'HOLD'][ml_prediction[0]],
                'ml_probabilities': {
                    'buy': ml_probabilities[0][0],
                    'sell': ml_probabilities[0][1],
                    'hold': ml_probabilities[0][2]
                },
                'hypothesis_summary': self.hypothesis_engine.get_hypothesis_summary(hypothesis_results),
                **integrated_results
            }
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None
            
    def scan_all_pairs(self):
        """مسح جميع الأزواج والأطر الزمنية"""
        opportunities = []
        
        if not self.ml_system.connect_mt5():
            return opportunities
            
        for symbol in self.ml_system.pairs:
            for timeframe in self.ml_system.timeframes:
                logging.info(f"Analyzing {symbol} {self.ml_system.timeframe_names[timeframe]}...")
                
                result = self.analyze_pair(symbol, timeframe)
                
                if result and result['total_confidence'] >= self.min_confidence:
                    if result['final_decision'] != 'HOLD':
                        opportunities.append(result)
                        
        # ترتيب الفرص حسب الثقة
        opportunities.sort(key=lambda x: x['total_confidence'], reverse=True)
        
        mt5.shutdown()
        
        return opportunities
        
    def generate_trading_signals(self, opportunities):
        """توليد إشارات التداول من الفرص المتاحة"""
        signals = []
        
        for opp in opportunities[:self.max_positions]:
            signal = {
                'symbol': opp['symbol'],
                'timeframe': opp['timeframe'],
                'action': opp['final_decision'],
                'confidence': opp['total_confidence'],
                'entry_price': opp['current_price'],
                'stop_loss': self.calculate_stop_loss(opp),
                'take_profit': self.calculate_take_profit(opp),
                'lot_size': self.calculate_lot_size(opp),
                'reasoning': {
                    'ml_contribution': opp['ml_contribution'],
                    'hypothesis_contribution': opp['hypothesis_contribution'],
                    'top_hypotheses': opp['hypothesis_summary'][:3]
                }
            }
            
            signals.append(signal)
            
        return signals
        
    def calculate_stop_loss(self, opportunity):
        """حساب وقف الخسارة"""
        # استخدام ATR لحساب وقف الخسارة الديناميكي
        base_sl = 50  # نقاط أساسية
        
        if opportunity['final_decision'] == 'BUY':
            return opportunity['current_price'] - (base_sl * 0.0001)
        else:
            return opportunity['current_price'] + (base_sl * 0.0001)
            
    def calculate_take_profit(self, opportunity):
        """حساب جني الأرباح"""
        # نسبة المخاطرة إلى المكافأة 1:2
        base_tp = 100  # نقاط أساسية
        
        if opportunity['final_decision'] == 'BUY':
            return opportunity['current_price'] + (base_tp * 0.0001)
        else:
            return opportunity['current_price'] - (base_tp * 0.0001)
            
    def calculate_lot_size(self, opportunity):
        """حساب حجم الصفقة بناءً على إدارة المخاطر"""
        # حجم ثابت للتبسيط
        base_lot = 0.01
        
        # تعديل بناءً على الثقة
        confidence_multiplier = opportunity['total_confidence']
        
        return round(base_lot * confidence_multiplier, 2)
        
    def backtest_strategy(self, start_date, end_date):
        """اختبار الاستراتيجية على البيانات التاريخية"""
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }
        
        # هنا يمكن إضافة منطق الاختبار الخلفي الكامل
        
        return results
        
    def save_analysis_report(self, opportunities, signals, filename='analysis_report.json'):
        """حفظ تقرير التحليل"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'opportunities_found': len(opportunities),
            'signals_generated': len(signals),
            'top_opportunities': opportunities[:10],
            'trading_signals': signals,
            'system_status': {
                'ml_models_loaded': len(self.ml_system.models),
                'hypotheses_active': len(self.hypothesis_engine.hypotheses),
                'min_confidence': self.min_confidence
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Analysis report saved to {filename}")
        
    def real_time_monitoring(self):
        """المراقبة في الوقت الفعلي"""
        logging.info("Starting real-time monitoring...")
        
        while True:
            try:
                # مسح الفرص
                opportunities = self.scan_all_pairs()
                
                if opportunities:
                    # توليد الإشارات
                    signals = self.generate_trading_signals(opportunities)
                    
                    # عرض الإشارات
                    for signal in signals:
                        logging.info(f"""
                        🎯 Trading Signal:
                        Symbol: {signal['symbol']}
                        Action: {signal['action']}
                        Confidence: {signal['confidence']:.1%}
                        Entry: {signal['entry_price']:.5f}
                        SL: {signal['stop_loss']:.5f}
                        TP: {signal['take_profit']:.5f}
                        """)
                        
                    # حفظ التقرير
                    self.save_analysis_report(opportunities, signals)
                    
                # الانتظار قبل المسح التالي
                import time
                time.sleep(300)  # 5 دقائق
                
            except KeyboardInterrupt:
                logging.info("Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in monitoring: {str(e)}")
                
def main():
    # إنشاء النظام المتكامل
    system = IntegratedForexSystem()
    
    # تحليل زوج واحد كمثال
    logging.info("\n" + "="*50)
    logging.info("Single Pair Analysis Example")
    logging.info("="*50)
    
    if system.ml_system.connect_mt5():
        result = system.analyze_pair('EURUSDm', mt5.TIMEFRAME_H1)
        
        if result:
            logging.info(f"""
            Analysis Result for {result['symbol']} {result['timeframe']}:
            - Final Decision: {result['final_decision']}
            - Total Confidence: {result['total_confidence']:.1%}
            - ML Prediction: {result['ml_prediction']}
            - Combined Signal: {result['combined_signal']:.3f}
            
            Top Hypotheses:
            """)
            
            for hyp in result['hypothesis_summary'][:5]:
                logging.info(f"  - {hyp}")
                
        mt5.shutdown()
        
    # مسح جميع الأزواج
    logging.info("\n" + "="*50)
    logging.info("Full Market Scan")
    logging.info("="*50)
    
    opportunities = system.scan_all_pairs()
    
    if opportunities:
        logging.info(f"\nFound {len(opportunities)} trading opportunities:")
        
        for i, opp in enumerate(opportunities[:5], 1):
            logging.info(f"""
            {i}. {opp['symbol']} {opp['timeframe']}
               Decision: {opp['final_decision']}
               Confidence: {opp['total_confidence']:.1%}
            """)
            
        # توليد إشارات التداول
        signals = system.generate_trading_signals(opportunities)
        
        # حفظ التقرير
        system.save_analysis_report(opportunities, signals)
    else:
        logging.info("No trading opportunities found")
        
if __name__ == "__main__":
    main()