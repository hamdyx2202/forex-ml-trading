#!/usr/bin/env python3
"""
MT5 Bridge Server - خادم API للتواصل مع Expert Advisor
يستقبل طلبات من EA ويرد بإشارات التداول من نظام ML
"""

# Import Linux compatibility first
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# تحميل التوافق مع Linux أولاً
try:
    import src.linux_compatibility
except:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from loguru import logger
import threading
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# محاولة استيراد المكونات مع معالجة الأخطاء
from src.predictor import Predictor
from src.risk_manager import RiskManager
from src.feature_engineer import FeatureEngineer
from src.continuous_learner import ContinuousLearner
from src.advanced_learner import AdvancedLearner

# محاولة استيراد data_collector
try:
    from src.data_collector import MT5DataCollector
except ImportError:
    logger.warning("MT5DataCollector not available on Linux - using mock mode")
    MT5DataCollector = None

app = Flask(__name__)
CORS(app)  # السماح بطلبات من MT5

class MT5BridgeServer:
    """خادم الجسر بين MT5 والنظام الذكي"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # تهيئة المكونات
        self.predictor = Predictor(config_path)
        self.risk_manager = RiskManager(config_path)
        
        # MT5DataCollector اختياري على Linux
        if MT5DataCollector:
            try:
                self.data_collector = MT5DataCollector(config_path)
            except Exception as e:
                logger.warning(f"Could not initialize MT5DataCollector: {e}")
                self.data_collector = None
        else:
            self.data_collector = None
            
        self.feature_engineer = FeatureEngineer(config_path)
        self.continuous_learner = ContinuousLearner(config_path)
        self.advanced_learner = AdvancedLearner(config_path)
        
        # ذاكرة مؤقتة للإشارات
        self.recent_signals = {}
        self.active_trades = {}
        
        logger.add("logs/mt5_bridge.log", rotation="1 day", retention="30 days")
        logger.info("MT5 Bridge Server initialized")
        
        # بدء مهام الخلفية
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """بدء المهام في الخلفية"""
        # تحديث البيانات كل 5 دقائق
        data_thread = threading.Thread(target=self._update_data_loop, daemon=True)
        data_thread.start()
        
        # تنظيف الذاكرة كل ساعة
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _update_data_loop(self):
        """حلقة تحديث البيانات"""
        while True:
            try:
                if self.data_collector:
                    self.data_collector.update_all_pairs()
                else:
                    logger.info("Data collector not available - skipping update")
                time.sleep(300)  # 5 دقائق
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                time.sleep(60)
    
    def _cleanup_loop(self):
        """حلقة تنظيف الذاكرة"""
        while True:
            try:
                # حذف الإشارات القديمة (أكثر من ساعة)
                current_time = datetime.now()
                for symbol in list(self.recent_signals.keys()):
                    if (current_time - self.recent_signals[symbol]['timestamp']).seconds > 3600:
                        del self.recent_signals[symbol]
                
                time.sleep(3600)  # ساعة واحدة
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                time.sleep(60)
    
    def get_signal(self, symbol: str, price: float) -> Dict:
        """الحصول على إشارة تداول لزوج معين"""
        try:
            # التحقق من الإشارات الحديثة
            if symbol in self.recent_signals:
                signal_age = (datetime.now() - self.recent_signals[symbol]['timestamp']).seconds
                if signal_age < 60:  # إشارة أقل من دقيقة
                    logger.info(f"Returning cached signal for {symbol}")
                    return self.recent_signals[symbol]['signal']
            
            # البحث عن فرص باستخدام التعلم المتقدم
            opportunities = self.advanced_learner.find_high_quality_opportunities(symbol, "H1")
            
            if opportunities:
                best_opp = opportunities[0]
                signal = {
                    "action": best_opp['direction'],
                    "confidence": best_opp['confidence'],
                    "sl": best_opp['sl'],
                    "tp": best_opp['tp'],
                    "lot": self._calculate_lot_size(symbol, price, best_opp['sl']),
                    "reasons": best_opp['reasons'][:3]  # أول 3 أسباب
                }
            else:
                # استخدام النموذج العادي
                prediction = self.predictor.predict_latest(symbol, "H1")
                
                if not prediction or prediction['recommendation'] in ['NO_TRADE', 'HOLD']:
                    return {"action": "NO_TRADE", "confidence": 0}
                
                # حساب SL و TP
                atr = self._get_current_atr(symbol)
                
                if prediction['recommendation'] in ['BUY', 'STRONG_BUY']:
                    action = "BUY"
                    sl = price - (atr * 2)
                    tp = price + (atr * 3)
                else:
                    action = "SELL"
                    sl = price + (atr * 2)
                    tp = price - (atr * 3)
                
                signal = {
                    "action": action,
                    "confidence": prediction['confidence'],
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "lot": self._calculate_lot_size(symbol, price, sl),
                    "reasons": prediction.get('reasons', [])[:3]
                }
            
            # حفظ الإشارة في الذاكرة
            self.recent_signals[symbol] = {
                'signal': signal,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Generated signal for {symbol}: {signal['action']} with {signal['confidence']:.1%} confidence")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {"action": "NO_TRADE", "confidence": 0, "error": str(e)}
    
    def _calculate_lot_size(self, symbol: str, price: float, sl: float) -> float:
        """حساب حجم اللوت بناءً على المخاطرة"""
        try:
            # استخدام قيمة افتراضية للرصيد (سيتم تحديثها من EA)
            balance = 10000  # افتراضي
            risk_per_trade = self.config['risk']['max_risk_per_trade']
            
            # حساب المخاطرة بالدولار
            risk_amount = balance * risk_per_trade
            
            # حساب النقاط
            pip_value = 0.0001 if "JPY" not in symbol else 0.01
            sl_pips = abs(price - sl) / pip_value
            
            # حساب حجم اللوت
            lot_size = risk_amount / (sl_pips * 10)  # 10$ لكل نقطة لكل لوت
            
            # تقريب لأقرب 0.01
            lot_size = round(lot_size, 2)
            
            # التأكد من الحد الأدنى والأقصى
            lot_size = max(0.01, min(lot_size, 1.0))
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return 0.01  # الحد الأدنى
    
    def _get_current_atr(self, symbol: str) -> float:
        """الحصول على ATR الحالي"""
        try:
            if self.data_collector:
                df = self.data_collector.get_latest_data(symbol, "H1", limit=100)
                if df.empty:
                    return 0.001  # قيمة افتراضية
                
                df = self.feature_engineer.add_technical_indicators(df)
                return df['atr_14'].iloc[-1]
            else:
                # قيمة افتراضية على Linux
                return 0.001
            
        except Exception as e:
            logger.error(f"Error getting ATR: {e}")
            return 0.001
    
    def confirm_trade(self, trade_data: Dict) -> Dict:
        """تأكيد تنفيذ صفقة"""
        try:
            symbol = trade_data['symbol']
            action = trade_data['action']
            lot = trade_data['lot']
            timestamp = datetime.fromisoformat(trade_data['timestamp'])
            
            # حفظ الصفقة النشطة
            trade_id = f"{symbol}_{timestamp.timestamp()}"
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'action': action,
                'lot': lot,
                'timestamp': timestamp,
                'confirmed': True
            }
            
            logger.info(f"Trade confirmed: {symbol} {action} {lot} lots")
            
            return {
                "status": "confirmed",
                "trade_id": trade_id,
                "message": f"Trade {action} {symbol} confirmed"
            }
            
        except Exception as e:
            logger.error(f"Error confirming trade: {e}")
            return {"status": "error", "message": str(e)}
    
    def report_trade_result(self, result_data: Dict) -> Dict:
        """استقبال نتيجة صفقة مغلقة"""
        try:
            # استخراج البيانات
            symbol = result_data['symbol']
            volume = result_data['volume']
            profit = result_data['profit']
            price = result_data['price']
            timestamp = datetime.fromisoformat(result_data['timestamp'])
            
            # إنشاء سجل نتيجة التداول
            trade_result = {
                'trade_id': f"{symbol}_{timestamp.timestamp()}",
                'symbol': symbol,
                'timeframe': 'H1',
                'entry_time': timestamp - pd.Timedelta(hours=1),  # تقديري
                'exit_time': timestamp,
                'direction': 'BUY' if profit > 0 else 'SELL',  # تقديري
                'entry_price': price - (profit / (volume * 100000)),  # تقديري
                'exit_price': price,
                'pnl_pips': profit / (volume * 10),
                'volume': volume
            }
            
            # تمرير للتعلم المستمر
            self.continuous_learner.learn_from_trade(trade_result)
            
            logger.info(f"Trade result reported: {symbol} P/L: ${profit:.2f}")
            
            # تحليل الأداء
            performance_analysis = self._analyze_trade_performance(trade_result)
            
            return {
                "status": "received",
                "analysis": performance_analysis,
                "message": f"Result processed for {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Error reporting trade result: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_trade_performance(self, trade_result: Dict) -> Dict:
        """تحليل أداء الصفقة"""
        try:
            profit_loss = "ربح" if trade_result['pnl_pips'] > 0 else "خسارة"
            
            # حساب معدل النجاح الحالي
            # هنا يمكن إضافة منطق لحساب معدل النجاح من قاعدة البيانات
            
            return {
                "result": profit_loss,
                "pips": round(trade_result['pnl_pips'], 1),
                "learning_impact": "تم تحديث نموذج التعلم",
                "recommendation": "متابعة الأداء"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    def get_market_status(self) -> Dict:
        """الحصول على حالة السوق"""
        try:
            status = {
                "server_time": datetime.now().isoformat(),
                "active_trades": len(self.active_trades),
                "recent_signals": len(self.recent_signals),
                "market_status": "open",  # يمكن تحسينها
                "system_health": "healthy"
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {"error": str(e)}


# إنشاء instance من الخادم
bridge_server = MT5BridgeServer()

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/get_signal', methods=['POST'])
def get_signal():
    """الحصول على إشارة تداول"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        price = data.get('price')
        
        if not symbol or not price:
            return jsonify({"error": "Missing symbol or price"}), 400
        
        signal = bridge_server.get_signal(symbol, price)
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Error in get_signal endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/confirm_trade', methods=['POST'])
def confirm_trade():
    """تأكيد تنفيذ صفقة"""
    try:
        data = request.get_json()
        result = bridge_server.confirm_trade(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in confirm_trade endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/report_trade', methods=['POST'])
def report_trade():
    """استقبال نتيجة صفقة"""
    try:
        data = request.get_json()
        result = bridge_server.report_trade_result(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in report_trade endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """الحصول على حالة النظام"""
    try:
        status = bridge_server.get_market_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_balance', methods=['POST'])
def update_balance():
    """تحديث رصيد الحساب من EA"""
    try:
        data = request.get_json()
        balance = data.get('balance')
        
        if balance:
            # يمكن حفظ الرصيد لاستخدامه في حسابات حجم اللوت
            bridge_server.account_balance = balance
            logger.info(f"Account balance updated: ${balance}")
            
        return jsonify({"status": "updated"})
        
    except Exception as e:
        logger.error(f"Error updating balance: {e}")
        return jsonify({"error": str(e)}), 500

def run_server(host='0.0.0.0', port=5000):
    """تشغيل الخادم"""
    logger.info(f"Starting MT5 Bridge Server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    # يمكن تغيير المنفذ حسب الحاجة
    run_server(port=5000)