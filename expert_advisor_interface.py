#!/usr/bin/env python3
"""
🤖 Expert Advisor Interface
📊 واجهة للتواصل مع Expert Advisor
🎯 يدعم إرسال واستقبال الإشارات والأهداف
"""

import json
import time
import socket
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class ExpertAdvisorInterface:
    """واجهة التواصل مع الاكسبيرت"""
    
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.signals_file = Path('signals/active_signals.json')
        self.trades_file = Path('trades/active_trades.json')
        self.is_running = False
        
        # إنشاء المجلدات
        self.signals_file.parent.mkdir(exist_ok=True)
        self.trades_file.parent.mkdir(exist_ok=True)
        
    def read_signals(self) -> List[Dict]:
        """قراءة الإشارات النشطة"""
        if self.signals_file.exists():
            with open(self.signals_file, 'r') as f:
                return json.load(f)
        return []
    
    def write_signal(self, signal: Dict):
        """كتابة إشارة جديدة"""
        signals = self.read_signals()
        signals.append(signal)
        
        with open(self.signals_file, 'w') as f:
            json.dump(signals, f, indent=2)
    
    def clear_old_signals(self, hours=24):
        """حذف الإشارات القديمة"""
        signals = self.read_signals()
        current_time = datetime.now()
        
        # فلترة الإشارات الحديثة فقط
        recent_signals = []
        for signal in signals:
            signal_time = datetime.fromisoformat(signal['timestamp'])
            if (current_time - signal_time).total_seconds() < hours * 3600:
                recent_signals.append(signal)
        
        # حفظ الإشارات الحديثة
        with open(self.signals_file, 'w') as f:
            json.dump(recent_signals, f, indent=2)
    
    def update_trade_status(self, magic_number: int, status: str, 
                           profit: float = 0, close_reason: str = ""):
        """تحديث حالة الصفقة"""
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                trades = json.load(f)
        
        # البحث عن الصفقة وتحديثها
        for trade in trades:
            if trade.get('magic_number') == magic_number:
                trade['status'] = status
                trade['profit'] = profit
                trade['close_reason'] = close_reason
                trade['close_time'] = datetime.now().isoformat()
                break
        else:
            # إضافة صفقة جديدة
            trades.append({
                'magic_number': magic_number,
                'status': status,
                'profit': profit,
                'close_reason': close_reason,
                'open_time': datetime.now().isoformat()
            })
        
        # حفظ التحديثات
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def start_socket_server(self):
        """بدء خادم Socket للتواصل المباشر"""
        self.is_running = True
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"🌐 Expert Advisor Interface listening on {self.host}:{self.port}")
        
        while self.is_running:
            try:
                client_socket, address = server_socket.accept()
                print(f"📡 Connection from {address}")
                
                # معالجة الطلب في thread منفصل
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                client_thread.start()
                
            except Exception as e:
                print(f"❌ Server error: {e}")
        
        server_socket.close()
    
    def handle_client(self, client_socket):
        """معالجة طلبات العميل"""
        try:
            # استقبال البيانات
            data = client_socket.recv(4096).decode('utf-8')
            request = json.loads(data)
            
            # معالجة الطلب
            if request['action'] == 'GET_SIGNALS':
                # إرسال الإشارات النشطة
                signals = self.read_signals()
                response = {'status': 'OK', 'signals': signals}
                
            elif request['action'] == 'UPDATE_TRADE':
                # تحديث حالة الصفقة
                self.update_trade_status(
                    request['magic_number'],
                    request['status'],
                    request.get('profit', 0),
                    request.get('close_reason', '')
                )
                response = {'status': 'OK'}
                
            elif request['action'] == 'HEARTBEAT':
                # نبضة للتأكد من الاتصال
                response = {'status': 'OK', 'timestamp': datetime.now().isoformat()}
                
            else:
                response = {'status': 'ERROR', 'message': 'Unknown action'}
            
            # إرسال الرد
            client_socket.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Client error: {e}")
            error_response = {'status': 'ERROR', 'message': str(e)}
            client_socket.send(json.dumps(error_response).encode('utf-8'))
        
        finally:
            client_socket.close()
    
    def get_performance_stats(self) -> Dict:
        """حساب إحصائيات الأداء"""
        if not self.trades_file.exists():
            return {}
        
        with open(self.trades_file, 'r') as f:
            trades = json.load(f)
        
        # حساب الإحصائيات
        total_trades = len(trades)
        closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) < 0]
        
        total_profit = sum(t.get('profit', 0) for t in closed_trades)
        
        stats = {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_profit': total_profit,
            'average_profit': total_profit / len(closed_trades) if closed_trades else 0
        }
        
        return stats

# مثال على صيغة الإشارة للاكسبيرت
SIGNAL_FORMAT = {
    "symbol": "EURUSD",
    "action": "BUY",  # BUY, SELL, CLOSE
    "entry_price": 1.0850,
    "stop_loss": 1.0820,
    "take_profit_1": 1.0880,
    "take_profit_2": 1.0900,
    "take_profit_3": 1.0930,
    "confidence": 0.75,
    "strategy": "scalping",
    "magic_number": 123456,
    "timestamp": "2024-01-01T12:00:00",
    "lot_size": 0.01,  # يمكن حسابه في الاكسبيرت
    "partial_close": [0.4, 0.3, 0.3],  # نسب الإغلاق الجزئي
    "trailing_stop": True,
    "breakeven_pips": 20
}

if __name__ == "__main__":
    # تشغيل الواجهة
    interface = ExpertAdvisorInterface()
    
    # مثال على إضافة إشارة
    test_signal = {
        "symbol": "EURUSDm",
        "action": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit_1": 1.0880,
        "take_profit_2": 1.0900,
        "take_profit_3": 1.0930,
        "confidence": 0.75,
        "strategy": "scalping",
        "magic_number": int(time.time() % 1000000),
        "timestamp": datetime.now().isoformat()
    }
    
    interface.write_signal(test_signal)
    print("✅ Test signal added")
    
    # بدء الخادم
    print("🚀 Starting Expert Advisor Interface...")
    interface.start_socket_server()