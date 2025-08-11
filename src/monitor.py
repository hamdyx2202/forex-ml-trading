import asyncio
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Optional
from loguru import logger
import schedule
import time
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


class TradingMonitor:
    """نظام المراقبة والتنبيهات"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_path = self.config["database"]["path"]
        
        logger.add("logs/monitor.log", rotation="1 day", retention="30 days")
        
        # Telegram bot setup
        self.telegram_enabled = self.config['monitoring']['telegram_enabled']
        if self.telegram_enabled:
            self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            
            if self.bot_token and self.chat_id:
                self.bot = Bot(token=self.bot_token)
            else:
                logger.warning("Telegram credentials not found")
                self.telegram_enabled = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown': 0.10,  # 10%
            'daily_loss': 0.05,    # 5%
            'error_count': 5,      # errors per hour
            'disconnection_time': 300  # 5 minutes
        }
        
        # Tracking
        self.error_count = 0
        self.last_heartbeat = datetime.now()
        self.alerts_sent = {}
        
    async def send_telegram_message(self, message: str, parse_mode: str = 'Markdown'):
        """إرسال رسالة عبر Telegram"""
        if not self.telegram_enabled:
            return
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
    
    def check_system_health(self) -> Dict:
        """فحص صحة النظام"""
        health_status = {
            'status': 'healthy',
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check database connection
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1")
            conn.close()
        except Exception as e:
            health_status['status'] = 'critical'
            health_status['issues'].append(f"Database error: {str(e)}")
        
        # Check last heartbeat
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).seconds
        if time_since_heartbeat > self.alert_thresholds['disconnection_time']:
            health_status['status'] = 'warning'
            health_status['issues'].append(f"No heartbeat for {time_since_heartbeat} seconds")
        
        # Check error rate
        if self.error_count > self.alert_thresholds['error_count']:
            health_status['status'] = 'warning'
            health_status['issues'].append(f"High error rate: {self.error_count} errors")
        
        return health_status
    
    def check_trading_performance(self) -> Dict:
        """فحص أداء التداول"""
        conn = sqlite3.connect(self.db_path)
        
        # Get today's trades
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        query = """
            SELECT * FROM trade_history 
            WHERE close_time >= ?
            ORDER BY close_time DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(today_start,))
        
        performance = {
            'daily_trades': len(df),
            'daily_pnl': df['pnl'].sum() if not df.empty else 0,
            'winning_trades': len(df[df['pnl'] > 0]) if not df.empty else 0,
            'losing_trades': len(df[df['pnl'] < 0]) if not df.empty else 0,
            'largest_win': df['pnl'].max() if not df.empty else 0,
            'largest_loss': df['pnl'].min() if not df.empty else 0
        }
        
        # Calculate drawdown
        if not df.empty:
            cumulative_pnl = df.sort_values('close_time')['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / running_max.abs()
            performance['current_drawdown'] = drawdown.min()
        else:
            performance['current_drawdown'] = 0
        
        conn.close()
        
        return performance
    
    def check_alerts(self):
        """فحص وإرسال التنبيهات"""
        # System health
        health = self.check_system_health()
        if health['status'] != 'healthy':
            alert_message = f"⚠️ *System Health Alert*\n"
            alert_message += f"Status: {health['status']}\n"
            alert_message += f"Issues:\n"
            for issue in health['issues']:
                alert_message += f"• {issue}\n"
            
            self._send_alert('system_health', alert_message)
        
        # Trading performance
        performance = self.check_trading_performance()
        
        # Check daily loss
        if performance['daily_pnl'] < 0:
            loss_percentage = abs(performance['daily_pnl']) / 10000  # Assuming $10k account
            if loss_percentage > self.alert_thresholds['daily_loss']:
                alert_message = f"🔴 *Daily Loss Alert*\n"
                alert_message += f"Daily P&L: ${performance['daily_pnl']:.2f}\n"
                alert_message += f"Loss: {loss_percentage:.1%}\n"
                
                self._send_alert('daily_loss', alert_message)
        
        # Check drawdown
        if abs(performance['current_drawdown']) > self.alert_thresholds['max_drawdown']:
            alert_message = f"📉 *Drawdown Alert*\n"
            alert_message += f"Current Drawdown: {performance['current_drawdown']:.1%}\n"
            alert_message += f"Threshold: {self.alert_thresholds['max_drawdown']:.1%}\n"
            
            self._send_alert('drawdown', alert_message)
    
    def _send_alert(self, alert_type: str, message: str):
        """إرسال تنبيه مع تجنب التكرار"""
        # Check if we already sent this alert recently
        if alert_type in self.alerts_sent:
            time_since_last = (datetime.now() - self.alerts_sent[alert_type]).seconds
            if time_since_last < 3600:  # 1 hour cooldown
                return
        
        # Send alert
        asyncio.create_task(self.send_telegram_message(message))
        self.alerts_sent[alert_type] = datetime.now()
        
        logger.warning(f"Alert sent: {alert_type}")
    
    def generate_daily_report(self) -> str:
        """توليد تقرير يومي"""
        performance = self.check_trading_performance()
        
        # Get account info (would need trader instance in real implementation)
        account_balance = 10000  # Placeholder
        
        report = f"📊 *Daily Trading Report*\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        report += f"*Performance Summary*\n"
        report += f"• Total Trades: {performance['daily_trades']}\n"
        report += f"• Winning Trades: {performance['winning_trades']}\n"
        report += f"• Losing Trades: {performance['losing_trades']}\n"
        
        if performance['daily_trades'] > 0:
            win_rate = performance['winning_trades'] / performance['daily_trades']
            report += f"• Win Rate: {win_rate:.1%}\n"
        
        report += f"\n*P&L Summary*\n"
        report += f"• Daily P&L: ${performance['daily_pnl']:.2f}\n"
        report += f"• Largest Win: ${performance['largest_win']:.2f}\n"
        report += f"• Largest Loss: ${performance['largest_loss']:.2f}\n"
        report += f"• Current Drawdown: {performance['current_drawdown']:.1%}\n"
        
        # Get weekly stats
        weekly_stats = self._get_weekly_stats()
        report += f"\n*Weekly Performance*\n"
        report += f"• Weekly P&L: ${weekly_stats['weekly_pnl']:.2f}\n"
        report += f"• Weekly Trades: {weekly_stats['weekly_trades']}\n"
        report += f"• Weekly Win Rate: {weekly_stats['weekly_win_rate']:.1%}\n"
        
        return report
    
    def _get_weekly_stats(self) -> Dict:
        """الحصول على إحصائيات أسبوعية"""
        conn = sqlite3.connect(self.db_path)
        
        week_start = datetime.now() - timedelta(days=7)
        
        query = """
            SELECT * FROM trade_history 
            WHERE close_time >= ?
        """
        
        df = pd.read_sql_query(query, conn, params=(week_start,))
        conn.close()
        
        if df.empty:
            return {
                'weekly_trades': 0,
                'weekly_pnl': 0,
                'weekly_win_rate': 0
            }
        
        winning_trades = len(df[df['pnl'] > 0])
        
        return {
            'weekly_trades': len(df),
            'weekly_pnl': df['pnl'].sum(),
            'weekly_win_rate': winning_trades / len(df) if len(df) > 0 else 0
        }
    
    def log_trade(self, trade_info: Dict):
        """تسجيل معلومات الصفقة"""
        logger.info(f"Trade executed: {trade_info}")
        
        # Send notification
        message = f"🔔 *New Trade*\n"
        message += f"Symbol: {trade_info.get('symbol', 'N/A')}\n"
        message += f"Direction: {trade_info.get('direction', 'N/A')}\n"
        message += f"Volume: {trade_info.get('volume', 'N/A')}\n"
        message += f"Entry: {trade_info.get('entry_price', 'N/A')}\n"
        message += f"SL: {trade_info.get('stop_loss', 'N/A')}\n"
        message += f"TP: {trade_info.get('take_profit', 'N/A')}\n"
        
        asyncio.create_task(self.send_telegram_message(message))
    
    def log_error(self, error_message: str, error_type: str = "general"):
        """تسجيل الأخطاء"""
        logger.error(f"{error_type}: {error_message}")
        self.error_count += 1
        
        # Send critical errors immediately
        if error_type == "critical":
            message = f"🚨 *Critical Error*\n{error_message}"
            asyncio.create_task(self.send_telegram_message(message))
    
    def update_heartbeat(self):
        """تحديث نبضة النظام"""
        self.last_heartbeat = datetime.now()
        
        # Reset error count every hour
        if self.last_heartbeat.minute == 0:
            self.error_count = 0
    
    def start_monitoring(self):
        """بدء المراقبة المجدولة"""
        logger.info("Starting monitoring service")
        
        # Schedule tasks
        schedule.every(5).minutes.do(self.check_alerts)
        schedule.every().day.at("18:00").do(lambda: asyncio.create_task(
            self.send_telegram_message(self.generate_daily_report())
        ))
        schedule.every().hour.do(self.update_heartbeat)
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    async def setup_telegram_bot(self):
        """إعداد bot Telegram للأوامر"""
        if not self.telegram_enabled:
            return
        
        application = Application.builder().token(self.bot_token).build()
        
        # Command handlers
        application.add_handler(CommandHandler("status", self.cmd_status))
        application.add_handler(CommandHandler("report", self.cmd_report))
        application.add_handler(CommandHandler("performance", self.cmd_performance))
        application.add_handler(CommandHandler("help", self.cmd_help))
        
        # Start bot
        await application.run_polling()
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج أمر /status"""
        health = self.check_system_health()
        
        message = f"*System Status*\n"
        message += f"Status: {health['status']}\n"
        
        if health['issues']:
            message += f"Issues:\n"
            for issue in health['issues']:
                message += f"• {issue}\n"
        else:
            message += "All systems operational ✅"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج أمر /report"""
        report = self.generate_daily_report()
        await update.message.reply_text(report, parse_mode='Markdown')
    
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج أمر /performance"""
        performance = self.check_trading_performance()
        
        message = f"*Current Performance*\n"
        message += f"Daily P&L: ${performance['daily_pnl']:.2f}\n"
        message += f"Daily Trades: {performance['daily_trades']}\n"
        message += f"Win/Loss: {performance['winning_trades']}/{performance['losing_trades']}\n"
        message += f"Drawdown: {performance['current_drawdown']:.1%}\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج أمر /help"""
        message = "*Available Commands*\n"
        message += "/status - System health status\n"
        message += "/report - Daily trading report\n"
        message += "/performance - Current performance\n"
        message += "/help - Show this help message"
        
        await update.message.reply_text(message, parse_mode='Markdown')


if __name__ == "__main__":
    monitor = TradingMonitor()
    
    # Start monitoring
    # monitor.start_monitoring()
    
    # Or setup Telegram bot
    # asyncio.run(monitor.setup_telegram_bot())