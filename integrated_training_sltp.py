#!/usr/bin/env python3
"""
Integrated Training System with SL/TP Optimization
نظام تدريب متكامل مع تحسين وقف الخسارة والأهداف
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from loguru import logger
import sqlite3
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# إضافة المسار
import sys
sys.path.append(str(Path(__file__).parent))

from advanced_learner_unified_sltp import AdvancedLearnerWithSLTP
from continuous_learner_unified_sltp import ContinuousLearnerWithSLTP
from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75
from instrument_manager import InstrumentManager
from performance_tracker import PerformanceTracker

class IntegratedTrainingSystemSLTP:
    """نظام تدريب متكامل يجمع بين التعلم المتقدم والمستمر مع SL/TP"""
    
    def __init__(self):
        self.advanced_learner = AdvancedLearnerWithSLTP()
        self.continuous_learner = ContinuousLearnerWithSLTP()
        self.instrument_manager = InstrumentManager()
        self.performance_tracker = PerformanceTracker()
        
        self.models_dir = Path("models/unified_sltp")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_config = {
            'min_data_points': 1000,
            'test_size': 0.2,
            'validation_size': 0.1,
            'max_workers': mp.cpu_count() - 1,
            'batch_size': 5,
            'timeframes': ['M5', 'M15', 'H1', 'H4'],
            'sl_tp_optimization': {
                'lookforward_candles': 100,
                'min_sl_pips': 10,
                'max_sl_pips': 200,
                'min_tp_pips': 10,
                'max_tp_pips': 500,
                'risk_reward_min': 1.0,
                'risk_reward_target': 2.0
            }
        }
        
    def train_all_instruments(self, instrument_types=None, force_retrain=False):
        """تدريب جميع الأدوات المالية"""
        logger.info("🚀 Starting integrated training system with SL/TP optimization...")
        
        # الحصول على قائمة الأدوات
        if instrument_types:
            instruments = self.instrument_manager.get_instruments_by_types(instrument_types)
        else:
            instruments = self.instrument_manager.get_all_instruments()
            
        logger.info(f"📊 Found {len(instruments)} instruments to train")
        
        # تجميع مهام التدريب
        training_tasks = []
        for instrument in instruments:
            for timeframe in self.training_config['timeframes']:
                if force_retrain or self.needs_training(instrument['symbol'], timeframe):
                    training_tasks.append((instrument['symbol'], timeframe, instrument['type']))
                    
        logger.info(f"📋 Total training tasks: {len(training_tasks)}")
        
        # تنفيذ التدريب بالتوازي
        results = self.parallel_training(training_tasks)
        
        # تلخيص النتائج
        self.summarize_results(results)
        
        # بدء التعلم المستمر للأدوات المهمة
        important_pairs = self.get_important_pairs(instruments)
        self.start_continuous_learning(important_pairs)
        
        return results
        
    def parallel_training(self, training_tasks):
        """تدريب متوازي للنماذج"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.training_config['max_workers']) as executor:
            # تقسيم المهام إلى دفعات
            batches = [training_tasks[i:i+self.training_config['batch_size']] 
                      for i in range(0, len(training_tasks), self.training_config['batch_size'])]
            
            future_to_task = {}
            
            for batch in batches:
                for task in batch:
                    future = executor.submit(self.train_single_model, *task)
                    future_to_task[future] = task
                    
                # انتظار اكتمال الدفعة الحالية
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            logger.info(f"✅ {task[0]} {task[1]} - Accuracy: {result['accuracy']:.2%}, "
                                      f"SL MAE: {result['sl_mae']:.1f}, TP MAE: {result['tp_mae']:.1f}")
                        else:
                            logger.warning(f"❌ {task[0]} {task[1]} - {result['error']}")
                            
                    except Exception as e:
                        logger.error(f"Error training {task[0]} {task[1]}: {str(e)}")
                        results.append({
                            'pair': task[0],
                            'timeframe': task[1],
                            'success': False,
                            'error': str(e)
                        })
                        
        return results
        
    def train_single_model(self, symbol, timeframe, instrument_type):
        """تدريب نموذج واحد مع SL/TP محسّن"""
        try:
            logger.info(f"🎯 Training {symbol} {timeframe} with enhanced SL/TP...")
            
            # جمع البيانات
            df = self.load_training_data(symbol, timeframe)
            if df is None or len(df) < self.training_config['min_data_points']:
                return {
                    'pair': symbol,
                    'timeframe': timeframe,
                    'success': False,
                    'error': 'Insufficient data'
                }
                
            # إضافة الميزات
            feature_engineer = AdaptiveFeatureEngineer75(target_features=75)
            df_features = feature_engineer.engineer_features(df, symbol)
            
            # حساب SL/TP الأمثل بناءً على نوع الأداة
            df_with_targets = self.calculate_optimal_targets_by_type(
                df_features, symbol, instrument_type
            )
            
            # تدريب النموذج
            success = self.advanced_learner.train_model_with_sltp(symbol, timeframe)
            
            if success:
                # تقييم النموذج
                evaluation = self.evaluate_model(symbol, timeframe, df_with_targets)
                
                return {
                    'pair': symbol,
                    'timeframe': timeframe,
                    'success': True,
                    'accuracy': evaluation['accuracy'],
                    'sl_mae': evaluation['sl_mae'],
                    'tp_mae': evaluation['tp_mae'],
                    'sharpe_ratio': evaluation['sharpe_ratio'],
                    'max_drawdown': evaluation['max_drawdown'],
                    'win_rate': evaluation['win_rate'],
                    'avg_rr': evaluation['avg_rr']
                }
            else:
                return {
                    'pair': symbol,
                    'timeframe': timeframe,
                    'success': False,
                    'error': 'Training failed'
                }
                
        except Exception as e:
            logger.error(f"Error in train_single_model: {str(e)}")
            return {
                'pair': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': str(e)
            }
            
    def calculate_optimal_targets_by_type(self, df, symbol, instrument_type):
        """حساب SL/TP الأمثل حسب نوع الأداة"""
        logger.info(f"💡 Calculating optimal targets for {instrument_type} instrument...")
        
        df = df.copy()
        config = self.training_config['sl_tp_optimization']
        
        # إعدادات خاصة بنوع الأداة
        type_configs = {
            'forex_major': {
                'sl_multiplier': 1.0,
                'tp_multiplier': 2.0,
                'min_sl': 10,
                'typical_sl': 30,
                'typical_tp': 60
            },
            'forex_cross': {
                'sl_multiplier': 1.2,
                'tp_multiplier': 2.2,
                'min_sl': 15,
                'typical_sl': 40,
                'typical_tp': 80
            },
            'metals': {
                'sl_multiplier': 1.5,
                'tp_multiplier': 2.5,
                'min_sl': 30,
                'typical_sl': 100,
                'typical_tp': 200
            },
            'indices': {
                'sl_multiplier': 1.3,
                'tp_multiplier': 2.0,
                'min_sl': 20,
                'typical_sl': 50,
                'typical_tp': 100
            },
            'crypto': {
                'sl_multiplier': 2.0,
                'tp_multiplier': 3.0,
                'min_sl': 50,
                'typical_sl': 150,
                'typical_tp': 300
            },
            'energy': {
                'sl_multiplier': 1.8,
                'tp_multiplier': 2.5,
                'min_sl': 40,
                'typical_sl': 80,
                'typical_tp': 160
            }
        }
        
        type_config = type_configs.get(instrument_type, type_configs['forex_major'])
        
        # حساب قيمة النقطة
        pip_value = self._get_pip_value(symbol)
        
        # إضافة أعمدة
        df['optimal_sl_pips'] = 0.0
        df['optimal_tp_pips'] = 0.0
        df['trade_quality'] = 'normal'
        df['confidence_level'] = 0.5
        
        lookforward = config['lookforward_candles']
        
        for i in range(len(df) - lookforward):
            current_price = df['close'].iloc[i]
            future_data = df.iloc[i+1:i+lookforward+1]
            
            # تحليل السوق
            volatility = df['ATR'].iloc[i] / current_price
            trend_strength = self._calculate_trend_strength(df.iloc[max(0, i-50):i+1])
            
            # تحديد الإشارة والثقة
            signal, confidence = self._determine_signal_with_confidence(df.iloc[i])
            
            if signal in ['BUY', 'SELL']:
                # حساب الحركة المستقبلية المحتملة
                if signal == 'BUY':
                    max_favorable = future_data['high'].max() - current_price
                    max_adverse = current_price - future_data['low'].min()
                else:
                    max_favorable = current_price - future_data['low'].min()
                    max_adverse = future_data['high'].max() - current_price
                
                # حساب SL الأمثل
                base_sl = max(max_adverse * 0.7, type_config['min_sl'] * pip_value)
                volatility_adjusted_sl = base_sl * (1 + volatility * 2)
                optimal_sl = min(volatility_adjusted_sl * type_config['sl_multiplier'], 
                               config['max_sl_pips'] * pip_value)
                
                # حساب TP الأمثل
                base_tp = max_favorable * 0.7  # استهداف 70% من الحركة المحتملة
                
                # تعديل بناءً على قوة الترند
                if trend_strength > 0.7:
                    base_tp *= 1.3  # أهداف أكبر في ترند قوي
                elif trend_strength < 0.3:
                    base_tp *= 0.8  # أهداف أصغر في سوق جانبي
                
                optimal_tp = min(base_tp * type_config['tp_multiplier'], 
                               config['max_tp_pips'] * pip_value)
                
                # ضمان نسبة مخاطرة/عائد مناسبة
                rr_ratio = optimal_tp / optimal_sl if optimal_sl > 0 else 1
                
                if rr_ratio < config['risk_reward_min']:
                    # تعديل TP لتحقيق الحد الأدنى من RR
                    optimal_tp = optimal_sl * config['risk_reward_min']
                elif rr_ratio > 5:
                    # تقليل TP إذا كان غير واقعي
                    optimal_tp = optimal_sl * 4
                
                # تعديل بناءً على الثقة
                if confidence < 0.6:
                    optimal_sl *= 0.8  # تقليل المخاطرة
                    optimal_tp *= 0.9
                elif confidence > 0.8:
                    optimal_tp *= 1.1  # زيادة الهدف قليلاً
                
                # تحويل إلى نقاط
                df.loc[i, 'optimal_sl_pips'] = optimal_sl / pip_value
                df.loc[i, 'optimal_tp_pips'] = optimal_tp / pip_value
                df.loc[i, 'confidence_level'] = confidence
                
                # تقييم جودة الصفقة
                if rr_ratio >= 2 and confidence >= 0.7:
                    df.loc[i, 'trade_quality'] = 'high'
                elif rr_ratio >= 1.5 or confidence >= 0.6:
                    df.loc[i, 'trade_quality'] = 'normal'
                else:
                    df.loc[i, 'trade_quality'] = 'low'
            else:
                # قيم افتراضية لـ NO_TRADE
                df.loc[i, 'optimal_sl_pips'] = type_config['typical_sl']
                df.loc[i, 'optimal_tp_pips'] = type_config['typical_tp']
                df.loc[i, 'confidence_level'] = 0.3
                
        # إحصائيات
        trades = df[df['trade_quality'] != 'normal']
        if len(trades) > 0:
            logger.info(f"   Trade quality distribution: {df['trade_quality'].value_counts().to_dict()}")
            logger.info(f"   Avg SL: {df['optimal_sl_pips'].mean():.1f} pips")
            logger.info(f"   Avg TP: {df['optimal_tp_pips'].mean():.1f} pips")
            logger.info(f"   Avg R:R: {(df['optimal_tp_pips'] / df['optimal_sl_pips']).mean():.2f}")
            
        return df
        
    def _determine_signal_with_confidence(self, row):
        """تحديد الإشارة مع مستوى الثقة"""
        indicators = {}
        signal_scores = {'BUY': 0, 'SELL': 0}
        
        # جمع المؤشرات
        if 'RSI' in row and not pd.isna(row['RSI']):
            if row['RSI'] < 30:
                signal_scores['BUY'] += 2
                indicators['RSI'] = 'oversold'
            elif row['RSI'] > 70:
                signal_scores['SELL'] += 2
                indicators['RSI'] = 'overbought'
                
        if 'MACD' in row and 'MACD_signal' in row:
            if row['MACD'] > row['MACD_signal']:
                signal_scores['BUY'] += 1.5
                indicators['MACD'] = 'bullish'
            else:
                signal_scores['SELL'] += 1.5
                indicators['MACD'] = 'bearish'
                
        if 'BB_upper' in row and 'BB_lower' in row:
            bb_position = (row['close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])
            if bb_position < 0.2:
                signal_scores['BUY'] += 1.5
                indicators['BB'] = 'lower'
            elif bb_position > 0.8:
                signal_scores['SELL'] += 1.5
                indicators['BB'] = 'upper'
                
        if 'distance_to_support_pct' in row and 'distance_to_resistance_pct' in row:
            if row['distance_to_support_pct'] < 1.0:
                signal_scores['BUY'] += 2
                indicators['SR'] = 'near_support'
            elif row['distance_to_resistance_pct'] < 1.0:
                signal_scores['SELL'] += 2
                indicators['SR'] = 'near_resistance'
                
        # حساب الإشارة والثقة
        total_score = signal_scores['BUY'] + signal_scores['SELL']
        
        if total_score == 0:
            return 'NO_TRADE', 0.0
            
        if signal_scores['BUY'] > signal_scores['SELL'] * 1.5:
            confidence = signal_scores['BUY'] / (total_score + 2)  # تطبيع الثقة
            return 'BUY', min(confidence, 0.95)
        elif signal_scores['SELL'] > signal_scores['BUY'] * 1.5:
            confidence = signal_scores['SELL'] / (total_score + 2)
            return 'SELL', min(confidence, 0.95)
        else:
            return 'NO_TRADE', 0.3
            
    def _calculate_trend_strength(self, df):
        """حساب قوة الترند"""
        if len(df) < 20:
            return 0.5
            
        # حساب ميل خط الاتجاه
        prices = df['close'].values
        x = np.arange(len(prices))
        
        # الانحدار الخطي
        slope = np.polyfit(x, prices, 1)[0]
        
        # تطبيع القوة
        price_range = prices.max() - prices.min()
        if price_range > 0:
            normalized_slope = abs(slope) / price_range * len(prices)
            return min(normalized_slope, 1.0)
        
        return 0.5
        
    def evaluate_model(self, symbol, timeframe, test_data):
        """تقييم شامل للنموذج"""
        try:
            # تحميل النموذج
            filename = f"{symbol}_{timeframe}_unified.pkl"
            filepath = self.models_dir / filename
            
            if not filepath.exists():
                return {
                    'accuracy': 0,
                    'sl_mae': 0,
                    'tp_mae': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'avg_rr': 0
                }
                
            model_data = joblib.load(filepath)
            
            # محاكاة التداول على بيانات الاختبار
            trades = []
            
            for i in range(100, len(test_data) - 1):
                # التنبؤ
                features = test_data.iloc[i:i+1]
                prediction = self.advanced_learner.predict_with_sltp(
                    features, symbol, timeframe
                )
                
                if prediction and prediction['signal'] != 'NO_TRADE':
                    # محاكاة الصفقة
                    entry_price = test_data['close'].iloc[i]
                    sl_pips = prediction['sl_pips']
                    tp_pips = prediction['tp_pips']
                    
                    # تتبع النتيجة
                    future_prices = test_data.iloc[i+1:i+50]
                    
                    trade_result = self._simulate_trade(
                        prediction['signal'], entry_price, 
                        sl_pips, tp_pips, future_prices
                    )
                    
                    trades.append(trade_result)
                    
            # حساب المقاييس
            if trades:
                df_trades = pd.DataFrame(trades)
                
                wins = df_trades['result'] == 'win'
                win_rate = wins.sum() / len(df_trades)
                
                returns = df_trades['pnl_pips'].values
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                
                cumulative_pnl = df_trades['pnl_pips'].cumsum()
                max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
                
                avg_rr = df_trades['actual_rr'].mean()
                
                return {
                    'accuracy': model_data['metrics']['signal_accuracy'],
                    'sl_mae': model_data['metrics']['sl_mae'],
                    'tp_mae': model_data['metrics']['tp_mae'],
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'avg_rr': avg_rr
                }
            else:
                return {
                    'accuracy': model_data['metrics']['signal_accuracy'],
                    'sl_mae': model_data['metrics']['sl_mae'],
                    'tp_mae': model_data['metrics']['tp_mae'],
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'avg_rr': 0
                }
                
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'accuracy': 0,
                'sl_mae': 0,
                'tp_mae': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_rr': 0
            }
            
    def _simulate_trade(self, signal, entry_price, sl_pips, tp_pips, future_prices):
        """محاكاة صفقة واحدة"""
        pip_value = 0.0001  # افتراضي
        
        if signal == 'BUY':
            sl_price = entry_price - (sl_pips * pip_value)
            tp_price = entry_price + (tp_pips * pip_value)
            
            for idx, row in future_prices.iterrows():
                if row['low'] <= sl_price:
                    return {
                        'result': 'loss',
                        'pnl_pips': -sl_pips,
                        'actual_rr': 0
                    }
                elif row['high'] >= tp_price:
                    return {
                        'result': 'win',
                        'pnl_pips': tp_pips,
                        'actual_rr': tp_pips / sl_pips
                    }
                    
        else:  # SELL
            sl_price = entry_price + (sl_pips * pip_value)
            tp_price = entry_price - (tp_pips * pip_value)
            
            for idx, row in future_prices.iterrows():
                if row['high'] >= sl_price:
                    return {
                        'result': 'loss',
                        'pnl_pips': -sl_pips,
                        'actual_rr': 0
                    }
                elif row['low'] <= tp_price:
                    return {
                        'result': 'win',
                        'pnl_pips': tp_pips,
                        'actual_rr': tp_pips / sl_pips
                    }
                    
        # لم يصل لأي هدف
        return {
            'result': 'open',
            'pnl_pips': 0,
            'actual_rr': 0
        }
        
    def needs_training(self, symbol, timeframe):
        """فحص ما إذا كان النموذج يحتاج تدريب"""
        filename = f"{symbol}_{timeframe}_unified.pkl"
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            return True
            
        try:
            model_data = joblib.load(filepath)
            training_date = datetime.fromisoformat(model_data['training_date'])
            
            # إعادة التدريب إذا مر أكثر من 7 أيام
            if (datetime.now() - training_date).days > 7:
                return True
                
            # إعادة التدريب إذا كان الأداء ضعيفاً
            if model_data['metrics']['signal_accuracy'] < 0.55:
                return True
                
        except:
            return True
            
        return False
        
    def load_training_data(self, symbol, timeframe):
        """تحميل بيانات التدريب"""
        try:
            conn = sqlite3.connect("trading_data.db")
            
            query = f"""
            SELECT * FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT 10000
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) > 0:
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            
        return None
        
    def get_important_pairs(self, instruments):
        """الحصول على الأزواج المهمة للتعلم المستمر"""
        important = []
        
        # الأزواج الرئيسية
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
        
        # المعادن المهمة
        metals = ['XAUUSD', 'XAGUSD']
        
        # المؤشرات الرئيسية
        indices = ['US30', 'NAS100', 'SP500']
        
        all_important = major_pairs + metals + indices
        
        for instrument in instruments:
            if instrument['symbol'] in all_important:
                important.append(instrument['symbol'])
                
        return important[:10]  # أعلى 10 أزواج
        
    def start_continuous_learning(self, pairs):
        """بدء التعلم المستمر للأزواج المهمة"""
        if pairs:
            logger.info(f"🔄 Starting continuous learning for {len(pairs)} important pairs...")
            self.continuous_learner.start_continuous_learning(
                pairs, 
                ['H1', 'H4']  # الأطر الزمنية الأكثر استقراراً
            )
            
    def summarize_results(self, results):
        """تلخيص نتائج التدريب"""
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        logger.info("\n" + "="*60)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*60)
        
        logger.info(f"✅ Successful: {len(successful)}")
        logger.info(f"❌ Failed: {len(failed)}")
        
        if successful:
            avg_accuracy = np.mean([r['accuracy'] for r in successful])
            avg_sl_mae = np.mean([r['sl_mae'] for r in successful])
            avg_tp_mae = np.mean([r['tp_mae'] for r in successful])
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in successful])
            avg_win_rate = np.mean([r.get('win_rate', 0) for r in successful])
            
            logger.info(f"\n📈 Average Metrics:")
            logger.info(f"   Accuracy: {avg_accuracy:.2%}")
            logger.info(f"   SL MAE: {avg_sl_mae:.1f} pips")
            logger.info(f"   TP MAE: {avg_tp_mae:.1f} pips")
            logger.info(f"   Sharpe Ratio: {avg_sharpe:.2f}")
            logger.info(f"   Win Rate: {avg_win_rate:.2%}")
            
        # حفظ التقرير
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }
        
        report_path = self.models_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📄 Report saved to: {report_path}")
        logger.info("="*60)
        
    def _get_pip_value(self, pair):
        """الحصول على قيمة النقطة"""
        if 'JPY' in pair:
            return 0.01
        elif 'XAU' in pair or 'GOLD' in pair:
            return 0.1
        elif 'BTC' in pair or any(idx in pair for idx in ['US30', 'NAS', 'DAX']):
            return 1.0
        else:
            return 0.0001


if __name__ == "__main__":
    # إنشاء النظام
    system = IntegratedTrainingSystemSLTP()
    
    # تدريب جميع الأدوات
    results = system.train_all_instruments(
        instrument_types=['forex_major', 'metals', 'indices'],
        force_retrain=False
    )
    
    logger.info("\n✅ Integrated training system completed!")
    logger.info("📊 Models are now optimized for SL/TP prediction")
    logger.info("🔄 Continuous learning is running in the background")