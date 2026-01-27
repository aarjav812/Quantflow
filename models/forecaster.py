"""
Stock Return Forecasting Engine
Clear, practical predictions with actionable insights
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


class StockForecaster:
    """
    Unified forecasting engine that provides clear, actionable predictions
    """
    
    def __init__(self, prices, dates=None):
        """
        Initialize with historical price data
        
        Parameters:
        -----------
        prices : array-like
            Historical closing prices
        dates : array-like, optional
            Corresponding dates
        """
        self.prices = pd.Series(prices).dropna()
        if dates is not None:
            self.dates = pd.to_datetime(dates)
            self.prices.index = self.dates[:len(self.prices)]
        else:
            self.dates = None
        
        self.returns = self.prices.pct_change().dropna()
        
    def analyze_trends(self):
        """
        Analyze historical trends and patterns
        """
        # Overall trend direction
        if len(self.prices) > 20:
            recent_20 = self.prices.iloc[-20:]
            trend_slope = np.polyfit(range(len(recent_20)), recent_20.values, 1)[0]
            trend_direction = "UPWARD" if trend_slope > 0 else "DOWNWARD"
            trend_strength = abs(trend_slope) / self.prices.mean() * 100
        else:
            trend_direction = "NEUTRAL"
            trend_strength = 0
        
        # Moving averages
        ma_20 = self.prices.rolling(20).mean().iloc[-1] if len(self.prices) >= 20 else self.prices.mean()
        ma_50 = self.prices.rolling(50).mean().iloc[-1] if len(self.prices) >= 50 else self.prices.mean()
        ma_200 = self.prices.rolling(200).mean().iloc[-1] if len(self.prices) >= 200 else self.prices.mean()
        
        current_price = self.prices.iloc[-1]
        
        # Position relative to MAs
        above_ma20 = current_price > ma_20
        above_ma50 = current_price > ma_50
        above_ma200 = current_price > ma_200
        
        # Momentum (Rate of Change)
        if len(self.prices) >= 10:
            momentum_10d = (current_price / self.prices.iloc[-10] - 1) * 100
        else:
            momentum_10d = 0
            
        if len(self.prices) >= 30:
            momentum_30d = (current_price / self.prices.iloc[-30] - 1) * 100
        else:
            momentum_30d = 0
        
        return {
            'current_price': float(current_price),
            'trend_direction': trend_direction,
            'trend_strength': float(trend_strength),
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'ma_200': float(ma_200),
            'above_ma20': bool(above_ma20),
            'above_ma50': bool(above_ma50),
            'above_ma200': bool(above_ma200),
            'momentum_10d': float(momentum_10d),
            'momentum_30d': float(momentum_30d),
            'bullish_signals': int(sum([above_ma20, above_ma50, above_ma200, momentum_10d > 0])),
        }
    
    def analyze_seasonality(self):
        """
        Analyze seasonal/periodic patterns in returns
        """
        if self.dates is None or len(self.returns) < 60:
            return None
        
        # Monthly returns
        try:
            ret_series = self.returns.copy()
            ret_series.index = pd.to_datetime(ret_series.index)
            
            monthly_returns = {}
            for month in range(1, 13):
                month_rets = ret_series[ret_series.index.month == month]
                if len(month_rets) > 0:
                    monthly_returns[month] = {
                        'avg_return': float(month_rets.mean() * 100),
                        'win_rate': float((month_rets > 0).mean() * 100),
                        'count': len(month_rets)
                    }
            
            # Find best and worst months
            if monthly_returns:
                best_month = max(monthly_returns.items(), key=lambda x: x[1]['avg_return'])
                worst_month = min(monthly_returns.items(), key=lambda x: x[1]['avg_return'])
                
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                return {
                    'monthly_returns': {month_names[k]: v for k, v in monthly_returns.items()},
                    'best_month': month_names[best_month[0]],
                    'best_month_return': best_month[1]['avg_return'],
                    'worst_month': month_names[worst_month[0]],
                    'worst_month_return': worst_month[1]['avg_return'],
                    'current_month': month_names[pd.Timestamp.now().month],
                }
        except:
            return None
        
        return None
    
    def forecast_returns(self, days=30):
        """
        Forecast expected returns for the next N days
        
        Combines:
        - Statistical trend projection
        - Historical average returns
        - Volatility-adjusted confidence intervals
        """
        # Historical statistics
        avg_daily_return = self.returns.mean()
        daily_volatility = self.returns.std()
        
        # Projected return over forecast period
        expected_return = avg_daily_return * days
        
        # Adjust based on recent momentum
        if len(self.returns) >= 20:
            recent_momentum = self.returns.iloc[-20:].mean()
            momentum_adjustment = (recent_momentum - avg_daily_return) * 0.5  # 50% weight to recent
            expected_return += momentum_adjustment * days
        
        # Confidence intervals (based on volatility)
        period_volatility = daily_volatility * np.sqrt(days)
        
        # 68% confidence (1 sigma)
        lower_68 = expected_return - period_volatility
        upper_68 = expected_return + period_volatility
        
        # 95% confidence (2 sigma)
        lower_95 = expected_return - 2 * period_volatility
        upper_95 = expected_return + 2 * period_volatility
        
        # Price targets
        current_price = self.prices.iloc[-1]
        target_price = current_price * (1 + expected_return)
        lower_target = current_price * (1 + lower_68)
        upper_target = current_price * (1 + upper_68)
        
        # Confidence score (0-100)
        # Higher if recent trend aligns with forecast
        trend_info = self.analyze_trends()
        signals = trend_info['bullish_signals']
        
        if expected_return > 0:
            confidence = min(100, 50 + signals * 12.5)  # Max boost of 50
        else:
            confidence = min(100, 50 + (4 - signals) * 12.5)
        
        return {
            'forecast_days': days,
            'expected_return_pct': float(expected_return * 100),
            'lower_bound_pct': float(lower_68 * 100),
            'upper_bound_pct': float(upper_68 * 100),
            'lower_95_pct': float(lower_95 * 100),
            'upper_95_pct': float(upper_95 * 100),
            'current_price': float(current_price),
            'target_price': float(target_price),
            'lower_target': float(lower_target),
            'upper_target': float(upper_target),
            'confidence_score': float(confidence),
            'annualized_return': float(avg_daily_return * 252 * 100),
            'annualized_volatility': float(daily_volatility * np.sqrt(252) * 100),
        }
    
    def generate_recommendation(self, forecast_days=30):
        """
        Generate a clear BUY/HOLD/SELL recommendation
        """
        trends = self.analyze_trends()
        forecast = self.forecast_returns(forecast_days)
        seasonality = self.analyze_seasonality()
        
        # Scoring system
        score = 0
        reasons = []
        
        # 1. Expected return direction and magnitude
        exp_return = forecast['expected_return_pct']
        if exp_return > 5:
            score += 3
            reasons.append(f"Strong upside potential: +{exp_return:.1f}%")
        elif exp_return > 2:
            score += 2
            reasons.append(f"Moderate upside: +{exp_return:.1f}%")
        elif exp_return > 0:
            score += 1
            reasons.append(f"Slight upside: +{exp_return:.1f}%")
        elif exp_return > -2:
            score -= 1
            reasons.append(f"Slight downside risk: {exp_return:.1f}%")
        else:
            score -= 2
            reasons.append(f"Downside risk: {exp_return:.1f}%")
        
        # 2. Trend alignment
        if trends['trend_direction'] == 'UPWARD':
            score += 1
            reasons.append("Price in upward trend")
        elif trends['trend_direction'] == 'DOWNWARD':
            score -= 1
            reasons.append("Price in downward trend")
        
        # 3. Moving average signals
        if trends['above_ma20'] and trends['above_ma50']:
            score += 1
            reasons.append("Trading above key moving averages")
        elif not trends['above_ma20'] and not trends['above_ma50']:
            score -= 1
            reasons.append("Trading below key moving averages")
        
        # 4. Momentum
        if trends['momentum_10d'] > 3:
            score += 1
            reasons.append(f"Strong 10-day momentum: +{trends['momentum_10d']:.1f}%")
        elif trends['momentum_10d'] < -3:
            score -= 1
            reasons.append(f"Weak 10-day momentum: {trends['momentum_10d']:.1f}%")
        
        # 5. Seasonal factor
        if seasonality:
            current_month_data = seasonality['monthly_returns'].get(seasonality['current_month'])
            if current_month_data:
                if current_month_data['avg_return'] > 1:
                    score += 1
                    reasons.append(f"Historically strong month: {seasonality['current_month']}")
                elif current_month_data['avg_return'] < -1:
                    score -= 1
                    reasons.append(f"Historically weak month: {seasonality['current_month']}")
        
        # Generate recommendation
        if score >= 3:
            recommendation = "STRONG BUY"
            action = "Consider buying now for potential gains"
            color = "#22c55e"
        elif score >= 1:
            recommendation = "BUY"
            action = "Good entry point for long-term investors"
            color = "#4ade80"
        elif score >= -1:
            recommendation = "HOLD"
            action = "Wait for clearer signals before acting"
            color = "#fbbf24"
        elif score >= -3:
            recommendation = "SELL"
            action = "Consider reducing position"
            color = "#f97316"
        else:
            recommendation = "STRONG SELL"
            action = "High risk - consider exiting position"
            color = "#ef4444"
        
        return {
            'recommendation': recommendation,
            'action': action,
            'score': score,
            'color': color,
            'reasons': reasons[:5],  # Top 5 reasons
            'trends': trends,
            'forecast': forecast,
            'seasonality': seasonality,
        }
    
    def get_price_history_chart_data(self, days=365):
        """
        Get data formatted for charting
        """
        prices = self.prices.iloc[-days:] if len(self.prices) > days else self.prices
        
        # Calculate moving averages
        ma20 = prices.rolling(20).mean()
        ma50 = prices.rolling(50).mean()
        
        dates = [str(d.date()) if hasattr(d, 'date') else str(i) for i, d in enumerate(prices.index)]
        
        return {
            'dates': dates,
            'prices': prices.tolist(),
            'ma20': [None if pd.isna(x) else x for x in ma20.tolist()],
            'ma50': [None if pd.isna(x) else x for x in ma50.tolist()],
        }
    
    def get_returns_distribution(self):
        """
        Get return distribution data
        """
        returns_pct = self.returns * 100
        
        # Histogram data
        hist, bin_edges = np.histogram(returns_pct, bins=30)
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean': float(returns_pct.mean()),
            'median': float(returns_pct.median()),
            'std': float(returns_pct.std()),
            'skew': float(stats.skew(returns_pct)),
            'positive_days_pct': float((returns_pct > 0).mean() * 100),
            'max_gain': float(returns_pct.max()),
            'max_loss': float(returns_pct.min()),
        }


def forecast_stock(prices, dates=None, forecast_days=30):
    """
    Main function to get complete forecast for a stock
    """
    forecaster = StockForecaster(prices, dates)
    
    result = forecaster.generate_recommendation(forecast_days)
    result['chart_data'] = forecaster.get_price_history_chart_data()
    result['returns_distribution'] = forecaster.get_returns_distribution()
    
    return result
