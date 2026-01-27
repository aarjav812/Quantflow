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

# Import ARIMA fitting function
from models.arima import fit_arima


# Import GARCH fitting function
from models.garch import fit_garch

class StockForecaster:
    """
    Forecasting engine for stock prices
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
        
    def run_garch_forecast(self, days=30):
        """
        Run GARCH model for volatility forecasting
        """
        try:
            # Fit GARCH(1,1)
            result = fit_garch(self.prices, p=1, q=1, forecast_horizon=days)
            return result
        except Exception as e:
            # Fallback
            hist_vol = float(self.returns.std() * np.sqrt(252))
            return {
                'error': str(e),
                'model_type': 'Historical Volatility',
                'current_vol': hist_vol,
                'forecast_volatility': [hist_vol] * days,
                'vix_style_vol': hist_vol * 100
            }
        
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
    
    def run_monte_carlo(self, days=30, n_simulations=1000):
        """
        Monte Carlo simulation for price path prediction
        Uses Geometric Brownian Motion (GBM): dS = μS*dt + σS*dW
        
        Returns simulated price paths and probability distribution
        """
        if len(self.returns) < 20:
            return None
        
        # Parameters from historical data
        mu = self.returns.mean()  # Daily drift
        sigma = self.returns.std()  # Daily volatility
        current_price = float(self.prices.iloc[-1])
        
        # Simulate paths
        np.random.seed(42)  # Reproducibility
        dt = 1  # Daily steps
        
        # Generate random walks
        random_shocks = np.random.normal(0, 1, (n_simulations, days))
        
        # GBM formula: S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
        daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
        
        # Cumulative returns
        price_paths = np.zeros((n_simulations, days + 1))
        price_paths[:, 0] = current_price
        
        for t in range(1, days + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(daily_returns[:, t-1])
        
        # Final prices (end of forecast period)
        final_prices = price_paths[:, -1]
        
        # Statistics
        mean_final = float(np.mean(final_prices))
        median_final = float(np.median(final_prices))
        std_final = float(np.std(final_prices))
        
        # Percentiles for confidence intervals
        p5 = float(np.percentile(final_prices, 5))
        p25 = float(np.percentile(final_prices, 25))
        p75 = float(np.percentile(final_prices, 75))
        p95 = float(np.percentile(final_prices, 95))
        
        # Probability of profit
        prob_profit = float(np.mean(final_prices > current_price) * 100)
        
        # Expected return
        expected_return = (mean_final / current_price - 1) * 100
        
        # Sample paths for visualization (5 representative paths)
        sample_indices = [0, n_simulations//4, n_simulations//2, 3*n_simulations//4, n_simulations-1]
        sample_paths = [price_paths[i, :].tolist() for i in sample_indices]
        
        return {
            'n_simulations': n_simulations,
            'forecast_days': days,
            'current_price': current_price,
            'mean_final_price': mean_final,
            'median_final_price': median_final,
            'std_final_price': std_final,
            'percentile_5': p5,
            'percentile_25': p25,
            'percentile_75': p75,
            'percentile_95': p95,
            'probability_of_profit': prob_profit,
            'expected_return_pct': expected_return,
            'sample_paths': sample_paths,
            'model': 'Geometric Brownian Motion',
            'parameters': {
                'daily_drift': float(mu * 100),
                'daily_volatility': float(sigma * 100),
            }
        }
    
    def run_arima_forecast(self, days=30):
        """
        Run ARIMA model for statistical price forecasting
        Uses fixed order (1,1,1) for speed - auto-selection is too slow
        """
        try:
            # Use fixed order for speed (grid search is too slow)
            result = fit_arima(self.prices, order=(1, 1, 1), forecast_days=days)
            
            # Add additional context
            current_price = float(self.prices.iloc[-1])
            final_forecast = result['forecast_prices'][-1] if result['forecast_prices'] else current_price
            
            result['expected_return_pct'] = (final_forecast / current_price - 1) * 100
            result['model'] = f"ARIMA{result['order']}"
            
            return result
        except Exception as e:
            # Fallback if ARIMA fails - use momentum
            avg_return = float(self.returns.mean() * days * 100)
            return {
                'error': str(e),
                'model': 'ARIMA (fallback)',
                'forecast_prices': [],
                'expected_return_pct': avg_return
            }


def forecast_stock(prices, dates=None, forecast_days=30):
    """
    Main function to get complete forecast for a stock
    
    Uses ensemble of 3 models:
    1. Statistical/Momentum model
    2. ARIMA time series model
    3. Monte Carlo simulation
    """
    forecaster = StockForecaster(prices, dates)
    
    # Base recommendation (momentum + trends)
    result = forecaster.generate_recommendation(forecast_days)
    result['chart_data'] = forecaster.get_price_history_chart_data()
    result['returns_distribution'] = forecaster.get_returns_distribution()
    
    # ARIMA forecast
    arima_result = forecaster.run_arima_forecast(forecast_days)
    result['arima'] = arima_result
    
    # Monte Carlo simulation
    monte_carlo_result = forecaster.run_monte_carlo(forecast_days)
    result['monte_carlo'] = monte_carlo_result
    
    # GARCH Volatility forecast
    garch_result = forecaster.run_garch_forecast(forecast_days)
    result['garch'] = garch_result
    
    # Ensemble prediction (weighted average)
    momentum_return = result['forecast']['expected_return_pct']
    arima_return = arima_result.get('expected_return_pct', momentum_return)
    mc_return = monte_carlo_result['expected_return_pct'] if monte_carlo_result else momentum_return
    
    # Weighted ensemble: 40% Momentum, 30% ARIMA, 30% Monte Carlo
    ensemble_return = 0.4 * momentum_return + 0.3 * arima_return + 0.3 * mc_return
    
    # Update the main forecast with ensemble
    result['ensemble'] = {
        'expected_return_pct': float(ensemble_return),
        'momentum_weight': 0.4,
        'arima_weight': 0.3,
        'monte_carlo_weight': 0.3,
        'component_returns': {
            'momentum': float(momentum_return),
            'arima': float(arima_return),
            'monte_carlo': float(mc_return)
        }
    }
    
    # Override main forecast return with ensemble
    result['forecast']['expected_return_pct'] = float(ensemble_return)
    
    # Update confidence based on model agreement
    returns_list = [momentum_return, arima_return, mc_return]
    direction_agreement = sum(1 for r in returns_list if r > 0) if ensemble_return > 0 else sum(1 for r in returns_list if r < 0)
    
    # Boost confidence if all models agree
    if direction_agreement == 3:
        result['forecast']['confidence_score'] = min(100, result['forecast']['confidence_score'] + 15)
        result['ensemble']['models_agree'] = True
    else:
        result['ensemble']['models_agree'] = False
    
    return result

