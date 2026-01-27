"""
Professional Risk Metrics
Comprehensive risk analysis for financial assets
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(prices):
    """Convert prices to returns"""
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.pct_change().dropna()


def value_at_risk(returns, confidence=0.95, method='historical', horizon=1):
    """
    Calculate Value at Risk (VaR)
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        Historical returns
    confidence : float
        Confidence level (0.95 = 95%)
    method : str
        'historical', 'parametric', or 'monte_carlo'
    horizon : int
        Holding period in days
        
    Returns:
    --------
    float : VaR as a positive number (loss)
    """
    returns = np.array(returns)
    
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
    
    elif method == 'parametric':
        # Assume normal distribution
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        var = mu + z_score * sigma
    
    elif method == 'monte_carlo':
        # Simulate future returns
        mu = np.mean(returns)
        sigma = np.std(returns)
        simulations = np.random.normal(mu, sigma, 10000)
        var = np.percentile(simulations, (1 - confidence) * 100)
    
    else:
        var = np.percentile(returns, (1 - confidence) * 100)
    
    # Scale for holding period
    var = var * np.sqrt(horizon)
    
    return float(-var)  # Return as positive loss


def conditional_var(returns, confidence=0.95, method='historical'):
    """
    Calculate Conditional VaR (CVaR) / Expected Shortfall
    
    The expected loss given that the loss exceeds VaR
    """
    returns = np.array(returns)
    
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
    
    elif method == 'parametric':
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        cvar = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence)
    
    else:
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
    
    return float(-cvar)  # Return as positive loss


def sharpe_ratio(returns, risk_free_rate=0.05):
    """
    Calculate Sharpe Ratio
    
    (Return - Risk-Free Rate) / Volatility
    """
    returns = np.array(returns)
    
    ann_return = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    rf = risk_free_rate
    
    if ann_vol == 0:
        return 0.0
    
    return float((ann_return - rf) / ann_vol)


def sortino_ratio(returns, risk_free_rate=0.05, target_return=0):
    """
    Calculate Sortino Ratio
    
    Uses downside deviation instead of total volatility
    """
    returns = np.array(returns)
    
    ann_return = np.mean(returns) * 252
    
    # Downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.std(downside_returns) * np.sqrt(252)
    
    if downside_std == 0:
        return float('inf')
    
    return float((ann_return - risk_free_rate) / downside_std)


def max_drawdown(prices):
    """
    Calculate Maximum Drawdown
    
    The largest peak-to-trough decline
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    cumulative = prices / prices.iloc[0]
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    max_dd = drawdown.min()
    
    # Find drawdown periods
    end_idx = drawdown.idxmin()
    start_idx = cumulative[:end_idx].idxmax() if end_idx is not None else None
    
    return {
        'max_drawdown': float(max_dd * 100),
        'start_date': str(start_idx) if start_idx is not None else None,
        'end_date': str(end_idx) if end_idx is not None else None,
        'recovery_date': None  # Would need forward looking data
    }


def calmar_ratio(returns, prices):
    """
    Calculate Calmar Ratio
    
    Annualized Return / Max Drawdown
    """
    returns = np.array(returns)
    ann_return = np.mean(returns) * 252
    
    mdd_result = max_drawdown(prices)
    mdd = abs(mdd_result['max_drawdown']) / 100
    
    if mdd == 0:
        return float('inf')
    
    return float(ann_return / mdd)


def information_ratio(returns, benchmark_returns, frequency=252):
    """
    Calculate Information Ratio
    
    Active Return / Tracking Error
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    # Align lengths
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[-min_len:]
    benchmark_returns = benchmark_returns[-min_len:]
    
    active_returns = returns - benchmark_returns
    active_return_ann = np.mean(active_returns) * frequency
    tracking_error = np.std(active_returns) * np.sqrt(frequency)
    
    if tracking_error == 0:
        return 0.0
    
    return float(active_return_ann / tracking_error)


def beta_alpha(returns, benchmark_returns, risk_free_rate=0.05):
    """
    Calculate Beta and Alpha using CAPM regression
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    # Align lengths
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[-min_len:]
    benchmark_returns = benchmark_returns[-min_len:]
    
    # Daily risk-free rate
    rf_daily = risk_free_rate / 252
    
    # Excess returns
    excess_returns = returns - rf_daily
    excess_benchmark = benchmark_returns - rf_daily
    
    # Regression
    if np.std(excess_benchmark) == 0:
        return {'beta': 0, 'alpha': 0}
    
    covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
    variance = np.var(excess_benchmark)
    
    beta = covariance / variance if variance > 0 else 0
    alpha = np.mean(excess_returns) - beta * np.mean(excess_benchmark)
    
    # Annualize alpha
    alpha_annual = alpha * 252
    
    return {
        'beta': float(beta),
        'alpha_daily': float(alpha),
        'alpha_annual': float(alpha_annual * 100),  # As percentage
        'r_squared': float(np.corrcoef(excess_returns, excess_benchmark)[0, 1] ** 2)
    }


def treynor_ratio(returns, benchmark_returns, risk_free_rate=0.05):
    """
    Calculate Treynor Ratio
    
    (Return - Risk-Free Rate) / Beta
    """
    returns = np.array(returns)
    
    ba = beta_alpha(returns, benchmark_returns, risk_free_rate)
    beta = ba['beta']
    
    if beta == 0:
        return float('inf')
    
    ann_return = np.mean(returns) * 252
    
    return float((ann_return - risk_free_rate) / beta)


def omega_ratio(returns, threshold=0):
    """
    Calculate Omega Ratio
    
    Probability-weighted ratio of gains vs losses
    """
    returns = np.array(returns)
    
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    if losses.sum() == 0:
        return float('inf')
    
    return float(gains.sum() / losses.sum())


def tail_ratio(returns, confidence=0.05):
    """
    Calculate Tail Ratio
    
    Right tail (95th percentile) / Left tail (5th percentile)
    """
    returns = np.array(returns)
    
    right_tail = np.percentile(returns, (1 - confidence) * 100)
    left_tail = np.percentile(returns, confidence * 100)
    
    if left_tail == 0:
        return float('inf')
    
    return float(abs(right_tail / left_tail))


def comprehensive_risk_analysis(prices, benchmark_prices=None, risk_free_rate=0.05):
    """
    Comprehensive risk analysis for a single asset
    
    Parameters:
    -----------
    prices : pd.Series or np.array
        Historical prices
    benchmark_prices : pd.Series or np.array, optional
        Benchmark prices for relative metrics
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    dict with all risk metrics
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = calculate_returns(prices)
    
    # Basic statistics
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    
    results = {
        # Return metrics
        'annual_return': float(ann_return * 100),
        'annual_volatility': float(ann_vol * 100),
        'total_return': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
        
        # Risk-adjusted returns
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calmar_ratio(returns, prices),
        'omega_ratio': omega_ratio(returns),
        
        # Downside risk
        'var_95_1d': value_at_risk(returns, 0.95, 'historical') * 100,
        'var_99_1d': value_at_risk(returns, 0.99, 'historical') * 100,
        'var_95_parametric': value_at_risk(returns, 0.95, 'parametric') * 100,
        'cvar_95': conditional_var(returns, 0.95) * 100,
        'cvar_99': conditional_var(returns, 0.99) * 100,
        'max_drawdown': max_drawdown(prices),
        
        # Distribution characteristics
        'skewness': float(stats.skew(returns)),
        'kurtosis': float(stats.kurtosis(returns)),
        'positive_days_pct': float((returns > 0).mean() * 100),
        'best_day': float(returns.max() * 100),
        'worst_day': float(returns.min() * 100),
        
        # Tail metrics
        'tail_ratio': tail_ratio(returns),
    }
    
    # Benchmark-relative metrics
    if benchmark_prices is not None:
        if isinstance(benchmark_prices, np.ndarray):
            benchmark_prices = pd.Series(benchmark_prices)
        
        benchmark_returns = calculate_returns(benchmark_prices)
        
        ba = beta_alpha(returns, benchmark_returns, risk_free_rate)
        results.update({
            'beta': ba['beta'],
            'alpha_annual': ba['alpha_annual'],
            'r_squared': ba['r_squared'],
            'information_ratio': information_ratio(returns, benchmark_returns),
            'treynor_ratio': treynor_ratio(returns, benchmark_returns, risk_free_rate)
        })
    
    return results


def rolling_risk_metrics(prices, window=63, risk_free_rate=0.05):
    """
    Calculate rolling risk metrics over time
    
    Parameters:
    -----------
    prices : pd.Series
        Historical prices
    window : int
        Rolling window in days (default 63 = 3 months)
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    dict with rolling metrics time series
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = calculate_returns(prices)
    
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_return = returns.rolling(window).mean() * 252 * 100
    rolling_sharpe = rolling_return / rolling_vol
    
    # Rolling VaR
    rolling_var = returns.rolling(window).apply(
        lambda x: value_at_risk(x, 0.95, 'historical') * 100
    )
    
    # Rolling max drawdown
    rolling_mdd = prices.rolling(window).apply(
        lambda x: max_drawdown(x)['max_drawdown']
    )
    
    return {
        'dates': returns.index[window-1:].tolist() if hasattr(returns, 'index') else list(range(window-1, len(returns))),
        'rolling_volatility': rolling_vol.dropna().tolist(),
        'rolling_return': rolling_return.dropna().tolist(),
        'rolling_sharpe': rolling_sharpe.dropna().tolist(),
        'rolling_var_95': rolling_var.dropna().tolist(),
        'rolling_max_drawdown': rolling_mdd.dropna().tolist(),
        'window_days': window
    }
