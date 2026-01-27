"""
Markowitz Portfolio Optimization
Mean-Variance Optimization with Efficient Frontier
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_portfolio_stats(weights, returns, cov_matrix, risk_free_rate=0.05):
    """
    Calculate portfolio return, volatility, and Sharpe ratio
    """
    portfolio_return = np.dot(weights, returns) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return portfolio_return, portfolio_vol, sharpe_ratio


def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.05):
    """Negative Sharpe ratio for minimization"""
    _, _, sharpe = calculate_portfolio_stats(weights, returns, cov_matrix, risk_free_rate)
    return -sharpe


def portfolio_volatility(weights, returns, cov_matrix):
    """Portfolio volatility for minimization"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))


def optimize_portfolio(returns_df, risk_free_rate=0.05, target_return=None):
    """
    Optimize portfolio using Markowitz Mean-Variance Optimization
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of daily returns for each asset
    risk_free_rate : float
        Annual risk-free rate
    target_return : float or None
        If specified, minimize variance for this target return
        
    Returns:
    --------
    dict with optimal weights and portfolio statistics
    """
    # Calculate expected returns and covariance
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n_assets = len(mean_returns)
    
    # Constraints: weights sum to 1, no short selling
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    if target_return is not None:
        # Minimize variance for target return
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: np.dot(x, mean_returns) * 252 - target_return
        })
        objective = lambda w: portfolio_volatility(w, mean_returns, cov_matrix)
    else:
        # Maximize Sharpe ratio
        objective = lambda w: negative_sharpe(w, mean_returns, cov_matrix, risk_free_rate)
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    port_return, port_vol, sharpe = calculate_portfolio_stats(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )
    
    return {
        'weights': dict(zip(returns_df.columns, optimal_weights.tolist())),
        'expected_return': float(port_return * 100),
        'volatility': float(port_vol * 100),
        'sharpe_ratio': float(sharpe),
        'optimization_success': result.success
    }


def minimum_variance_portfolio(returns_df):
    """
    Calculate the minimum variance portfolio
    """
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n_assets = len(mean_returns)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    objective = lambda w: portfolio_volatility(w, mean_returns, cov_matrix)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    port_return, port_vol, sharpe = calculate_portfolio_stats(
        optimal_weights, mean_returns, cov_matrix
    )
    
    return {
        'weights': dict(zip(returns_df.columns, optimal_weights.tolist())),
        'expected_return': float(port_return * 100),
        'volatility': float(port_vol * 100),
        'sharpe_ratio': float(sharpe),
        'type': 'minimum_variance'
    }


def efficient_frontier(returns_df, n_points=50, risk_free_rate=0.05):
    """
    Calculate the efficient frontier
    
    Returns:
    --------
    dict with frontier points and optimal portfolios
    """
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Get return range
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []
    
    for target in target_returns:
        try:
            result = optimize_portfolio(returns_df, risk_free_rate, target_return=target)
            if result['optimization_success']:
                frontier_returns.append(result['expected_return'])
                frontier_volatility.append(result['volatility'])
                frontier_weights.append(result['weights'])
        except:
            continue
    
    # Find tangency portfolio (max Sharpe)
    max_sharpe_result = optimize_portfolio(returns_df, risk_free_rate)
    
    # Find minimum variance portfolio
    min_var_result = minimum_variance_portfolio(returns_df)
    
    return {
        'frontier_returns': frontier_returns,
        'frontier_volatility': frontier_volatility,
        'frontier_weights': frontier_weights,
        'max_sharpe_portfolio': max_sharpe_result,
        'min_variance_portfolio': min_var_result,
        'capital_market_line': {
            'risk_free_rate': risk_free_rate * 100,
            'slope': max_sharpe_result['sharpe_ratio'],
            'tangency_return': max_sharpe_result['expected_return'],
            'tangency_volatility': max_sharpe_result['volatility']
        }
    }


def risk_parity_portfolio(returns_df):
    """
    Calculate risk parity portfolio (equal risk contribution)
    
    Each asset contributes equally to portfolio risk
    """
    cov_matrix = returns_df.cov().values * 252
    n_assets = len(returns_df.columns)
    
    def risk_budget_objective(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        # Target: equal risk contribution
        target_risk = portfolio_vol / n_assets
        return np.sum((risk_contrib - target_risk) ** 2)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0.01, 1) for _ in range(n_assets))
    
    initial_weights = np.array([1/n_assets] * n_assets)
    result = minimize(risk_budget_objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    weights = result.x
    mean_returns = returns_df.mean()
    
    port_return, port_vol, sharpe = calculate_portfolio_stats(
        weights, mean_returns, returns_df.cov()
    )
    
    # Calculate risk contributions
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    risk_contrib = weights * marginal_contrib / portfolio_vol * 100
    
    return {
        'weights': dict(zip(returns_df.columns, weights.tolist())),
        'risk_contributions': dict(zip(returns_df.columns, risk_contrib.tolist())),
        'expected_return': float(port_return * 100),
        'volatility': float(port_vol * 100),
        'sharpe_ratio': float(sharpe),
        'type': 'risk_parity'
    }


def monte_carlo_simulation(returns_df, n_simulations=10000):
    """
    Monte Carlo simulation for portfolio analysis
    
    Generates random portfolios to visualize risk-return tradeoffs
    """
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n_assets = len(mean_returns)
    
    results = {
        'returns': [],
        'volatility': [],
        'sharpe': [],
        'weights': []
    }
    
    for _ in range(n_simulations):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        port_return, port_vol, sharpe = calculate_portfolio_stats(
            weights, mean_returns, cov_matrix
        )
        
        results['returns'].append(port_return * 100)
        results['volatility'].append(port_vol * 100)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights.tolist())
    
    # Find best portfolios
    sharpe_array = np.array(results['sharpe'])
    vol_array = np.array(results['volatility'])
    
    max_sharpe_idx = sharpe_array.argmax()
    min_vol_idx = vol_array.argmin()
    
    return {
        'simulation_returns': results['returns'],
        'simulation_volatility': results['volatility'],
        'simulation_sharpe': results['sharpe'],
        'best_sharpe': {
            'return': results['returns'][max_sharpe_idx],
            'volatility': results['volatility'][max_sharpe_idx],
            'sharpe': results['sharpe'][max_sharpe_idx],
            'weights': dict(zip(returns_df.columns, results['weights'][max_sharpe_idx]))
        },
        'best_min_vol': {
            'return': results['returns'][min_vol_idx],
            'volatility': results['volatility'][min_vol_idx],
            'sharpe': results['sharpe'][min_vol_idx],
            'weights': dict(zip(returns_df.columns, results['weights'][min_vol_idx]))
        },
        'n_simulations': n_simulations
    }


def portfolio_analytics(weights, returns_df, risk_free_rate=0.05):
    """
    Comprehensive portfolio analytics
    """
    weights_array = np.array([weights.get(col, 0) for col in returns_df.columns])
    
    # Portfolio returns series
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    
    # Basic stats
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
    sortino = (ann_return - risk_free_rate) / downside_std
    
    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # VaR and CVaR
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    # Skewness and Kurtosis
    skew = stats.skew(portfolio_returns)
    kurt = stats.kurtosis(portfolio_returns)
    
    return {
        'annual_return': float(ann_return * 100),
        'annual_volatility': float(ann_vol * 100),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd * 100),
        'calmar_ratio': float(calmar),
        'var_95': float(var_95 * 100),
        'cvar_95': float(cvar_95 * 100),
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'positive_days': float((portfolio_returns > 0).mean() * 100),
        'best_day': float(portfolio_returns.max() * 100),
        'worst_day': float(portfolio_returns.min() * 100)
    }
