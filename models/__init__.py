"""
Models Package Initialization
"""
from .arima import fit_arima, rolling_forecast, check_stationarity
from .garch import fit_garch, fit_egarch, volatility_term_structure, realized_vs_implied
from .factors import construct_factors, factor_regression, factor_performance, factor_correlation
from .portfolio import (
    optimize_portfolio, 
    minimum_variance_portfolio, 
    efficient_frontier,
    risk_parity_portfolio,
    monte_carlo_simulation,
    portfolio_analytics
)
from .risk import (
    value_at_risk,
    conditional_var,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    comprehensive_risk_analysis,
    rolling_risk_metrics
)
from .backtest import BacktestEngine, run_strategy_backtest

__all__ = [
    # ARIMA
    'fit_arima', 'rolling_forecast', 'check_stationarity',
    # GARCH
    'fit_garch', 'fit_egarch', 'volatility_term_structure', 'realized_vs_implied',
    # Factors
    'construct_factors', 'factor_regression', 'factor_performance', 'factor_correlation',
    # Portfolio
    'optimize_portfolio', 'minimum_variance_portfolio', 'efficient_frontier',
    'risk_parity_portfolio', 'monte_carlo_simulation', 'portfolio_analytics',
    # Risk
    'value_at_risk', 'conditional_var', 'sharpe_ratio', 'sortino_ratio',
    'max_drawdown', 'calmar_ratio', 'comprehensive_risk_analysis', 'rolling_risk_metrics',
    # Backtest
    'BacktestEngine', 'run_strategy_backtest'
]
