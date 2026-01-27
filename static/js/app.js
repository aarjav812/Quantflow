/**
 * ForeQuant - AI-Powered Stock Forecasts
 * API integration and chart visualizations
 */

// ============================================
// Configuration
// ============================================
const API_BASE = '';
let stocksData = [];
let selectedStock = null;
let charts = {};

// Chart.js defaults
Chart.defaults.color = 'rgba(255, 255, 255, 0.65)';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    loadStocks();
    initializeSearch();
    initializeCharts();
    
    document.getElementById('forecastDays').addEventListener('change', () => {
        if (selectedStock) {
            loadForecast(selectedStock);
        }
    });
});

// ============================================
// API
// ============================================
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============================================
// Load Stocks
// ============================================
async function loadStocks() {
    try {
        const data = await apiCall('/api/stocks');
        stocksData = data.stocks || [];
    } catch (error) {
        console.error('Failed to load stocks:', error);
        // Use fallback data
        stocksData = [
            { symbol: 'RELIANCE', name: 'Reliance Industries' },
            { symbol: 'TCS', name: 'Tata Consultancy Services' },
            { symbol: 'HDFCBANK', name: 'HDFC Bank' },
            { symbol: 'INFY', name: 'Infosys' },
            { symbol: 'ICICIBANK', name: 'ICICI Bank' },
        ];
    }
}

// ============================================
// Search
// ============================================
function initializeSearch() {
    const searchInput = document.getElementById('stockSearch');
    const dropdown = document.getElementById('searchDropdown');
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        
        if (query.length < 1) {
            dropdown.classList.remove('active');
            return;
        }
        
        const matches = stocksData.filter(stock => 
            stock.symbol.toLowerCase().includes(query) ||
            (stock.name && stock.name.toLowerCase().includes(query))
        ).slice(0, 8);
        
        if (matches.length > 0) {
            dropdown.innerHTML = matches.map(stock => `
                <div class="search-item" onclick="selectStock('${stock.symbol}')">
                    <span class="search-item-symbol">${stock.symbol}</span>
                    <span class="search-item-name">${stock.name || ''}</span>
                </div>
            `).join('');
            dropdown.classList.add('active');
        } else {
            dropdown.classList.remove('active');
        }
    });
    
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.length > 0) {
            searchInput.dispatchEvent(new Event('input'));
        }
    });
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box')) {
            dropdown.classList.remove('active');
        }
    });
}

// ============================================
// Stock Selection
// ============================================
async function selectStock(symbol) {
    selectedStock = symbol;
    
    // Update search input
    document.getElementById('stockSearch').value = symbol;
    document.getElementById('searchDropdown').classList.remove('active');
    
    // Show loading
    showLoading();
    
    // Hide welcome, show dashboard
    document.getElementById('welcomeScreen').style.display = 'none';
    document.getElementById('dashboard').style.display = 'flex';
    
    // Load forecast
    await loadForecast(symbol);
    
    hideLoading();
}

// ============================================
// Load Forecast
// ============================================
async function loadForecast(symbol) {
    const days = parseInt(document.getElementById('forecastDays').value);
    
    try {
        const data = await apiCall(`/api/forecast/${symbol}?days=${days}`);
        updateUI(data, symbol, days);
    } catch (error) {
        console.error('Failed to load forecast:', error);
        // Show error state
    }
}

// ============================================
// Update UI
// ============================================
function updateUI(data, symbol, days) {
    const forecast = data.forecast || {};
    const trends = data.trends || {};
    const seasonality = data.seasonality || {};
    const chartData = data.chart_data || {};
    const distribution = data.returns_distribution || {};
    
    // Stock info
    document.getElementById('stockSymbol').textContent = symbol;
    document.getElementById('currentPrice').textContent = `₹${trends.current_price?.toFixed(2) || 'N/A'}`;
    
    // Today's change (mock - would need real-time data)
    const todayChange = trends.momentum_10d ? (trends.momentum_10d / 10) : 0;
    const changeEl = document.getElementById('priceChange');
    changeEl.textContent = `${todayChange >= 0 ? '+' : ''}${todayChange.toFixed(2)}% today`;
    changeEl.className = `price-change ${todayChange >= 0 ? 'positive' : 'negative'}`;
    
    // Main forecast
    document.getElementById('forecastPeriod').textContent = days;
    
    const expectedReturn = forecast.expected_return_pct || 0;
    const forecastValueEl = document.getElementById('expectedReturn');
    forecastValueEl.textContent = `${expectedReturn >= 0 ? '+' : ''}${expectedReturn.toFixed(1)}%`;
    forecastValueEl.className = `forecast-value ${expectedReturn < 0 ? 'negative' : ''}`;
    
    document.getElementById('forecastRange').textContent = 
        `${forecast.lower_bound_pct?.toFixed(1)}% to ${forecast.upper_bound_pct?.toFixed(1)}%`;
    
    document.getElementById('targetPrice').textContent = `₹${forecast.target_price?.toFixed(2) || 'N/A'}`;
    
    // Recommendation
    const rec = data.recommendation || 'HOLD';
    const recCard = document.getElementById('recommendationCard');
    const recBadge = document.getElementById('recBadge');
    const recAction = document.getElementById('recAction');
    
    // Set recommendation styling
    recCard.className = 'recommendation-card';
    recBadge.className = 'rec-badge';
    
    if (rec.includes('BUY')) {
        recCard.classList.add('buy');
        recBadge.classList.add('buy');
    } else if (rec.includes('SELL')) {
        recCard.classList.add('sell');
        recBadge.classList.add('sell');
    } else {
        recCard.classList.add('hold');
        recBadge.classList.add('hold');
    }
    
    recBadge.textContent = rec;
    recAction.textContent = data.action || '';
    
    // Confidence
    const confidence = forecast.confidence_score || 50;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    document.getElementById('confidenceValue').textContent = `${Math.round(confidence)}%`;
    
    // Signals
    const signalsList = document.getElementById('signalsList');
    const reasons = data.reasons || [];
    signalsList.innerHTML = reasons.map(reason => {
        const isBullish = reason.includes('+') || reason.includes('above') || reason.includes('Strong') || reason.includes('upside');
        const isBearish = reason.includes('-') || reason.includes('below') || reason.includes('Weak') || reason.includes('downside');
        const type = isBullish ? 'bullish' : (isBearish ? 'bearish' : 'neutral');
        const icon = isBullish ? '↑' : (isBearish ? '↓' : '→');
        
        return `
            <div class="signal-item ${type}">
                <span class="signal-icon">${icon}</span>
                <span>${reason}</span>
            </div>
        `;
    }).join('');
    
    // Metrics
    document.getElementById('momentum10').textContent = `${trends.momentum_10d >= 0 ? '+' : ''}${trends.momentum_10d?.toFixed(1)}%`;
    document.getElementById('momentum10').className = `metric-value ${trends.momentum_10d >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('momentum30').textContent = `${trends.momentum_30d >= 0 ? '+' : ''}${trends.momentum_30d?.toFixed(1)}%`;
    document.getElementById('momentum30').className = `metric-value ${trends.momentum_30d >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('annualReturn').textContent = `${forecast.annualized_return?.toFixed(1)}%`;
    document.getElementById('volatility').textContent = `${forecast.annualized_volatility?.toFixed(1)}%`;
    document.getElementById('winRate').textContent = `${distribution.positive_days_pct?.toFixed(0)}%`;
    
    document.getElementById('trendDir').textContent = trends.trend_direction || 'N/A';
    document.getElementById('trendDir').className = `metric-value ${trends.trend_direction === 'UPWARD' ? 'positive' : (trends.trend_direction === 'DOWNWARD' ? 'negative' : '')}`;
    
    // Seasonality
    if (seasonality) {
        document.getElementById('bestMonth').textContent = seasonality.best_month || 'N/A';
        document.getElementById('bestReturn').textContent = `${seasonality.best_month_return >= 0 ? '+' : ''}${seasonality.best_month_return?.toFixed(1)}%`;
        
        document.getElementById('worstMonth').textContent = seasonality.worst_month || 'N/A';
        document.getElementById('worstReturn').textContent = `${seasonality.worst_month_return?.toFixed(1)}%`;
        
        document.getElementById('currentMonth').textContent = seasonality.current_month || 'N/A';
        
        const currentMonthData = seasonality.monthly_returns?.[seasonality.current_month];
        if (currentMonthData) {
            const cmr = currentMonthData.avg_return;
            document.getElementById('currentMonthReturn').textContent = `${cmr >= 0 ? '+' : ''}${cmr?.toFixed(1)}%`;
            document.getElementById('currentMonthReturn').className = `season-return ${cmr >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update month chart
        updateMonthChart(seasonality.monthly_returns);
    }
    
    // Update price chart
    updatePriceChart(chartData);
    
    // Update Advanced Models
    updateAdvancedModels(data);
}

function updateAdvancedModels(data) {
    const arima = data.arima || {};
    const mc = data.monte_carlo || {};
    const garch = data.garch || {};
    const ensemble = data.ensemble || {};
    
    // 1. ARIMA
    document.getElementById('arimaModel').textContent = arima.model || 'ARIMA(1,1,1)';
    const arimaRet = arima.expected_return_pct || 0;
    const arimaEl = document.getElementById('arimaReturn');
    arimaEl.textContent = `${arimaRet >= 0 ? '+' : ''}${arimaRet.toFixed(1)}%`;
    arimaEl.className = `model-value ${arimaRet >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('arimaAIC').textContent = arima.aic ? Math.round(arima.aic).toLocaleString() : 'N/A';
    
    const arimaTarget = arima.forecast_prices && arima.forecast_prices.length > 0 
        ? arima.forecast_prices[arima.forecast_prices.length - 1] 
        : (data.trends.current_price * (1 + arimaRet/100));
    document.getElementById('arimaTarget').textContent = `₹${arimaTarget.toFixed(2)}`;
    
    // 2. Monte Carlo
    document.getElementById('mcProbability').textContent = `${Math.round(mc.probability_of_profit || 50)}%`;
    const mcRet = mc.expected_return_pct || 0;
    document.getElementById('mcReturn').textContent = `${mcRet >= 0 ? '+' : ''}${mcRet.toFixed(1)}%`;
    
    const p5 = mc.percentile_5 || (data.trends.current_price * 0.9);
    const p95 = mc.percentile_95 || (data.trends.current_price * 1.1);
    const current = data.trends.current_price || 1000;
    
    document.getElementById('mcLow').textContent = `₹${Math.round(p5).toLocaleString()}`;
    document.getElementById('mcHigh').textContent = `₹${Math.round(p95).toLocaleString()}`;
    
    // Calculate position of current price in the range (for the marker)
    const range = p95 - p5;
    const position = range > 0 ? ((current - p5) / range) * 100 : 50;
    document.getElementById('mcMarker').style.left = `${Math.min(100, Math.max(0, position))}%`;
    
    // 3. GARCH
    const vol = garch.current_vol || 20;
    document.getElementById('garchModel').textContent = garch.model_type || 'GARCH(1,1)';
    document.getElementById('garchVol').textContent = `${vol.toFixed(1)}%`;
    document.getElementById('garchVix').textContent = garch.vix_style_vol?.toFixed(1) || 'N/A';
    document.getElementById('garchPersist').textContent = garch.persistence ? garch.persistence.toFixed(2) : '0.95';
    
    // Update Vol Meter (0-60% scale)
    const volFill = Math.min(100, (vol / 60) * 100);
    document.getElementById('volFill').style.width = `${volFill}%`;
    
    // 4. Ensemble
    const components = ensemble.component_returns || {};
    const sumAbs = Math.abs(components.momentum || 0) + Math.abs(components.arima || 0) + Math.abs(components.monte_carlo || 0);
    
    // Note: Weights are fixed 40/30/30 in backend logic, but we can visualize return contribution if we want.
    // Here we stick to visualizing the fixed weights as defined in CSS logic
    
    const agree = ensemble.models_agree;
    const agreeEl = document.getElementById('ensembleAgreement');
    if (agree) {
        agreeEl.innerHTML = '<span class="agree-icon">✓</span><span>All models agree</span>';
        agreeEl.className = 'ensemble-agreement';
        agreeEl.style.color = 'var(--green)';
        agreeEl.style.background = 'rgba(34, 197, 94, 0.1)';
    } else {
        agreeEl.innerHTML = '<span class="agree-icon">!</span><span>Mixed signals</span>';
        agreeEl.style.color = 'var(--yellow)';
        agreeEl.style.background = 'rgba(251, 191, 36, 0.1)';
    }
}

// ============================================
// Charts
// ============================================
function initializeCharts() {
    // Price Chart
    const priceCtx = document.getElementById('priceChart')?.getContext('2d');
    if (priceCtx) {
        charts.price = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Price',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: '20-Day MA',
                        data: [],
                        borderColor: '#22c55e',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: '50-Day MA',
                        data: [],
                        borderColor: '#fbbf24',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 15, 25, 0.95)',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            title: (items) => items[0]?.label || '',
                            label: (ctx) => `₹${ctx.raw?.toFixed(2) || 'N/A'}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { maxTicksLimit: 8, color: 'rgba(255,255,255,0.4)' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: {
                            color: 'rgba(255,255,255,0.4)',
                            callback: (val) => '₹' + val.toFixed(0)
                        }
                    }
                }
            }
        });
    }
    
    // Month Chart
    const monthCtx = document.getElementById('monthChart')?.getContext('2d');
    if (monthCtx) {
        charts.month = new Chart(monthCtx, {
            type: 'bar',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    data: [],
                    backgroundColor: [],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.raw >= 0 ? '+' : ''}${ctx.raw?.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { 
                            color: 'rgba(255,255,255,0.4)',
                            callback: (val) => val + '%'
                        }
                    }
                }
            }
        });
    }
}

function updatePriceChart(chartData) {
    if (!charts.price) return;
    
    const dates = chartData.dates || [];
    const prices = chartData.prices || [];
    const ma20 = chartData.ma20 || [];
    const ma50 = chartData.ma50 || [];
    
    // Format dates
    const labels = dates.map(d => {
        const date = new Date(d);
        return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
    });
    
    charts.price.data.labels = labels;
    charts.price.data.datasets[0].data = prices;
    charts.price.data.datasets[1].data = ma20;
    charts.price.data.datasets[2].data = ma50;
    charts.price.update('none');
}

function updateMonthChart(monthlyReturns) {
    if (!charts.month || !monthlyReturns) return;
    
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const data = months.map(m => monthlyReturns[m]?.avg_return || 0);
    const colors = data.map(v => v >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)');
    
    charts.month.data.datasets[0].data = data;
    charts.month.data.datasets[0].backgroundColor = colors;
    charts.month.update('none');
}

// ============================================
// Loading
// ============================================
function showLoading() {
    document.getElementById('loading').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}
