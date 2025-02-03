# RLTradingAgent

## Overview
This project implements a Reinforcement Learning (RL) agent for stock trading using the FinRL library. The agent interacts with a custom trading environment based on OpenAI Gym, making buy, sell, or hold decisions to maximize portfolio returns. The core RL algorithm used is **Proximal Policy Optimization (PPO)** from Stable Baselines3.

---

## Methodology

### Environment
- **Custom Stock Trading Environment**: Extends `StockTradingEnv` from the FinRL library.
- **Market Simulation**: The environment mimics real-world trading, handling transactions, stock holdings, and portfolio management.

### Actions
Actions involve placing trades such as buying, selling, or holding stocks. More formally:

- **Action Space**:
  - The action is a continuous value `𝒜 ∈ [-1,1]`, scaled by `hmax` (the max number of stocks the agent can trade at a time):
    
    ```
    hmax = ⌊ initial_amount / max_price ⌋
    ```
  - The action space is defined as:
    
    ```
    A = stock_dim * 𝒜
    ```
- **Transaction Costs**:
  - Buy/Sell costs: **0.1%** per transaction.
- **Turbulence Threshold**:
  - Restricts excessive trading in volatile markets.

### State Space

```python
state_space = 1 + 2 * stock_dim + len(indicators) * stock_dim
```

- `1`: Portfolio value.
- `2 * stock_dim`: Stock holdings and prices.
- `len(indicators) * stock_dim`: Technical indicators per stock.

### Technical Indicators Used
- **Volume**: Total number of shares traded.
- **MACD (Moving Average Convergence Divergence)**: Trend-following indicator.
- **Bollinger Bands (Upper & Lower)**: Identify overbought/oversold conditions.
- **RSI (Relative Strength Index, 30-period)**: Measures momentum.
- **CCI (Commodity Channel Index, 30-period)**: Identifies trends.
- **DX (Directional Movement Index, 30-period)**: Measures trend strength.
- **SMA (Simple Moving Averages, 30 & 60 periods)**: Smooths price fluctuations.
- **Turbulence**: Market volatility measure.

---

## Tools and Libraries

- **Python 3**
- **Kaggle Notebooks
- **NumPy & Pandas**: Data manipulation and processing.
- **Matplotlib & Seaborn**: Visualization.
- **Yahoo Finance API (`yfinance`)**: Stock data fetching.
- **FinRL**: RL environment and preprocessing tools.
- **Stable Baselines3 (PPO)**: RL agent implementation.

---

## Data Collection

Stock data is fetched using `YahooDownloader` from FinRL:
```python
df_stock = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[ticker]).fetch_data()
df_benchmark = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[benchmark_ticker]).fetch_data()
```

Data is merged for benchmark comparison:
```python
df = pd.merge(df_stock, df_benchmark[['date', 'close']], on='date', suffixes=('', '_benchmark'))
```

---

## Reward Functions

### Profit & Loss (PnL)
```python
Reward = Current Portfolio Value – Previous Portfolio Value
```
- Measures daily profit/loss.
- Prone to early model convergence.

### Moving Average of Return
```python
Reward = (1/N) * Σ(Returns over N days)
```
- Reduces reward volatility.

### Custom Reward (Risk-Adjusted)
```python
Reward = α × return_moving_average - β × downside_return
```
- Balances return and risk using hyperparameters `α` and `β`.

### Differential Return
- Uses risk penalty measurement as described in research papers.

---

## RL Model Implementation

### PPO Algorithm
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Library**: Stable Baselines3
- **Vectorized Environment**:
  ```python
  vec_env = make_vec_env(lambda: env, n_envs=1)
  ```
- **Training the Model**:
  ```python
  model = PPO("MlpPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=100000)
  ```
- **Portfolio Value Tracking**:
  ```python
  portfolio_values = []
  while True:
      action, _ = model.predict(obs, deterministic=True)
      obs, rewards, done, info = vec_env.step(action)
      if done[0]: break
      portfolio_values.append(vec_env.envs[0].unwrapped.asset_memory[-1])
  ```

---

## Results & Evaluation

### Key Evaluation Metrics
- **Sharpe Ratio**:
  ```
  Sharpe Ratio = (Average Return - Risk-Free Rate) / Std Dev of Returns
  ```
- **Sortino Ratio** (focuses on downside risk):
  ```
  Sortino Ratio = (Average Return - Risk-Free Rate) / Downside Deviation
  ```
- **Treynor Ratio** (accounts for systematic risk using beta):
  ```
  Treynor Ratio = (Average Return - Risk-Free Rate) / Beta
  ```

### Performance Visualization
Stock price trends are visualized using Matplotlib and Seaborn:
```python
sns.lineplot(x=pd.to_datetime(df['date']), y=df['close'], label='Closing Price', color='blue')
plt.title(f"{ticker} Closing Stock Over Time")
```

Final portfolio performance:
```python
plt.plot(timesteps, portfolio_values, label="Portfolio Value", color='blue')
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value ($)")
plt.title("RL Model Performance")
plt.legend()
plt.grid()
plt.show()
```
## PnL (Profit & Loss)

Simple reward function that measures the total profit/loss or return obtained on day. Leads to issues such as early convergence of model.

Reward = Current Portfolio Value – Previous Portfolio Value


| day: 2515, episode: 0  |
| -- |
| begin_total_asset: 10000.00  |
| end_total_asset: 67467.96  |
| total_reward: 57467.96  |
| total_cost: 907.57  |
| total_trades: 2515  |
| Sharpe: 0.811  |

---

## Conclusion
- The RL agent successfully learns trading strategies based on historical stock data.
- The custom reward function improves risk-adjusted returns compared to naive profit maximization.
- Future improvements could include testing different RL algorithms (e.g., A2C, DDPG) and incorporating more financial indicators.

---

## References
1. FinRL Documentation: https://github.com/AI4Finance-Foundation/FinRL
2. Stable Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
3. Differential Return Paper: https://www.researchgate.net/publication/356127405

---

**Author:** RL Trading Project Team
