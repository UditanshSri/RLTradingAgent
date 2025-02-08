# RLTradingAgent Readme

<h1 align="center">RLTradingAgent</h1>
<p align="center">
</p>
<a href="https://weekendofcode.computercodingclub.in/"> <img src="https://i.postimg.cc/njCM24kx/woc.jpg" height=30px> </a>

## Introduction:
This project implements a Reinforcement Learning (RL) agent for stock trading using the FinRL library. The agent interacts with a custom trading environment based on OpenAI Gym, making buy, sell, or hold decisions to maximize portfolio returns. The core RL algorithm used is **Proximal Policy Optimization (PPO)** from Stable Baselines3.

---

## Table of Contents:
- [Methodology](#methodology)
- [Tools and Libraries](#tools-and-libraries)
- [Data Collection](#data-collection)
- [Reward Function](#reward-function)
- [RL Model Implementation](#rl-model-implementation)
- [Results & Evaluation](#results--evaluation)
- [Conclusion](#conclusion)
- [References](#references)

## Technology Stack:
1) Python 3  
2) FinRL  
3) Stable Baselines3  
4) OpenAI Gym  
5) Yahoo Finance API  
6) NumPy, Pandas, Matplotlib, Seaborn

---

## Methodology

### Environment
- **Custom Stock Trading Environment**: Extends `StockTradingEnv` from the FinRL library.
- **Market Simulation**: The environment mimics real-world trading, handling transactions, stock holdings, and portfolio management.

### Actions
Actions involve placing trades such as buying, selling, or holding stocks.
- **Transaction Costs**: 0.1% per transaction.
- **Turbulence Threshold**: Restricts excessive trading in volatile markets.

### Technical Indicators Used
- **Volume**  
- **MACD**  
- **Bollinger Bands**  
- **RSI**  
- **CCI**  
- **DX**  
- **SMA (30 & 60 periods)**  
- **Turbulence**

---

## Tools and Libraries

- **Python 3**  
- **Kaggle Notebooks**  
- **NumPy & Pandas**  
- **Matplotlib & Seaborn**  
- **Yahoo Finance API (`yfinance`)**  
- **FinRL**  
- **Stable Baselines3 (PPO)**

---

## Data Collection
Stock data is fetched using `YahooDownloader` from FinRL. Data is merged for benchmark comparison to analyze relative performance.

---

## Reward Function

### Custom Risk-Adjusted Reward Function
The reward function balances profitability and risk using weighted metrics:
- **Annualized Return**
- **Downside Standard Deviation (Risk Penalty)**
- **Treynor Ratio**
- **Differential Return vs. Benchmark**

The reward function penalizes excessive downside risk and improves risk-adjusted returns by optimizing tunable weights.

#### Key Formulas:

1. **Annualized Return**  
   ```
   R_annual = (1 + R_daily)^252 - 1
   ```
   Where:
   - R_annual = Annualized return
   - R_daily = Daily return
   - 252 = Number of trading days in a year

2. **Sharpe Ratio**  
   ```
   S = (R_p - R_f) / σ_p
   ```
   Where:
   - R_p = Portfolio return
   - R_f = Risk-free rate
   - σ_p = Portfolio standard deviation

3. **Sortino Ratio**  
   ```
   S_sortino = (R_p - R_f) / σ_d
   ```
   Where:
   - R_p = Portfolio return
   - R_f = Risk-free rate
   - σ_d = Downside deviation (standard deviation of negative returns only)

4. **Treynor Ratio**  
   ```
   T = (R_p - R_f) / β_p
   ```
   Where:
   - R_p = Portfolio return
   - R_f = Risk-free rate
   - β_p = Portfolio beta (systematic risk)

5. **Differential Return vs. Benchmark**  
   ```
   DR = R_p - R_b
   ```
   Where:
   - R_p = Portfolio return
   - R_b = Benchmark return

6. **Final Weighted Reward Function**  
   ```
   Reward = w₁ · R_annual - w₂ · σ_d + w₃ · T + w₄ · DR
   ```
   Where:
   - w₁ = Weight for annualized return
   - w₂ = Weight for downside risk penalty
   - w₃ = Weight for Treynor ratio
   - w₄ = Weight for differential return
   - σ_d = Downside deviation
   - T = Treynor ratio
   - DR = Differential return

### Implementation Notes:
- Weights (w₁, w₂, w₃, w₄) are hyperparameters that can be tuned
- Downside deviation only considers returns below the target return
- Beta (β) is calculated using regression against the market benchmark
- All returns are calculated on a risk-adjusted basis


---

## RL Model Implementation

### PPO Algorithm
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Library**: Stable Baselines3
- **Vectorized Environment** is used for training efficiency.
- **Model Training** is performed over a large number of timesteps to optimize portfolio returns.

---

## Results & Evaluation

### Key Evaluation Metrics
- **Sharpe Ratio**: Measures return relative to risk.
- **Sortino Ratio**: Focuses on downside risk.
- **Treynor Ratio**: Accounts for systematic risk using beta.

### Performance Visualization
Performance evaluation includes stock price trends and portfolio value tracking over time.

---

## Contributors:

Team Name: CJYTHON

* [Uditansh Srivastava](https://github.com/Uditansh-Srivastava)  
* [Shivam Aryan](https://github.com/Aryan10)  
* [Shaurya Pratap Singh](https://github.com/shauryasf)
* * [Samudraneel Sarkar](https://github.com/samudraneel05)

### Made at:
<a href="https://weekendofcode.computercodingclub.in/"> <img src="https://i.postimg.cc/mrCCnTbN/tpg.jpg" height=30px> </a>

---

## Conclusion
- The RL agent successfully learns trading strategies based on historical stock data.
- The custom reward function improves risk-adjusted returns.
- Future improvements could include testing different RL algorithms (e.g., DDPG) and multi-stock trading strategies.

---

## References
1. [FinRL Documentation](https://github.com/AI4Finance-Foundation/FinRL)  
2. [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)  
3. [Differential Return Paper](https://www.researchgate.net/publication/356127405)
