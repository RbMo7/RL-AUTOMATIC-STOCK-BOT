# 🧠 Intelligent Reinforcement Learning Trading Agent

This project is a sophisticated stock trading bot that uses **Reinforcement Learning (RL)** to develop and backtest active trading strategies. The agent is built with **Stable-Baselines3** using the **Proximal Policy Optimization (PPO)** algorithm and interacts with a custom trading environment built with **Gymnasium**. The entire application is wrapped in an interactive web UI created with **Streamlit**.

The agent evolves beyond simple indicators, incorporating concepts like **market trends**, **support/resistance**, and **Fibonacci levels** to make nuanced decisions. Its goal is not just to maximize profit but to **learn from the consequences of its actions**, including taking profits, managing risk, and avoiding passivity.

> *(You can replace the URL above with a screenshot of your running application)*

## ✨ Features

### 🧠 Sophisticated RL Agent
- Utilizes the **PPO** algorithm for stable and effective learning.
- The agent's "brain" is a **Multi-Layer Perceptron (MLP)**.

### 🔍 Advanced Feature Engineering ("The Genius Vision")
- **Standard Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator.
- **Long-Term Trend Analysis**: Learns market context using 50-day and 200-day moving averages (Golden/Death Cross).
- **Market Structure**: Calculates distance to support and resistance levels.
- **Fibonacci Retracement**: Identifies key reversal zones.
- **Seasonality**: Uses the month of the year to detect seasonal behavior.

### 🎯 Intelligent Reward System
- **Consequence-Driven**: Rewards based on the long-term impact of actions.
- **Profit-Taking Bonus**: Extra reward for profitable exits.
- **Risk Management**: Penalizes for holding losses or invalid trades.
- **Cost of Capital**: Penalizes idle capital to encourage action.

### 💻 Interactive Web UI (Streamlit)
- Full control over stock ticker, date ranges, and initial balance.
- **Live Training Dashboard**: See real-time training progress and rewards.
- **Advanced Model Management**:
  - Auto saves/loads agents for each ticker (e.g., `agents/agent_AAPL.zip`).
  - Falls back to a generic model `agents/genius_trader_agent.zip` if needed.
  - Supports continual learning with saved models.
- **Backtesting Reports**:
  - Portfolio vs. Buy-and-Hold charts.
  - Buy/Sell signals on stock price chart.
  - Transaction logs and performance metrics.

### 🧪 Experiment Tracking
- Training logs saved in `tensorboard_logs/`.
- Fully compatible with **TensorBoard** for deep analysis.

## 🛠️ Technology Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://www.farama.org/Gymnasium/)
- **Pandas**, **NumPy**
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)
- **Matplotlib**, **TensorBoard**

## ⚙️ Installation & Setup

**1. Clone the Repository:** Run `git clone <your-repo-url>` and `cd <your-repo-folder>`

**2. Create and Activate Virtual Environment:** Run `python -m venv myenv`, then activate it with `myenv\Scripts\activate` (on **Windows**) or `source myenv/bin/activate` (on **macOS/Linux**)

**3. Install Dependencies:** If you don't have a `requirements.txt`, create one with:

```
streamlit
pandas
numpy
yfinance
pandas-ta
matplotlib
gymnasium
stable-baselines3[extra]
tensorflow
Then run: `pip install -r requirements.txt`
```

## 🚀 How to Use

**1. Run the Streamlit App:**  
With your environment activated, run:  
`streamlit run your_script_name.py`

**2. Configure Training:**  
Use the sidebar to set:

- **Stock Ticker:** e.g., `AAPL`, `SPY`, `NVDA`
- **Agent Mode:**
  - Train new specialist agent
  - Load saved model
  - Continue training
- **Date Ranges:** Define training & testing periods (3+ years recommended for training)
- **Agent Parameters:** Adjust initial balance, training timesteps, transaction fees

**3. Run & Analyze:**  
Click **"🚀 Run Analysis"** and monitor the training. Analyze backtest performance, buy/sell signals, and transaction logs.

## 🔄 Creating a Generic Agent

To create `genius_trader_agent.zip`:

1. Train on a broad ETF like `SPY` for high timesteps (e.g., 1,000,000)  
2. Save the model as `agent_SPY.zip`  
3. Rename it:  
   `mv agents/agent_SPY.zip agents/genius_trader_agent.zip`

## 📂 File Structure

```bash
.
├── agents/
│   ├── agent_AAPL.zip            # Saved model for AAPL
│   └── genius_trader_agent.zip   # Generic fallback model
├── tensorboard_logs/
│   └── AAPL_2025-06-28.../       # TensorBoard training logs
├── your_script_name.py           # Main Streamlit app
└── requirements.txt              # Dependency list
```

## ⚠️ Disclaimer
This project is for **educational and research purposes only**. It is **not financial advice**.  
Trading real money is risky. Do your own research before using any automated strategy in live markets.
