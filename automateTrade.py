import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os

# --- Create agents directory if it doesn't exist ---
os.makedirs("agents", exist_ok=True)
# Define the default model path. You can create this by training on 'SPY' and renaming the file.
DEFAULT_MODEL_PATH = "agents/genius_trader_agent.zip"

plt.switch_backend('Agg')

# --- 1. RL Environment ---
class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001, lookback_window=20):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.lookback_window = lookback_window
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns) + 4,), dtype=np.float32)
        self.trades = []

    def _normalize(self, data):
        return (data - data.mean()) / (data.std() + 1e-9)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = self.lookback_window
        self.net_worth_history = [self.initial_balance] * self.lookback_window
        self.trades = []
        self.average_buy_price = 0
        return self._next_observation(), {}

    def _next_observation(self):
        frame = self.df.iloc[self.current_step].values
        current_price = self.df.loc[self.df.index[self.current_step], 'close']
        profit_loss_percent = 0.0
        if self.shares_held > 0 and self.average_buy_price > 0:
            profit_loss_percent = (current_price - self.average_buy_price) / self.average_buy_price
        obs = np.concatenate((
            self._normalize(frame),
            [self.balance / self.initial_balance],
            [self.shares_held / 1000],
            [self.net_worth / self.initial_balance],
            [profit_loss_percent]
        ))
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.loc[self.df.index[self.current_step], 'close']
        profit_from_sale, trade_executed = self._take_action(action, current_price)
        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        reward = 0
        if profit_from_sale > 0:
            reward += 1.0
        if self.shares_held > 0 and self.average_buy_price > 0:
            unrealized_pnl_percent = (current_price - self.average_buy_price) / self.average_buy_price
            reward += unrealized_pnl_percent
        if not trade_executed and action != 0:
            reward -= 0.5
        if not self.shares_held > 0 and action == 0:
             reward -= 0.05
        terminated = self.net_worth <= self.initial_balance * 0.5
        truncated = self.current_step >= len(self.df) - 1
        return self._next_observation(), reward, terminated, truncated, {}

    def _take_action(self, action, current_price):
        profit_from_sale = 0
        trade_executed = False
        if action > 0 and action <= 4: # Buy
            buy_percentage = action * 0.25
            investment_amount = self.balance * buy_percentage
            if investment_amount > 1:
                shares_to_buy = investment_amount / current_price
                fee = shares_to_buy * current_price * self.transaction_fee_percent
                total_investment = (self.average_buy_price * self.shares_held) + (shares_to_buy * current_price)
                total_shares = self.shares_held + shares_to_buy
                self.average_buy_price = total_investment / total_shares if total_shares > 0 else 0
                self.shares_held += shares_to_buy
                self.balance -= (shares_to_buy * current_price + fee)
                trade_executed = True
                self.trades.append({'step': self.current_step, 'type': 'BUY', 'shares': shares_to_buy, 'price': current_price, 'amount': investment_amount, 'balance': self.balance})
        elif action > 4: # Sell
            sell_percentage = (action - 4) * 0.25
            shares_to_sell = self.shares_held * sell_percentage
            if shares_to_sell > 1e-6:
                sale_value = shares_to_sell * current_price
                fee = sale_value * self.transaction_fee_percent
                if current_price > self.average_buy_price:
                    profit_from_sale = (current_price - self.average_buy_price) * shares_to_sell
                self.balance += (sale_value - fee)
                self.shares_held -= shares_to_sell
                if self.shares_held < 1e-6: self.average_buy_price = 0
                trade_executed = True
                self.trades.append({'step': self.current_step, 'type': 'SELL', 'shares': shares_to_sell, 'price': current_price, 'amount': sale_value, 'balance': self.balance})
        return profit_from_sale, trade_executed

# --- Live Training Callback ---
class StreamlitCallback(BaseCallback):
    def __init__(self, total_timesteps, progress_bar, chart_placeholder, metrics_placeholder, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps; self.progress_bar = progress_bar; self.chart_placeholder = chart_placeholder; self.metrics_placeholder = metrics_placeholder; self.reward_history = []
    def _on_step(self) -> bool:
        progress = self.n_calls / self.total_timesteps; self.progress_bar.progress(min(progress, 1.0), text=f"Training Timestep: {self.n_calls}/{self.total_timesteps}")
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.reward_history.append(info['episode']['r'])
                    if len(self.reward_history) % 5 == 0: self.chart_placeholder.line_chart(pd.DataFrame(self.reward_history, columns=['Episodic Reward']))
        if self.n_calls % 2048 == 0:
            if len(self.model.logger.name_to_value) > 0:
                metrics_text = "### Latest Training Metrics:\n"; display_metrics = {"rollout/ep_rew_mean": "Mean Reward", "train/loss": "Loss", "time/fps": "FPS"}
                for key, name in display_metrics.items():
                    if key in self.model.logger.name_to_value: metrics_text += f"- **{name}**: {self.model.logger.name_to_value[key]:.4f}\n"
                self.metrics_placeholder.markdown(metrics_text)
        return True

# --- 2. Advanced Data Processing Pipeline ("Genius" Features) ---
@st.cache_data
def process_data(ticker, start_date, end_date):
    """Downloads and engineers a rich set of features for the 'Genius Agent'."""
    raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if raw_df.empty: return None, None
    df = pd.DataFrame(index=raw_df.index)
    df['open'] = raw_df['Open']; df['high'] = raw_df['High']; df['low'] = raw_df['Low']; df['volume'] = raw_df['Volume']; df['close'] = raw_df['Close']
    df.columns = [col.lower() for col in df.columns]

    # 1. Standard Technical Indicators
    df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.bbands(append=True); df.ta.atr(append=True); df.ta.stoch(append=True)

    # 2. Long-Term Trend & Seasonal Features
    df['month'] = df.index.month
    df['price_vs_sma200'] = df['close'] / ta.sma(df['close'], length=200)
    df['sma50_vs_sma200'] = ta.sma(df['close'], length=50) / ta.sma(df['close'], length=200)

    # 3. Support, Resistance, and Fibonacci Features
    long_window = 252 
    df['rolling_low'] = df['close'].rolling(window=long_window).min()
    df['rolling_high'] = df['close'].rolling(window=long_window).max()
    df['fib_382'] = df['rolling_low'] + (df['rolling_high'] - df['rolling_low']) * 0.382
    df['fib_618'] = df['rolling_low'] + (df['rolling_high'] - df['rolling_low']) * 0.618
    df['dist_to_fib382'] = (df['close'] - df['fib_382']) / df['close']
    df['dist_to_fib618'] = (df['close'] - df['fib_618']) / df['close']
    
    pivots = []
    # Find pivots with a smaller window for more frequent signals
    pivot_window = 22 # Approx 1 month
    for i in range(pivot_window, len(df) - pivot_window):
        window_slice = df['close'][i-pivot_window:i+pivot_window]
        if df['close'][i] == window_slice.max() or df['close'][i] == window_slice.min():
            pivots.append(df.index[i])
    
    # Calculate S/R based on longer-term pivots
    long_pivots = []
    for i in range(long_window, len(df) - long_window):
        window_slice = df['close'][i-long_window:i+long_window]
        if df['close'][i] == window_slice.max() or df['close'][i] == window_slice.min():
            long_pivots.append(df['close'][i])

    if len(long_pivots) > 2:
        current_price = df['close'].iloc[-1]
        support = max([p for p in long_pivots if p < current_price], default=df['close'].min())
        resistance = min([p for p in long_pivots if p > current_price], default=df['close'].max())
        df['dist_to_support'] = (df['close'] - support) / df['close']
        df['dist_to_resistance'] = (df['close'] - resistance) / df['close']
    else:
        df['dist_to_support'] = 0; df['dist_to_resistance'] = 0

    df.drop(columns=['rolling_low', 'rolling_high', 'fib_382', 'fib_618'], errors='ignore', inplace=True)
    df.dropna(inplace=True)
    if df.empty: return None, None
    df_processed = df.select_dtypes(include=np.number)
    return df, df_processed

# --- 3. Backtesting ---
def backtest_and_visualize(model, test_env_df, test_df_viz, initial_balance):
    aligned_test_df_viz = test_df_viz.loc[test_env_df.index]
    test_env = StockTradingEnv(test_env_df, initial_balance=initial_balance)
    obs, _ = test_env.reset()
    portfolio_values, buy_signals, sell_signals = [], [], []
    for i in range(len(test_env_df) - test_env.lookback_window - 1):
        portfolio_values.append(test_env.net_worth)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        price = aligned_test_df_viz.loc[aligned_test_df_viz.index[test_env.current_step], 'close']
        if action > 0 and action <= 4:
            buy_signals.append(price); sell_signals.append(np.nan)
        elif action > 4:
            sell_signals.append(price); buy_signals.append(np.nan)
        else:
            buy_signals.append(np.nan); sell_signals.append(np.nan)
        if terminated or truncated: break
    portfolio_values.append(test_env.net_worth)
    trades_df = pd.DataFrame(test_env.trades)
    if not trades_df.empty:
        trades_df['date'] = aligned_test_df_viz.index[trades_df['step']]
        trades_df = trades_df[['date', 'type', 'shares', 'price', 'amount', 'balance']].set_index('date')
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance
    buy_hold_return = (aligned_test_df_viz['close'].iloc[-1] - aligned_test_df_viz['close'].iloc[0]) / aligned_test_df_viz['close'].iloc[0]
    trading_days = len(test_env_df)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    rolling_max = pd.Series(portfolio_values).cummax()
    daily_drawdown = (pd.Series(portfolio_values) / rolling_max) - 1.0
    max_drawdown = daily_drawdown.min()
    metrics = {"Final Portfolio Value": f"${portfolio_values[-1]:,.2f}", "Total Return": f"{total_return:.2%}", "Buy and Hold Return": f"{buy_hold_return:.2%}", "Annualized Return": f"{annualized_return:.2%}", "Annualized Volatility": f"{annualized_volatility:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}", "Max Drawdown": f"{max_drawdown:.2%}"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    index_for_plot = aligned_test_df_viz.index[test_env.lookback_window:test_env.lookback_window+len(portfolio_values)]
    portfolio_series = pd.Series(portfolio_values, index=index_for_plot)
    portfolio_series.plot(ax=ax1, label='RL Agent Portfolio', color='blue')
    buy_hold_series = (aligned_test_df_viz['close'] / aligned_test_df_viz['close'].iloc[0] * initial_balance)
    buy_hold_series.plot(ax=ax1, label='Buy and Hold Strategy', linestyle='--', color='orange')
    ax1.set_title('Portfolio Value: RL Agent vs. Buy and Hold', fontsize=16); ax1.set_ylabel('Portfolio Value ($)'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    aligned_test_df_viz['close'].plot(ax=ax2, label='Stock Price', color='black', alpha=0.7)
    signal_dates = aligned_test_df_viz.index[test_env.lookback_window + 1 : test_env.lookback_window + 1 + len(buy_signals)]
    ax2.plot(signal_dates, buy_signals, '^', markersize=10, color='green', label='Buy Signal', alpha=0.8)
    ax2.plot(signal_dates, sell_signals, 'v', markersize=10, color='red', label='Sell Signal', alpha=0.8)
    ax2.set_title('Trading Signals on Price Chart', fontsize=16); ax2.set_ylabel('Stock Price ($)'); ax2.set_xlabel('Date'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return metrics, fig, trades_df

# --- 4. Streamlit UI with advanced Save/Load Logic ---
st.set_page_config(layout="wide")
st.title("🧠 Genius RL Stock Trading Agent")
st.markdown("This agent uses a rich set of features and can manage a portfolio of specialized models.")
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    
    specialized_model_path = f"agents/agent_{ticker}.zip"
    
    specialized_model_exists = os.path.exists(specialized_model_path)
    default_model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    
    options = ["Train a new specialist agent"]
    default_index = 0
    
    if specialized_model_exists:
        options.insert(0, f"Load specialist agent for {ticker}")
        options.append(f"Continue training {ticker} agent")
        default_index = 0
    elif default_model_exists:
        options.insert(0, "Load generic 'genius' agent")
        default_index = 0
        
    training_mode = st.radio("Select Agent Mode:", options, index=default_index, help="Each stock can have its own specialized agent.")
    st.markdown("---"); st.subheader("Date Ranges")
    col1, col2 = st.columns(2)
    with col1: train_start_date = st.date_input("Training Start", datetime.date(2018, 1, 1)); test_start_date = st.date_input("Testing Start", datetime.date(2023, 1, 1))
    with col2: train_end_date = st.date_input("Training End", datetime.date(2022, 12, 31)); test_end_date = st.date_input("Testing End", datetime.date.today())
    st.markdown("---"); st.subheader("Agent Parameters")
    initial_balance = st.number_input("Initial Balance ($)", min_value=1000, value=10000, step=1000)
    total_timesteps = st.number_input("Training Timesteps", min_value=50000, value=100000, step=10000, help="The 'Genius' agent needs more timesteps to learn from its many features.")

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    st.cache_data.clear(); st.cache_resource.clear()
    if train_start_date >= train_end_date or test_start_date >= test_end_date or train_end_date >= test_start_date:
        st.error("Error: Please ensure date ranges are valid and do not overlap.")
    else:
        try:
            with st.spinner("Downloading and engineering advanced features..."):
                train_df_viz, train_df_processed = process_data(ticker, train_start_date, train_end_date)
                test_df_viz, test_df_processed = process_data(ticker, test_start_date, test_end_date)
            if train_df_processed is None or len(train_df_processed) < 252*2: st.error(f"Not enough training data for {ticker} to calculate long-term patterns. Please select a much wider date range (e.g., 3+ years).")
            elif test_df_processed is None or len(test_df_processed) < 252*2: st.error(f"Not enough testing data for {ticker} to calculate long-term patterns. Please select a much wider date range (e.g., 3+ years).")
            else:
                st.success("Data processing complete!")
                train_env = DummyVecEnv([lambda: StockTradingEnv(train_df_processed)])
                model = None
                log_dir = f"./tensorboard_logs/{ticker}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
                
                if training_mode.startswith("Load"):
                    path_to_load = specialized_model_path if training_mode.startswith(f"Load specialist") else DEFAULT_MODEL_PATH
                    st.info(f"Loading model from {path_to_load}...")
                    model = PPO.load(path_to_load, env=train_env)
                    st.success("Model loaded successfully!")
                
                elif training_mode.startswith("Continue"):
                    st.info(f"Loading specialist agent for {ticker} to continue training...")
                    model = PPO.load(specialized_model_path, env=train_env)
                    st.success("Model loaded. Starting fine-tuning...")
                    st.subheader(f"🤖 Fine-Tuning Agent for {ticker}...")
                    progress_bar = st.progress(0, text="Initializing training...")
                    col1, col2 = st.columns([3, 2]);
                    with col1: chart_placeholder = st.empty()
                    with col2: metrics_placeholder = st.empty()
                    streamlit_callback = StreamlitCallback(total_timesteps, progress_bar, chart_placeholder, metrics_placeholder)
                    model.learn(total_timesteps=total_timesteps, callback=streamlit_callback, reset_num_timesteps=False)
                    st.success(f"Agent fine-tuning complete! Saving improved model to {specialized_model_path}...")
                    model.save(specialized_model_path); st.success("Improved model saved successfully!")
                
                if model is None:
                    st.warning(f"Training a new specialist agent for {ticker}...")
                    st.subheader(f"🤖 Training New Agent for {ticker}...")
                    progress_bar = st.progress(0, text="Initializing training...")
                    col1, col2 = st.columns([3, 2])
                    with col1: chart_placeholder = st.empty()
                    with col2: metrics_placeholder = st.empty()
                    streamlit_callback = StreamlitCallback(total_timesteps, progress_bar, chart_placeholder, metrics_placeholder)
                    model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log=log_dir)
                    model.learn(total_timesteps=total_timesteps, callback=streamlit_callback)
                    st.success(f"Agent training complete! Saving model to {specialized_model_path}...")
                    model.save(specialized_model_path); st.success("Model saved successfully!")
                
                with st.spinner("Running backtest and generating results..."):
                    metrics, fig, trades_df = backtest_and_visualize(model, test_df_processed, test_df_viz, initial_balance)
                st.success("Backtesting complete!")
                st.header("Backtesting Results")
                res_col1, res_col2, res_col3 = st.columns(3)
                agent_return_val = float(metrics['Total Return'].strip('%'))
                bh_return_val = float(metrics['Buy and Hold Return'].strip('%'))
                delta_val = f"{agent_return_val - bh_return_val:.2f}%"
                with res_col1:
                    st.metric("Final Portfolio Value", metrics["Final Portfolio Value"]); st.metric("Agent Total Return", metrics["Total Return"]); st.metric("Buy and Hold Return", metrics["Buy and Hold Return"], delta=delta_val)
                with res_col2:
                    st.metric("Annualized Return", metrics["Annualized Return"]); st.metric("Annualized Volatility", metrics["Annualized Volatility"])
                with res_col3:
                    st.metric("Sharpe Ratio", metrics["Sharpe Ratio"]); st.metric("Max Drawdown", metrics["Max Drawdown"])
                st.pyplot(fig)
                
                st.header("Transaction History")
                if trades_df.empty:
                    st.info("The agent decided not to make any trades during this backtest period.")
                else:
                    st.dataframe(trades_df.style.format({ "shares": "{:.4f}", "price": "${:,.2f}", "amount": "${:,.2f}", "balance": "${:,.2f}"}))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")