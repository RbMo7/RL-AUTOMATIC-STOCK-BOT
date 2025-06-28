🧠 Intelligent Reinforcement Learning Trading Agent

This project is a sophisticated stock trading bot that uses Reinforcement Learning (RL) to develop and backtest active trading strategies. The agent is built with Stable-Baselines3 using the Proximal Policy Optimization (PPO) algorithm and interacts with a custom trading environment built with Gymnasium. The entire application is wrapped in an interactive web UI created with Streamlit.

The agent evolves beyond simple indicators, incorporating concepts like market trends, support/resistance, and Fibonacci levels to make nuanced decisions. Its goal is not just to maximize profit, but to do so by learning from the consequences of its actions, including taking profits, managing risk, and avoiding passivity.


(You can replace the URL above with a screenshot of your running application)
✨ Features

    Sophisticated RL Agent:

        Utilizes the PPO algorithm for stable and effective learning.

        The agent's "brain" is a Multi-Layer Perceptron (MLP).

    Advanced Feature Engineering (The "Genius" Vision):

        Standard Indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator.

        Long-Term Trend Analysis: Understands the market context using the 200-day and 50-day moving averages (Golden/Death Cross proxy).

        Market Structure Analysis: Automatically calculates and uses the distance to key Support and Resistance levels.

        Psychological Levels: Calculates and uses Fibonacci Retracement levels to identify potential reversal points.

        Seasonality: Uses the month of the year as a feature to learn seasonal patterns.

    Intelligent Reward System:

        Consequence-Driven: The agent is rewarded based on the future outcome of its actions, teaching it to anticipate.

        Profit-Taking Bonus: Receives a large, stable bonus for selling a position for a profit.

        Risk Management: Is penalized for holding losing positions and for attempting invalid trades (e.g., selling with no shares).

        "Cost of Capital": Is penalized for being passive and holding cash, encouraging it to seek opportunities.

    Interactive Web UI (Streamlit):

        Full control over stock ticker, date ranges, and initial balance.

        Live Training Dashboard: Monitor the agent's learning process in real-time with a progress bar and reward chart.

        Advanced Model Management:

            Automatically saves/loads specialist agents for each ticker (e.g., agents/agent_AAPL.zip).

            Intelligently falls back to a generic model (agents/genius_trader_agent.zip) if a specialist doesn't exist.

            Supports continual learning to fine-tune saved agents on new data.

        Detailed Backtesting Results: View comprehensive performance metrics, a portfolio value chart vs. a "Buy and Hold" strategy, and a price chart with clear Buy/Sell signals.

        Full Transaction History: A detailed log of every trade the agent makes during a backtest.

    Experiment Tracking:

        Automatically saves detailed training logs into timestamped folders inside tensorboard_logs/.

        Fully compatible with TensorBoard for advanced analysis of training performance.

🛠️ Technology Stack

    Python 3.9+

    Streamlit: For the interactive web interface.

    Stable-Baselines3: For the PPO Reinforcement Learning algorithm.

    Gymnasium: For building the custom trading environment.

    Pandas & NumPy: For data manipulation and numerical operations.

    yfinance: For downloading historical stock data from Yahoo Finance.

    pandas-ta: For calculating technical analysis indicators.

    Matplotlib: For generating backtesting charts.

⚙️ Installation & Setup

Follow these steps to get the project running on your local machine.

    Clone the Repository (or download the files):

    git clone <your-repo-url>
    cd <your-repo-folder>

    Create a Virtual Environment:
    It's highly recommended to use a virtual environment to manage dependencies.

    python -m venv myenv

    Activate the environment:

        On Windows: myenv\Scripts\activate

        On macOS/Linux: source myenv/bin/activate

    Install Dependencies:
    Install all the required libraries from the requirements.txt file.

    pip install -r requirements.txt

    (If you don't have a requirements.txt file, create one and paste the following content into it):

    streamlit
    pandas
    numpy
    yfinance
    pandas-ta
    matplotlib
    gymnasium
    stable-baselines3[extra]
    tensorflow

🚀 How to Use

    Run the Streamlit App:
    Make sure your virtual environment is activated. Then, run the following command in your terminal:

    streamlit run your_script_name.py

    Your web browser should automatically open with the application.

    Configure a Training Run:
    Use the sidebar to configure your session:

        Stock Ticker: Enter the symbol of the stock you want to analyze (e.g., AAPL, NVDA, SPY).

        Agent Mode:

            Train a new specialist agent: This is the default. It will train an agent specifically for the chosen ticker and save it (e.g., as agents/agent_AAPL.zip). You must do this first for any new stock.

            Load saved model...: Once a model is saved, this option appears. It will instantly load the agent for backtesting without retraining. It intelligently looks for a specialist first, then a generic fallback model.

            Continue training...: Loads a saved specialist agent and trains it for more timesteps to improve it.

        Date Ranges: Select periods for training and testing. Ensure they do not overlap and are long enough for the "Genius" features (3+ years is recommended for training).

        Agent Parameters: Adjust the initial balance, training timesteps, and transaction fees. Higher fees will make the agent more cautious.

    Run and Analyze:

        Click the "🚀 Run Analysis" button.

        If training, watch the live progress dashboard.

        Review the backtesting results, charts, and transaction history to evaluate the agent's performance.

    Creating the Generic genius_trader_agent.zip:

        To create your fallback model, train an agent on a broad market ETF like SPY for a high number of timesteps (e.g., 1,000,000).

        Once it's saved as agents/agent_SPY.zip, manually rename the file to agents/genius_trader_agent.zip.


⚠️ Disclaimer

This project is for educational and research purposes only. It is not financial advice. Automated trading with real money is extremely risky and can lead to significant financial loss. Always perform your own thorough research and risk assessment before deploying any trading strategy.
