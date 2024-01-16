# RiskManagementDashboard

This is a Dash web application for analyzing and managing an investment portfolio. The dashboard provides various metrics and visualizations to help users make informed decisions about their investments.

Features:

Portfolio Metrics:
Displays key metrics related to the overall portfolio, including a pie chart of investments and profits.

Portfolio Allocation:
Visualizes the allocation of the portfolio across selected stocks.

Stock Prices:
Allows users to select stocks and view their historical closing prices over time.

Stock Performance:
Provides a bar chart and table summarizing the performance metrics for selected stocks.

Value at Risk (VaR) Decomposition:
Analyzes the VaR decomposition of selected stocks, considering market risk, credit risk, and liquidity risk.

Combined Chart: Starting Balance, Profit, and Final Balance:
Presents a combined bar chart showcasing the starting balance, profit, and final balance for selected stocks.

Multifactor Scenarios:
Enables users to run multifactor scenarios by selecting stocks, factors (e.g., interest rate, inflation rate), and specifying a percentage change.
How to Run:

Install the required libraries by running pip install dash yfinance numpy pandas plotly scipy.
Run the script using python script_name.py in the terminal.
Open a web browser and go to http://127.0.0.1:8050/ to access the dashboard.
Additional Information:

The dashboard uses real-time stock data obtained from Yahoo Finance (yfinance library).
Users can save various charts and figures from the dashboard to their desktop.
The application provides a scenario analysis feature, allowing users to simulate portfolio changes based on a specified percentage change.
VaR decomposition helps users understand the contribution of different risks to the overall Value at Risk.
Multifactor scenarios consider the impact of external factors on stock prices.
Feel free to explore, analyze, and enhance the dashboard for personalized portfolio management.
