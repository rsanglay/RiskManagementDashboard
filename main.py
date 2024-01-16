# Import necessary libraries
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime


# Load external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout of the app
app.layout = html.Div(children=[
    # Header
    html.Div([
        html.H1('Portfolio Management Dashboard', style={'textAlign': 'center'}),
        html.P('Analyze and manage your investment portfolio', style={'textAlign': 'center'}),
    ], style={'marginBottom': 50}),

    # Portfolio Metrics
    html.Div([
        html.H3('Portfolio Metrics'),
        dcc.Graph(id='portfolio-metrics'),
    ]),

    # Portfolio Allocation
    html.Div([
        html.H3('Portfolio Allocation'),
        dcc.Graph(id='portfolio-allocation'),
    ]),

    # Stock Prices
    html.Div([
        html.H3('Stock Prices'),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[
                {'label': 'AAPL', 'value': 'AAPL'},
                {'label': 'GOOGL', 'value': 'GOOGL'},
                {'label': 'MSFT', 'value': 'MSFT'},
                {'label': 'AMZN', 'value': 'AMZN'}
            ],
            value=['AAPL'],
            multi=True,
            style={'marginBottom': 20}
        ),
        dcc.Graph(id='stock-prices'),
    ]),

    # Stock Performance
    html.Div([
        html.H3('Stock Performance'),
        dcc.Graph(id='stock-chart'),
        dash_table.DataTable(
            id='stock-table',
            columns=[
                {'name': 'Stock', 'id': 'Stock'},
                {'name': 'Investment', 'id': 'Investment'},
                {'name': 'Return', 'id': 'Return'},
                {'name': 'Volatility', 'id': 'Volatility'},
                {'name': 'Mean Daily Return', 'id': 'Mean Daily Return'},
                {'name': 'Sharpe Ratio', 'id': 'Sharpe Ratio'},
                {'name': 'Risk Score', 'id': 'Risk Score'},
            ],
            style_table={'height': '300px', 'overflowY': 'auto'},
        ),
    ]),

    # VaR Decomposition
    html.Div([
        html.H3('Value at Risk (VaR) Decomposition'),
        dcc.Graph(id='var-decomposition-chart'),
        dash_table.DataTable(
            id='var-decomposition-table',
            columns=[
                {'name': 'Component', 'id': 'Component'},
                {'name': 'VaR Contribution', 'id': 'VaR Contribution'},
            ],
            style_table={'height': '200px', 'overflowY': 'auto'},
        ),
    ]),

    # Combined Chart
    html.Div([
        html.H3('Combined Chart: Starting Balance, Profit, and Final Balance'),
        dcc.Graph(id='combined-chart'),
        html.Div(id='final-balance-text', style={'textAlign': 'center', 'fontSize': 18}),
    ]),


    # Multifactor Scenarios
    html.Div([
        html.H3('Multifactor Scenarios'),
        dcc.Dropdown(
            id='multifactor-stock-dropdown',
            options=[
                {'label': 'AAPL', 'value': 'AAPL'},
                {'label': 'GOOGL', 'value': 'GOOGL'},
                {'label': 'MSFT', 'value': 'MSFT'},
                {'label': 'AMZN', 'value': 'AMZN'}
            ],
            value=['AAPL'],
            multi=True,
            style={'marginBottom': 20}
        ),
        dcc.Dropdown(
            id='multifactor-factors',
            options=[
                {'label': 'Interest Rate', 'value': 'interest_rate'},
                {'label': 'Inflation Rate', 'value': 'inflation_rate'},
                {'label': 'Equity Market Returns', 'value': 'equity_returns'},
            ],
            value=['interest_rate'],
            multi=True,
            style={'marginBottom': 20}
        ),
        dcc.Input(id='multifactor-input', type='number', placeholder='Enter percentage change'),
        html.Button('Run Multifactor Scenario', id='run-multifactor-scenario-button', n_clicks=0),
        dcc.Graph(id='multifactor-scenario-chart'),
    ]),


    ])


# Updated callback to include multifactor scenario chart
@app.callback(
    [
        Output('portfolio-metrics', 'figure'),
        Output('portfolio-allocation', 'figure'),
        Output('stock-chart', 'figure'),
        Output('stock-table', 'data'),
        Output('stock-prices', 'figure'),
        Output('combined-chart', 'figure'),
        Output('final-balance-text', 'children'),
        Output('multifactor-scenario-chart', 'figure'),
    ],
    [
        Input('stock-dropdown', 'value'),
        Input('run-multifactor-scenario-button', 'n_clicks')
    ],
    [
        State('multifactor-stock-dropdown', 'value'),
        State('multifactor-factors', 'value'),
        State('multifactor-input', 'value')
    ]
)
def update_dashboard(selected_stocks, n_clicks_multifactor,
                      multifactor_selected_stocks, selected_factors, multifactor_percentage):
    # Initialize figures and data
    portfolio_metrics_fig, portfolio_allocation_fig, stock_chart, stock_table_data, stock_prices_fig, combined_chart, final_balance_text = generate_dashboard(
        selected_stocks)

    multifactor_scenario_chart = go.Figure()

    # Check if the multifactor scenario button was clicked
    if n_clicks_multifactor > 0 and multifactor_percentage is not None:
        multifactor_scenario_chart = run_multifactor_scenario(multifactor_selected_stocks, selected_factors,
                                                             multifactor_percentage)

    return (
        portfolio_metrics_fig,
        portfolio_allocation_fig,
        stock_chart,
        stock_table_data,
        stock_prices_fig,
        combined_chart,
        final_balance_text,
        multifactor_scenario_chart,  # Add the multifactor scenario chart
    )


# Callback for VaR decomposition chart and table
@app.callback(
    [Output('var-decomposition-chart', 'figure'),
     Output('var-decomposition-table', 'data')],
    [Input('stock-dropdown', 'value')]
)
def update_var_decomposition(selected_stocks):
    var_decomposition_chart, var_decomposition_table = generate_var_decomposition(selected_stocks)
    return var_decomposition_chart, var_decomposition_table




# Function to generate dashboard components
def generate_dashboard(selected_stocks):
    # Get real-time portfolio data
    portfolio_data = get_portfolio_data(selected_stocks)
    portfolio_metrics_fig = generate_pie_chart(portfolio_data, 'Portfolio Metrics')

    # Portfolio Allocation
    portfolio_allocation_fig = generate_portfolio_allocation_chart()

    # Stock Prices
    stock_prices_fig = generate_stock_prices(selected_stocks)

    # Stock Performance
    stock_chart, stock_table_data = generate_stock_performance(selected_stocks)

    # Combined Chart
    combined_chart, final_balance_text = generate_combined_chart(selected_stocks)

    return portfolio_metrics_fig, portfolio_allocation_fig, stock_chart, stock_table_data, stock_prices_fig, combined_chart, final_balance_text


# Function to run a scenario
def run_scenario(selected_stocks, percentage_change):
    scenario_chart, scenario_final_balance_text, _, var_decomposition_chart, var_decomposition_table = generate_scenario_chart(
        selected_stocks, percentage_change)

    return scenario_chart, scenario_final_balance_text, None, var_decomposition_chart, var_decomposition_table


# Function to generate scenario chart and breakdown data
def generate_scenario_chart(selected_stocks, percentage_change):
    scenario_chart = go.Figure()
    scenario_final_balance_text = ""
    scenario_breakdown_data = []

    for stock in selected_stocks:
        stock_data = yf.download(stock, start='2021-01-01', end='2024-01-04', progress=False)

        # Scenario Analysis
        scenario_prices = stock_data['Close'] * (1 + percentage_change / 100)
        scenario_chart.add_trace(
            go.Scatter(x=stock_data.index, y=scenario_prices, mode='lines', name=f'{stock} (Scenario: {percentage_change}%)'))

        # Highlight initial and final points
        scenario_chart.add_trace(
            go.Scatter(x=[stock_data.index[0], stock_data.index[-1]],
                       y=[stock_data['Close'].iloc[0], scenario_prices.iloc[-1]],
                       mode='markers',
                       marker=dict(size=[10, 10], color=['blue', 'red']),
                       name=f'{stock} (Initial/Final)'))

        # Calculate start balance, profit, and final balance for each selected stock in the scenario
        start_balance = 500000 / len(selected_stocks)  # Placeholder for start balance, replace with actual calculation
        start_balance_series = pd.Series(index=stock_data.index, data=start_balance)

        # Calculate daily returns based on scenario prices
        daily_returns = scenario_prices.pct_change()
        daily_returns.iloc[0] = 0  # Set the first element to 0

        # Calculate cumulative returns to get the final balance
        cumulative_returns = (daily_returns + 1).cumprod()

        # Calculate the portfolio balance over time
        portfolio_balance = start_balance_series * cumulative_returns

        profit = portfolio_balance.iloc[-1] - start_balance
        final_balance = portfolio_balance.iloc[-1]

        scenario_final_balance_text += f"Final Balance ({stock}): ${final_balance:,.2f}<br>"

        scenario_breakdown_data.append({
            'Stock': stock,
            'Profit': f"${profit:,.2f}",
            'Final Balance': f"${final_balance:,.2f}",
            'Scenario Price': scenario_prices.tolist()  # Add scenario prices to the breakdown data
        })

    # Set layout for scenario chart
    scenario_chart.update_layout(
        title='Scenario Analysis',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=True,
        legend=dict(x=0.7, y=0.9),  # Adjust the legend position
    )

    return scenario_chart, scenario_final_balance_text, scenario_breakdown_data





# Function to generate VaR decomposition chart and table
def generate_var_decomposition(selected_stocks):
    var_decomposition_chart = go.Figure()
    var_decomposition_table = []

    for stock in selected_stocks:
        stock_data = yf.download(stock, start='2021-01-01', end='2024-01-04', progress=False)

        # Calculate daily returns
        daily_returns = stock_data['Close'].pct_change().dropna()

        # Calculate VaR using historical simulation (assuming a 95% confidence level)
        var_95 = -np.percentile(daily_returns, 5) * 500000  # Negative sign as we want the loss

        # VaR Decomposition Table Formatting
        var_decomposition_table.extend([
            {'Stock': stock, 'Component': 'Market Risk', 'VaR Contribution': f"${var_95 * 0.2:,.2f}"},
            {'Stock': stock, 'Component': 'Credit Risk', 'VaR Contribution': f"${var_95 * 0.1:,.2f}"},
            {'Stock': stock, 'Component': 'Liquidity Risk', 'VaR Contribution': f"${var_95 * 0.05:,.2f}"},
        ])

        # VaR Decomposition Chart
        var_decomposition_chart.add_trace(
            go.Bar(x=['Market Risk', 'Credit Risk', 'Liquidity Risk'],
                   y=[var_95 * 0.2, var_95 * 0.1, var_95 * 0.05],
                   name=f'{stock}'))

    # Set layout for VaR decomposition chart
    var_decomposition_chart.update_layout(
        title='Value at Risk (VaR) Decomposition',
        xaxis_title='Risk Component',
        yaxis_title='VaR Contribution',
        showlegend=True,
        legend=dict(x=0.7, y=0.9),  # Adjust the legend position
    )

    return var_decomposition_chart, var_decomposition_table


# Function to run a multifactor scenario
def run_multifactor_scenario(selected_stocks, selected_factors, percentage_change):
    multifactor_scenario_chart = go.Figure()

    # Generate multifactor scenario data
    multifactor_data = generate_multifactor_data(selected_factors, percentage_change)

    # Plot multifactor scenario data
    for factor in selected_factors:
        multifactor_scenario_chart.add_trace(
            go.Scatter(x=multifactor_data.index, y=multifactor_data[factor], mode='lines', name=f'{factor} (Scenario)'))

    # Set layout for multifactor scenario chart
    multifactor_scenario_chart.update_layout(
        title='Multifactor Scenario Analysis',
        xaxis_title='Date',
        yaxis_title='Factor Value',
        showlegend=True,
        legend=dict(x=0.7, y=0.9),  # Adjust the legend position
    )

    return multifactor_scenario_chart


# Function to generate multifactor scenario data
def generate_multifactor_data(selected_factors, percentage_change):
    # Placeholder for multifactor scenario data, replace with actual calculations
    date_rng = pd.date_range(start='2021-01-01', end='2024-01-04', freq='B')
    multifactor_data = pd.DataFrame(index=date_rng)

    for factor in selected_factors:
        # Simulate factor changes over time
        multifactor_data[factor] = np.cumsum(np.random.normal(0, 0.001, len(date_rng))) * percentage_change

    return multifactor_data

# New function for multifactor scenario chart
def generate_multifactor_scenario_chart(selected_factors, percentage_change):
    multifactor_scenario_chart = run_multifactor_scenario(selected_factors, percentage_change)
    return multifactor_scenario_chart, None  # Return None as a placeholder for the second expected value



# Function to get real-time portfolio data
def get_portfolio_data(selected_stocks):
    portfolio_data = pd.DataFrame(
        columns=['Stock', 'Investment', 'Return', 'Volatility', 'Mean Daily Return', 'Sharpe Ratio', 'Risk Score',
                 'Profit'])
    for stock in selected_stocks:
        stock_data = yf.download(stock, start='2021-01-01', end='2024-01-04', progress=False)
        stock_return = (stock_data['Close'].pct_change() + 1).prod() - 1
        daily_return = stock_data['Close'].pct_change().mean()
        volatility = stock_data['Close'].pct_change().std()

        # Placeholder values for metrics, replace with actual calculations
        investment = 50000
        profit = investment * stock_return
        sharpe_ratio = (daily_return / volatility) * np.sqrt(252)  # 252 trading days in a year
        risk_score = calculate_risk_score(sharpe_ratio)

        portfolio_data = pd.concat([portfolio_data, pd.DataFrame({
            'Stock': [stock],
            'Investment': [investment],
            'Return': [stock_return],
            'Volatility': [volatility],
            'Mean Daily Return': [daily_return],
            'Sharpe Ratio': [sharpe_ratio],
            'Risk Score': [risk_score],
            'Profit': [profit]
        })])

    return portfolio_data


# Function to generate pie chart
def generate_pie_chart(data, title):
    fig = go.Figure(data=[go.Pie(labels=data['Stock'], values=data['Investment'] + data['Profit'])])
    fig.update_layout(title=title)
    return fig


# Function to generate portfolio allocation chart
def generate_portfolio_allocation_chart():
    return px.bar(x=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                  y=[0.25, 0.25, 0.25, 0.25],
                  title='Portfolio Allocation',
                  labels={'x': 'Stock', 'y': 'Allocation'},
                  template='plotly',  # Use the 'plotly' template to avoid pattern-related issues
                  )


# Function to generate stock prices chart
def generate_stock_prices(selected_stocks):
    stock_prices_fig = go.Figure()

    for stock in selected_stocks:
        stock_data = yf.download(stock, start='2021-01-01', end='2024-01-04', progress=False)
        stock_prices_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=stock))

    stock_prices_fig.update_layout(title='Stock Prices Over Time',
                                   xaxis_title='Date',
                                   yaxis_title='Stock Price',
                                   showlegend=True)

    return stock_prices_fig


# Function to generate stock performance chart and table
def generate_stock_performance(selected_stocks):
    columns = ['Stock', 'Investment', 'Return', 'Volatility', 'Mean Daily Return', 'Sharpe Ratio', 'Risk Score',
               'Profit']
    stock_data = pd.DataFrame(columns=columns)

    for stock in selected_stocks:
        stock_data = pd.concat([stock_data, pd.DataFrame({
            'Stock': [stock],
            'Investment': [50000],  # Placeholder for investment, replace with actual calculation
            'Return': [0.02],  # Placeholder for return, replace with actual calculation
            'Volatility': [0.03],  # Placeholder for volatility, replace with actual calculation
            'Mean Daily Return': [0.001],  # Placeholder for mean daily return, replace with actual calculation
            'Sharpe Ratio': [1.2],  # Placeholder for Sharpe ratio, replace with actual calculation
            'Risk Score': [3],  # Placeholder for risk score, replace with actual calculation
            'Profit': [10000]  # Placeholder for profit, replace with actual calculation
        })])

    stock_chart = px.bar(stock_data, x='Stock', y='Return', title='Stock Performance')
    stock_table_data = stock_data.to_dict('records')

    return stock_chart, stock_table_data


# Function to generate combined bar chart for Starting Balance, Profit, and Final Balance
def generate_combined_chart(selected_stocks):
    start_balances = []
    profits = []
    final_balances = []

    for stock in selected_stocks:
        stock_data = yf.download(stock, start='2021-01-01', end='2024-01-04', progress=False)

        # Calculate start balance, profit, and final balance for each selected stock
        start_balance = 500000 / len(selected_stocks)
        profit = start_balance * (stock_data['Close'].pct_change() + 1).prod() - start_balance
        final_balance = start_balance + profit

        start_balances.append(start_balance)
        profits.append(profit)
        final_balances.append(final_balance)

    # Create a DataFrame for combined chart
    combined_data = pd.DataFrame({
        'Stock': selected_stocks,
        'Start Balance': start_balances,
        'Profit': profits,
        'Final Balance': final_balances
    })

    # Melt the DataFrame for easier plotting
    melted_data = pd.melt(combined_data, id_vars='Stock', var_name='Metric', value_name='Amount')

    # Generate the combined bar chart (vertical bars)
    combined_chart = px.bar(melted_data, x='Metric', y='Amount', color='Stock',
                            labels={'Metric': 'Metric', 'Amount': 'Amount ($)'},
                            title='Combined Chart: Starting Balance, Profit, and Final Balance',
                            barmode='group')  # Use barmode='group' to group bars by stock

    # Generate final balance text
    final_balance_text = f"Final Balance: ${combined_data['Final Balance'].sum():,.2f}"

    return combined_chart, final_balance_text


# Function to calculate risk score based on Sharpe ratio
def calculate_risk_score(sharpe_ratio):
    if sharpe_ratio < 0:
        return 1
    elif 0 <= sharpe_ratio < 0.5:
        return 2
    elif 0.5 <= sharpe_ratio < 1:
        return 3
    elif 1 <= sharpe_ratio < 1.5:
        return 4
    else:
        return 5


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)