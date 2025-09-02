import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Guide to Using YFinance with Python for Effective Stock Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""YFinance ([`yfinance`](https://github.com/ranaroussi/yfinance)) remains one of the most accessible and versatile libraries in Python for gathering stock market data. It allows users to retrieve extensive financial data, including historical prices, financial statements, dividends, splits, and more — essential for crafting actionable insights. This guide explores how to use yfinance effectively for stock analysis, covering everything from basic setups to advanced data manipulations.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Setting Up `yfinance` for Python
    Before you can start analyzing data with yfinance, you need to set it up within your Python environment.

    ### Installation

    Install yfinance using pip to ensure the latest version with updated features and fixes:

    ```python
    pip install yfinance --upgrade --no-cache-dir
    ```

    ### Import Libraries

    In addition to `yfinance`, we’ll use some other essential libraries like `pandas` and `matplotlib` for data manipulation and visualization:
    """
    )
    return


@app.cell
def _():
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    return plt, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Retrieving Basic Stock Information
    To begin, let’s pull some basic information about a stock. YFinance uses ticker symbols (e.g., “AAPL” for Apple, “MSFT” for Microsoft) to identify stocks.
    """
    )
    return


@app.cell
def _(yf):
    # Define the ticker and create a Ticker object
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    return (stock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `info` attribute retrieves essential information like: 

    - Company name, industry, and sector
    - Market cap
    - PE Ratio, beta, dividend yield
    - Current trading volume
    """
    )
    return


@app.cell
def _(stock):
    # Fetch basic stock information
    stock_info = stock.info
    print(stock_info)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Fetching Historical Price Data
    Historical price data is essential for analyzing stock performance over time. You can retrieve it with the `history()` method:
    """
    )
    return


@app.cell
def _(stock):
    # Fetch historical data for a specific period
    historical_data = stock.history(period="5y")
    print(historical_data.tail())
    return (historical_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Customizing Historical Data
    The history() function allows for customized data retrieval. You can specify:

    - **Period**: "1mo", "1y", "5y", or "max"
    - **Interval**: "1d", "1wk", "1mo"
    """
    )
    return


@app.cell
def _(stock):
    # Fetch historical data with a custom range and interval
    custom_data = stock.history(start="2020-01-01", end="2023-12-31", interval="1wk")
    print(custom_data.tail())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Analyzing Daily Stock Price Movements
    Analyzing daily price movements helps in understanding short-term trends. With yfinance, you can retrieve intraday data with minute-level intervals.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch intraday data with minute intervals for the current day
    intraday_data = stock.history(period="1d", interval="1m")
    print(intraday_data.tail())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Retrieving Bulk Data for Multiple Tickers
    Analyzing a portfolio or comparing stocks? YFinance lets you pull data for multiple tickers simultaneously.
    """
    )
    return


@app.cell
def _(yf):
    # Define multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Fetch data for all tickers over the last year
    multi_data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True)
    print(multi_data.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Calculating Returns and Volatility
    Historical data is essential for calculating metrics like daily returns and volatility.
    """
    )
    return


@app.cell
def _(historical_data):
    # Calculate daily returns
    historical_data['Daily Return'] = historical_data['Close'].pct_change()

    # Calculate rolling volatility (30-day)
    historical_data['Volatility'] = historical_data['Daily Return'].rolling(window=30).std()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the data to understand the stock’s historical volatility and returns over time:""")
    return


@app.cell
def _(historical_data, plt):
    # Plot daily return and volatility
    historical_data[['Daily Return', 'Volatility']].plot(subplots=True, title="Daily Returns and Volatility")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Dividend Data Analysis
    Dividends are a key component of total returns for many investors. YFinance provides historical dividend information.

    Using this data, you can calculate the **dividend yield** or plot the **dividend growth rate** over time to assess whether the stock is a reliable income generator.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch dividend history
    dividends = stock.dividends
    print(dividends.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. Retrieving Stock Split Data
    Stock splits increase liquidity and may influence stock price trends. YFinance provides the split history for a given ticker.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch stock split history
    splits = stock.splits
    print(splits)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 9. Analyzing Institutional and Insider Holding
    YFinance includes information on major holders, such as institutional and insider holdings.
    """
    )
    return


@app.cell
def _(stock):
    # Retrieve institutional holders and major holders
    institutional_holders = stock.institutional_holders
    major_holders = stock.major_holders

    print("Institutional Holders:\n", institutional_holders)
    print("="*50)
    print("Major Holders:\n", major_holders)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 10. Accessing Financial Statements
    Financial statements are the backbone of fundamental analysis. YFinance provides income statements, balance sheets, and cash flow data.

    With this data, you can calculate financial ratios like **return on assets**, **current ratio**, and **operating margin** to assess the company’s financial health.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch financial statements
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow

    print("Income Statement:\n", income_statement)
    print("="*50)
    print("Balance Sheet:\n", balance_sheet)
    print("="*50)
    print("Cash Flow:\n", cash_flow)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 11. Environmental, Social, and Governance (ESG) Data
    Many investors prioritize sustainable and ethical investments. YFinance offers ESG data for companies that report these metrics.

    ESG data is increasingly used by socially responsible investors to screen stocks based on environmental impact, social responsibility, and governance practices.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch ESG scores
    esg_data = stock.sustainability
    print(esg_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 12. Accessing Analyst Recommendations
    Analyst ratings offer insight into market sentiment and valuation perspectives.
    """
    )
    return


@app.cell
def _(stock):
    # Fetch analyst recommendations
    analyst_recommendations = stock.recommendations
    print(analyst_recommendations)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 13. Options Data
    For options traders, yfinance provides options chains with expiration dates and the associated call and put options.
    """
    )
    return


@app.cell
def _(stock):
    # Get available options expiration dates
    expiration_dates = stock.options
    print("Expiration Dates:\n", expiration_dates)

    # Retrieve option chain for a specific expiration date
    option_chain = stock.option_chain(expiration_dates[0])
    print("Calls:\n", option_chain.calls)
    print("Puts:\n", option_chain.puts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 14. Visualizing Stock Data with `yfinance`
    Visualizations make it easier to interpret stock data trends. `yfinance` integrates well with visualization libraries like `matplotlib`.
    """
    )
    return


@app.cell
def _(historical_data, plt):
    # Plot stock price history
    historical_data["Close"].plot(title="AAPL Stock Price Over Time", figsize=(12, 6))
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
