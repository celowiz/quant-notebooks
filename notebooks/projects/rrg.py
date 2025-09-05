import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Relative Rotation Graph (RRG) Implementation

    This script implements the famous **Relative Rotation Graph (RRG)** visualization technique
    for analyzing relative strength and momentum of any list of assets against a benchmark.

    A Relative Rotation Graph (RRG) plots two normalized indicators for a universe of securities versus a benchmark:

    - **JdK RS-Ratio (horizontal axis)** — measures trend in relative performance
    - **JdK RS-Momentum (vertical axis)** — momentum (rate-of-change) of the RS-Ratio

    RRG was developed by Julius de Kempenaer and visualizes relative performance in four quadrants:

    - 🟢 **Leading (top-right):** The asset is stronger than the benchmark and it's getting stronger (positive momentum).
    - 🟡 **Weakening** (bottom-right):** The asset is stronger than the benchmark, but it's getting weaker (negative momentum).
    - 🔴 **Lagging** (bottom-left):** The asset is weaker than the benchmark and it's getting weaker.
    - 🔵 **Improving** (top-right):** The asset is weaker than the benchmark, but it's getting stronger.

    /// details | References: 
    - [RRG Weights - Optuma Whitepaper](https://www.optuma.com/wp-content/uploads/2023/02/RRG-Weights.pdf)
    - [StockCharts – Relative Rotation Graphs (RRG)]([https://school.stockcharts.com/doku.php?id=chart_analysis:rrg_charts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rrg-relative-strength?utm_source=chatgpt.com))
    - [Investopedia – Relative Rotation Graph (RRG)]([https://www.investopedia.com/terms/r/relative-rotation-graph-rrg.asp](https://www.investopedia.com/relative-rotation-graph-8418457))
    - [Official RRG educational page]([https://relativerotationgraphs.com/blog/](https://relativerotationgraphs.com/educational/))

    ///

    **Author:** Marcelo Wizenberg

    **Date:** September, 2025

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Library Imports & Config""")
    return


@app.cell(hide_code=True)
def _():
    import os
    import requests
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from typing import List, Tuple, Optional
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')

    # Configure matplotlib for better plots
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10

    # Global configuration variables

    # Available resampling frequencies
    RESAMPLES = ['D', 'W', 'M']

    # Available asset universes
    UNIVERSE = ['Brazilian Stocks', 'US Stocks', 'US Sectors', 'World iShares MSCI ETFs']

    # Fallback tickers for different universes
    FALLBACK_TICKERS = {
        'Brazilian Stocks': [
            'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
            'BBAS3.SA', 'WEGE3.SA', 'MGLU3.SA', 'B3SA3.SA', 'SUZB3.SA',
            'RENT3.SA', 'LREN3.SA', 'JBSS3.SA', 'EMBR3.SA', 'CIEL3.SA',
            'HAPV3.SA', 'RADL3.SA', 'UGPA3.SA', 'CCRO3.SA', 'GGBR4.SA',
            'CSNA3.SA', 'USIM5.SA', 'GOAU4.SA', 'SBSP3.SA', 'ELET3.SA',
            'CMIG4.SA', 'TAEE11.SA', 'CSAN3.SA', 'RAIL3.SA', 'AZUL4.SA',
            'GOLL4.SA', 'CVCB3.SA', 'FLRY3.SA', 'QUAL3.SA', 'PCAR3.SA',
            'COGN3.SA', 'YDUQ3.SA', 'MRFG3.SA', 'BEEF3.SA', 'SMTO3.SA',
            'JHSF3.SA', 'EZTC3.SA', 'MRVE3.SA', 'CYRE3.SA', 'EVEN3.SA',
            'MULT3.SA', 'GFSA3.SA', 'TCSA3.SA', 'TOTS3.SA', 'VIVT3.SA'
        ],
        'US Stocks': [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 
            'JNJ', 'PG', 'UNH', 'HD', 'BAC', 'XOM', 'DIS', 'ADBE', 'CRM', 
            'NFLX', 'KO', 'PFE', 'ABBV', 'TMO', 'COST', 'AVGO', 'WMT'
        ],
        'US Sectors': [
            'XLE',   # Energy
            'XLU',   # Utilities  
            'XLB',   # Materials
            'XLP',   # Consumer Staples
            'XLK',   # Technology
            'XLV',   # Healthcare
            'XLI',   # Industrials
            'XLF',   # Financials
            'XLY'    # Consumer Discretionary
        ],
        'World iShares MSCI ETFs': [
            'EWA',   # Australia
            'EWC',   # Canada
            'EWD',   # Sweden
            'EWG',   # Germany
            'EWH',   # Hong Kong
            'EWI',   # Italy
            'EWJ',   # Japan
            'EWK',   # Belgium
            'EWL',   # Switzerland
            'EWM',   # Malaysia
            'EWN',   # Netherlands
            'EWO',   # Austria
            'EWP',   # Spain
            'EWQ',   # France
            'EWS',   # Singapore
            'EWU',   # United Kingdom
            'EWW',   # Mexico
            'EWT',   # Taiwan
            'EWY',   # South Korea
            'EWZ',   # Brazil
            'EZA'    # South Africa
        ]
    }

    # Available benchmarks
    BENCHMARKS = {
        'Brazilian Stocks': ['^BVSP'], # Ibovespa
        'US Stocks': ['^GSPC'], # S&P 500
        'US Sectors': ['^GSPC'], # S&P 500
        'World iShares MSCI ETFs': ['XWD.TO'] # iShares MSCI World
    }
    SMOOTHING_WINDOW = 10  # Rolling window for smoothing RS calculations
    TRAIL = 12      # Number of periods to show in the trail

    # Available resampling frequencies
    RESAMPLES = ['D', 'W', 'M']

    # Default date range (2 years back from today)
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    # Export files
    # OUTPUT_DIR = r'/data/rrg/{END_DATE}'
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    return (
        BENCHMARKS,
        END_DATE,
        FALLBACK_TICKERS,
        List,
        Optional,
        RESAMPLES,
        SMOOTHING_WINDOW,
        START_DATE,
        TRAIL,
        Tuple,
        UNIVERSE,
        np,
        patches,
        pd,
        plt,
        requests,
        yf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Input Parameters""")
    return


@app.cell(hide_code=True)
def _(UNIVERSE, mo):
    universe_dropdown = mo.ui.dropdown(options=UNIVERSE, value='Brazilian Stocks', label='Select universe of tickers')
    return (universe_dropdown,)


@app.cell(hide_code=True)
def _(BENCHMARKS, RESAMPLES, SMOOTHING_WINDOW, TRAIL, mo, universe_dropdown):
    benchmark = mo.ui.text(value=BENCHMARKS[universe_dropdown.value][0], placeholder='Benchmark')
    fallback_switch = mo.ui.switch(label='Fallback tickers for Brazilian Stocks',)
    frequency_dropdown = mo.ui.dropdown(options=RESAMPLES, value='W', label='Select timeframe')
    smoothing_window_slider = mo.ui.slider(start=1, stop=63, label='Smoothing window', value=SMOOTHING_WINDOW)
    trail_slider = mo.ui.slider(start=3, stop=20, label='Trail', value=TRAIL)

    mo.hstack([
        mo.vstack([
            universe_dropdown, 
            fallback_switch,
            benchmark
        
        ]),
        mo.vstack([
            frequency_dropdown,
            smoothing_window_slider,
            trail_slider
        ]),
    ])
    return (
        benchmark,
        fallback_switch,
        frequency_dropdown,
        smoothing_window_slider,
        trail_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Get appropriate tickers based on universe""")
    return


@app.cell(hide_code=True)
def _(FALLBACK_TICKERS, List, UNIVERSE, fallback_switch, requests):
    def get_asset_tickers(universe: str, limit: int = 50) -> List[str]:
        """
        Get asset tickers based on the specified universe
    
        Args:
            universe: Asset universe ('Brazilian Stocks', 'US Stocks', 'US Sectors', 'World iShares MSCI ETFs')
            limit: Maximum number of assets to fetch (only applies to 'Brazilian Stocks')
        
        Returns:
            List of asset tickers
        """
        if universe not in UNIVERSE:
            raise ValueError(f"Universe must be one of: {UNIVERSE}")
    
        if universe == 'Brazilian Stocks' and not fallback_switch.value:
            return fetch_brazilian_stocks(limit)
        elif universe == 'US Stocks':
            print(f"Using US Stocks fallback list ({len(FALLBACK_TICKERS[universe])} tickers)")
            return FALLBACK_TICKERS[universe][:limit]
        elif universe == 'US Sectors':
            print(f"Using US Sectors MSCI ETFs ({len(FALLBACK_TICKERS[universe])} sectors)")
            return FALLBACK_TICKERS[universe]
        elif universe == 'World iShares MSCI ETFs':
            print(f"Using World iShares MSCI ETFs ({len(FALLBACK_TICKERS[universe])} countries)")
            return FALLBACK_TICKERS[universe]
        else:
            return FALLBACK_TICKERS[universe][:limit]

    def fetch_brazilian_stocks(limit: int = 50) -> List[str]:
        """
        Fetch Brazilian stock tickers from brapi.dev API
    
        Args:
            limit: Maximum number of stocks to fetch
        
        Returns:
            List of stock tickers formatted for Yahoo Finance
        """
        print("Fetching Brazilian stock tickers from brapi.dev...")
    
        try:
            # Try to fetch without authentication first (public endpoint)
            url = "https://brapi.dev/api/quote/list?type=stock&sortBy=market_cap_basic&sortOrder=desc"
            response = requests.get(url, timeout=10)
        
            if response.status_code == 200:
                data = response.json()
                if 'stocks' in data:
                    tickers = [stock['stock'] for stock in data['stocks'] if not stock['stock'].endswith('F')]
                    # Limit list of tickers
                    tickers = tickers[:limit]
                    # Convert to Yahoo Finance format (add .SA suffix)
                    yahoo_tickers = [f"{ticker}.SA" for ticker in tickers]
                    print(f"Successfully fetched {len(yahoo_tickers)} stock tickers")
                    return yahoo_tickers
        
            # Fallback to predefined list if API fails
            print("API request failed, using predefined stock list...")
            return FALLBACK_TICKERS['Brazilian Stocks'][:limit]
        
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            print("Using fallback ticker list...")
            return FALLBACK_TICKERS['Brazilian Stocks'][:limit]        
    return (get_asset_tickers,)


@app.cell(hide_code=True)
def _(get_asset_tickers, universe_dropdown):
    if universe_dropdown.value == 'US Sectors':
        tickers = get_asset_tickers(universe_dropdown.value)
    else:
        tickers = get_asset_tickers(universe_dropdown.value, limit=25)
    print('Tickers:', tickers)
    return (tickers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. RRG Class""")
    return


@app.cell(hide_code=True)
def _(
    END_DATE,
    List,
    Optional,
    START_DATE,
    Tuple,
    np,
    patches,
    pd,
    plt,
    tickers,
    yf,
):
    class RRG:
        """
        Relative Rotation Graph Implementation
    
        This class creates RRG visualizations showing relative strength and momentum 
        of any list of assets against a specified benchmark.
        """
    
        def __init__(self, 
                     tickers: List[str],
                     benchmark: str,
                     frequency: str, 
                     smoothing_window: int,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     trail: int = 12):
            """
            Initialize the RRG analyzer
        
            Args:
                tickers: List of asset tickers to analyze
                benchmark: Benchmark ticker for comparison
                frequency: Data frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
                lookback_periods: Number of periods for RS calculations (default: 52)
                start_date: Start date in YYYY-MM-DD format (default: 2 years ago)
                end_date: End date in YYYY-MM-DD format (default: today)
                trail: Number of steps to show in the trail (default: 12)
            """
            self.tickers = tickers
            self.benchmark = benchmark
            self.frequency = frequency.upper()
            self.smoothing_window = smoothing_window
            self.start_date = start_date or START_DATE
            self.end_date = end_date or END_DATE
            self.trail = trail

            # Validate frequency
            if self.frequency not in ['D', 'W', 'M']:
                raise ValueError("Frequency must be 'D' (daily), 'W' (weekly), or 'M' (monthly)")

            # Set lookback periods based on frequency
            if frequency == 'D':
                self.lookback_periods = 252  # ~1 year of trading days
            elif frequency == 'W':
                self.lookback_periods = 52   # 52 weeks
            else:  # Monthly
                self.lookback_periods = 12   # 12 months

            self.stocks_data: Optional[pd.DataFrame] = None
            self.benchmark_data: Optional[pd.Series] = None
            self.rrg_data: Optional[pd.DataFrame] = None

        def fetch_price_data(self) -> pd.DataFrame:
            """
            Fetch historical price data for stocks and benchmark
        
            Returns:
                DataFrame with resampled adjusted close prices
            """
            # print(f"Fetching price data for {len(self.tickers)} tickers...")
            # print(f"Tickers: {','.join(self.tickers)}")
            # print(f"Benchmark: {self.benchmark}")
            # print(f"Date range: {self.start_date} to {self.end_date}")
            # print(f"Frequency: {self.frequency}")
        
            # Add benchmark to ticker list
            all_tickers = tickers + [self.benchmark]
        
            # Fetch data
            data = yf.download(all_tickers, start=self.start_date, end=self.end_date, 
                              interval='1d', auto_adjust=True, progress=False)
        
            if len(all_tickers) == 1:
                # Handle single ticker case
                prices = data['Close'].to_frame()
                prices.columns = all_tickers
            else:
                # Handle multiple tickers
                prices = data['Close']
        
            # Resample based on frequency
            if self.frequency == 'D':
                resampled_prices = prices  # Keep daily data as is
            elif self.frequency == 'W':
                resampled_prices = prices.resample('W-FRI').last()  # Weekly (Friday close)
            elif self.frequency == 'M':
                resampled_prices = prices.resample('M').last()  # Monthly (end of month)
        
            # Drop rows with all NaN values
            resampled_prices = resampled_prices.dropna(how='all')
        
            # Forward fill missing values and then drop remaining NaN
            resampled_prices = resampled_prices.fillna(method='ffill').dropna()
        
            # print(f"Data shape after processing: {resampled_prices.shape}")
            # print(f"Date range: {resampled_prices.index[0]} to {resampled_prices.index[-1]}")
        
            return resampled_prices
    
        def calculate_rs_ratio(self, stock_prices: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
            """
            Calculate RS-Ratio (Relative Strength Ratio)
        
            The RS-Ratio measures the trend of relative performance using a momentum-based
            approach normalized around 100.
        
            Args:
                stock_prices: Stock price series
                benchmark_prices: Benchmark price series
            
            Returns:
                RS-Ratio series normalized around 100
            """
            # Calculate price relative (stock / benchmark)
            price_relative = stock_prices / benchmark_prices
        
            # Calculate RS-Ratio using rate of change over lookback period
            # This is a simplified version - the actual JdK formula is proprietary
            rs_raw = price_relative.pct_change(periods=self.lookback_periods//4) * 100
        
            # Normalize around 100 using a rolling mean approach
            rs_ratio = 100 + rs_raw.rolling(window=self.smoothing_window, min_periods=1).mean()
        
            return rs_ratio #.fillna(100)
    
        def calculate_rs_momentum(self, rs_ratio: pd.Series) -> pd.Series:
            """
            Calculate RS-Momentum (Rate of change of RS-Ratio)
        
            RS-Momentum measures the momentum of the RS-Ratio, normalized around 100.
        
            Args:
                rs_ratio: RS-Ratio series
            
            Returns:
                RS-Momentum series normalized around 100
            """
            # Calculate rate of change of RS-Ratio
            rs_momentum_raw = rs_ratio.pct_change(periods=4) * 100
        
            # Normalize around 100
            rs_momentum = 100 + rs_momentum_raw.rolling(window=2, min_periods=1).mean()
        
            return rs_momentum #.fillna(100)
    
        def calculate_rrg_data(self, prices_df: pd.DataFrame) -> pd.DataFrame:
            """
            Calculate RRG data (RS-Ratio and RS-Momentum) for all stocks
        
            Args:
                prices_df: DataFrame with stock and benchmark prices
            
            Returns:
                DataFrame with RS-Ratio and RS-Momentum for each stock
            """
            print("Calculating RRG metrics...")
        
            benchmark_prices = prices_df[self.benchmark]
            stock_columns = [col for col in prices_df.columns if col != self.benchmark]
        
            rrg_results = {}
        
            for stock in stock_columns:
                if stock in prices_df.columns:
                    stock_prices = prices_df[stock]
                
                    # Skip if not enough data
                    if len(stock_prices.dropna()) < self.lookback_periods//2:
                        continue
                
                    # Calculate RS-Ratio and RS-Momentum
                    rs_ratio = self.calculate_rs_ratio(stock_prices, benchmark_prices)
                    rs_momentum = self.calculate_rs_momentum(rs_ratio)
                
                    # Store results
                    for i, date in enumerate(prices_df.index):
                        if date not in rrg_results:
                            rrg_results[date] = {}
                    
                        rrg_results[date][stock] = {
                            'rs_ratio': rs_ratio.iloc[i] if i < len(rs_ratio) else np.nan,
                            'rs_momentum': rs_momentum.iloc[i] if i < len(rs_momentum) else np.nan
                        }
        
            # Convert to DataFrame format
            rrg_df_data = []
            for date, stocks_data in rrg_results.items():
                for stock, metrics in stocks_data.items():
                    rrg_df_data.append({
                        'date': date,
                        'stock': stock,
                        'rs_ratio': metrics['rs_ratio'],
                        'rs_momentum': metrics['rs_momentum']
                    })
        
            rrg_df = pd.DataFrame(rrg_df_data)
            rrg_df = rrg_df.dropna()
        
            print(f"RRG data calculated for {len(rrg_df['stock'].unique())} stocks")
        
            return rrg_df
    
        def get_quadrant_info(self, rs_ratio: float, rs_momentum: float) -> Tuple[str, str]:
            """
            Determine which quadrant a stock is in based on RS-Ratio and RS-Momentum
        
            Args:
                rs_ratio: RS-Ratio value
                rs_momentum: RS-Momentum value
            
            Returns:
                Tuple of (quadrant_name, color)
            """
            if rs_ratio >= 100 and rs_momentum >= 100:
                return "Leading", "green"
            elif rs_ratio < 100 and rs_momentum >= 100:
                return "Improving", "blue"
            elif rs_ratio < 100 and rs_momentum < 100:
                return "Lagging", "red"
            else:  # rs_ratio >= 100 and rs_momentum < 100
                return "Weakening", "gold"
    
        def plot_rrg_chart(self, save_path: Optional[str] = None) -> None:
            """
            Create the RRG visualization with trails for the last 12 steps
        
            Args:
                save_path: Optional path to save the chart
            """
            if self.rrg_data is None:
                raise ValueError("No RRG data available. Run calculate_rrg_data first.")
        
            print("Creating RRG visualization...")
        
            # Get the last N periods of data
            latest_dates = sorted(self.rrg_data['date'].unique())[-self.trail:]
            trail_data = self.rrg_data[self.rrg_data['date'].isin(latest_dates)]
        
            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 10))
        
            # Set up the coordinate system
            ax.set_xlim(80, 120)
            ax.set_ylim(80, 120)
        
            # Add quadrant lines
            ax.axhline(y=100, color='black', linewidth=2, alpha=0.8)
            ax.axvline(x=100, color='black', linewidth=2, alpha=0.8)
        
            # Leading quadrant (top-right)
            ax.add_patch(patches.Rectangle((100, 100), 20, 20, 
                                         facecolor='green', alpha=0.1))
        
            # Improving quadrant (top-left) - Blue background
            ax.add_patch(patches.Rectangle((80, 100), 20, 20, 
                                         facecolor='blue', alpha=0.1))
        
            # Lagging quadrant (bottom-left)
            ax.add_patch(patches.Rectangle((80, 80), 20, 20, 
                                         facecolor='red', alpha=0.1))
        
            # Weakening quadrant (bottom-right) - Yellow background
            ax.add_patch(patches.Rectangle((100, 80), 20, 20, 
                                         facecolor='yellow', alpha=0.1))
        
            # Add quadrant labels
            ax.text(110, 110, 'Leading', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='green')
            ax.text(90, 110, 'Improving', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='blue')
            ax.text(90, 90, 'Lagging', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='red')
            ax.text(110, 90, 'Weakening', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='gold')
        
            # Plot trails for each stock
            stocks = trail_data['stock'].unique()
        
            for stock in stocks:
                stock_data = trail_data[trail_data['stock'] == stock].sort_values('date')
            
                if len(stock_data) < 2:
                    continue
            
                # Get the trail coordinates
                x_coords = stock_data['rs_ratio'].values
                y_coords = stock_data['rs_momentum'].values
            
                # Determine current quadrant for color
                current_rs_ratio = x_coords[-1]
                current_rs_momentum = y_coords[-1]
                quadrant, color = self.get_quadrant_info(current_rs_ratio, current_rs_momentum)
            
                # Plot the trail
                ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=1.5)
            
                # Plot trail points with increasing size (older = smaller)
                sizes = np.linspace(10, 40, len(x_coords))
                ax.scatter(x_coords[:-1], y_coords[:-1], c=color, s=sizes[:-1], 
                          alpha=0.4, edgecolors='white', linewidths=0.5)
            
                # Plot current position (larger point)
                ax.scatter(current_rs_ratio, current_rs_momentum, c=color, s=80, 
                          alpha=0.8, edgecolors='black', linewidths=1)
            
                # Add stock label for current position
                stock_label = stock.replace('.SA', '')
                ax.annotate(stock_label, (current_rs_ratio, current_rs_momentum), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, fontweight='bold')
        
            # Customize the plot
            ax.set_xlabel('RS-Ratio', fontsize=14, fontweight='bold')
            ax.set_ylabel('RS-Momentum', fontsize=14, fontweight='bold')
            ax.set_title(f'Relative Rotation Graph (RRG) - Assets vs {self.benchmark}\n'
                        f'Trail: Last {self.trail} periods ({self.frequency})', 
                        fontsize=16, fontweight='bold', pad=20)
        
            # Add grid
            ax.grid(True, alpha=0.3)
        
            # Add explanation text
            # explanation = (
            #     "RRG Quadrants:\n"
            #     "• Leading (Green): Strong performance, positive momentum\n"
            #     "• Improving (Blue): Weak performance, improving momentum\n"
            #     "• Lagging (Red): Weak performance, negative momentum\n"
            #     "• Weakening (Yellow): Strong performance, declining momentum"
            # )
        
            # ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
            #        fontsize=9, verticalalignment='top', 
            #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
            plt.tight_layout()
        
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Chart saved to: {save_path}")
        
            plt.show()
    
        def get_current_positions(self) -> pd.DataFrame:
            """
            Get current quadrant positions for all stocks
        
            Returns:
                DataFrame with current stock positions and quadrants
            """
            if self.rrg_data is None:
                raise ValueError("No RRG data available. Run calculate_rrg_data first.")
        
            # Get the most recent data
            latest_date = self.rrg_data['date'].max()
            current_data = self.rrg_data[self.rrg_data['date'] == latest_date].copy()
        
            # Add quadrant information
            quadrant_info = current_data.apply(
                lambda row: self.get_quadrant_info(row['rs_ratio'], row['rs_momentum']), 
                axis=1
            )
        
            current_data['quadrant'] = [info[0] for info in quadrant_info]
            current_data['color'] = [info[1] for info in quadrant_info]
        
            # Clean up stock names
            current_data['stock_clean'] = current_data['stock'].str.replace('.SA', '')
        
            # Sort by quadrant and RS-Ratio
            quadrant_order = ['Leading', 'Improving', 'Lagging', 'Weakening']
            current_data['quadrant_order'] = current_data['quadrant'].map(
                {q: i for i, q in enumerate(quadrant_order)}
            )
        
            result = current_data.sort_values(['quadrant_order', 'rs_ratio'], 
                                            ascending=[True, False])
        
            return result[['stock_clean', 'rs_ratio', 'rs_momentum', 'quadrant']]
    
        def run_analysis(self, num_stocks: int = 30) -> None:
            """
            Run the complete RRG analysis
        
            Args:
                num_stocks: Number of stocks to analyze
            """
            print("=" * 60)
            print("RELATIVE ROTATION GRAPH (RRG) ANALYSIS")
            print("=" * 60)
            print(f"Assets: {len(self.tickers)}")
            print(f"Benchmark: {self.benchmark}")
            print(f"Frequency: {self.frequency}")
            print(f"Date range: {self.start_date} to {self.end_date}")
        
            # Step 1: Fetch stock tickers
            price_data = self.fetch_price_data()
            print(f"Selected {len(tickers)} stocks for analysis")
        
            # Step 2: Fetch price data
            price_data = self.fetch_price_data()
            self.stocks_data = price_data
        
            # Step 3: Calculate RRG metrics
            self.rrg_data = self.calculate_rrg_data(price_data)
        
            # Step 4: Display current positions
            print("\nCurrent Stock Positions:")
            print("-" * 40)
            current_positions = self.get_current_positions()
        
            for quadrant in ['Leading', 'Improving', 'Lagging', 'Weakening']:
                stocks_in_quadrant = current_positions[
                    current_positions['quadrant'] == quadrant
                ]
                print(f"\n{quadrant.upper()} ({len(stocks_in_quadrant)} stocks):")
                for _, row in stocks_in_quadrant.iterrows():
                    print(f"  {row['stock_clean']:8} | RS-Ratio: {row['rs_ratio']:6.1f} | "
                         f"RS-Momentum: {row['rs_momentum']:6.1f}")
        
            # Step 5: Create visualization
            print("\nGenerating RRG chart...")
            self.plot_rrg_chart()
        
            print("\nAnalysis complete!")
    return (RRG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run Analysis""")
    return


@app.cell(hide_code=True)
def _(
    RRG,
    benchmark,
    frequency_dropdown,
    smoothing_window_slider,
    tickers,
    trail_slider,
):
    # Initialize and run analysis
    rrg_analyzer = RRG(
        tickers=tickers,
        benchmark=benchmark.value,
        frequency=frequency_dropdown.value,
        smoothing_window=smoothing_window_slider.value,
        trail=trail_slider.value
    )

    rrg_analyzer.run_analysis()
    return


if __name__ == "__main__":
    app.run()
