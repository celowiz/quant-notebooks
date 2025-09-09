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
    # Brazil Core Inflation Analysis

    This notebook analyzes Brazil's core inflation measures using data from the Central Bank of Brazil (BCB) and IBGE.

    **Macroeconomic Context:**
    Core inflation measures are essential tools for monetary policy as they exclude volatile components like food and energy prices, providing a clearer view of underlying inflationary pressures. Brazil uses multiple core inflation indicators to assess price stability and guide interest rate decisions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 0. Import Libraries

    We'll use:

    - **bcb**: To access Brazilian Central Bank time series data
    - **pandas/numpy**: For data manipulation and analysis
    - **matplotlib/seaborn**: For data visualization
    - **statsmodels**: For time series decomposition
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # Import required libraries
    from bcb import sgs  # Brazilian Central Bank API
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from pandas import Series

    # Define color palette for consistent visualization
    colors = {
        'blue': '#282f6b', 
        'red': '#b22200',
        'green': '#224f20',
        'purple': '#5f487c',
        'gray': '#666666',
        'orange': '#b35c1e'
    }
    return colors, np, pd, plt, sgs, sm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Extract Data

    **Core Inflation Measures in Brazil:**

    Brazil's Central Bank tracks several core inflation indicators:

    - **EX0**: IPCA excluding food at home and monitored prices
    - **EX3**: IPCA excluding food and energy
    - **DP**: Double-weighted IPCA (Dupla Ponderação)
    - **MS**: Trimmed mean IPCA (Média Aparada)
    - **P55**: 55% trimmed mean IPCA
    - **IPCA**: Full Consumer Price Index for comparison

    These measures help identify persistent inflation trends by removing temporary price shocks.
    """
    )
    return


@app.cell(hide_code=True)
def _(sgs):
    # Fetch core inflation time series from BCB using SGS codes
    core_inflation_df = sgs.get({
        'ex0': 11427,    # IPCA excluding food at home and monitored prices
        'ex3': 27839,    # IPCA excluding food and energy
        'dp': 16122,     # Double-weighted IPCA
        'ms': 4466,      # Trimmed mean IPCA
        'p55': 28750,    # 55% trimmed mean IPCA
        'ipca': 433      # Full IPCA for comparison
    }, start='2002-01-01')

    core_inflation_df
    return (core_inflation_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Transform Data

    **Creating a Composite Core Inflation Measure:**

    We calculate the average of all core inflation measures to create a single, robust indicator. This approach:

    - Reduces noise from individual measures
    - Provides a more stable signal of underlying inflation
    - Helps identify seasonal patterns in price behavior

    We also analyze monthly seasonality to understand when inflation typically peaks or troughs during the year.
    """
    )
    return


@app.cell(hide_code=True)
def _(core_inflation_df, pd):
    # Create working copy and calculate composite core inflation measure
    core_data = core_inflation_df.copy()

    # Calculate mean of all core measures (excluding headline IPCA)
    core_data['mean_core_inflation'] = core_data.drop('ipca', axis=1).mean(axis=1)

    # Extract month names for seasonality analysis
    core_data['month'] = core_data.index.month_name()

    # Calculate monthly statistics for seasonality analysis
    monthly_mean = core_data.groupby('month')['mean_core_inflation'].mean()
    monthly_p20 = core_data.groupby('month')['mean_core_inflation'].quantile(0.2)
    monthly_p80 = core_data.groupby('month')['mean_core_inflation'].quantile(0.8)

    # Combine monthly statistics into results dataframe
    monthly_results = pd.DataFrame({
        'month': monthly_mean.index,
        'monthly_mean': monthly_mean.values,
        'percentile_20': monthly_p20.values,
        'percentile_80': monthly_p80.values
    })

    monthly_results
    return core_data, monthly_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Current Year Analysis

    **Recent Performance Context:**

    Analyzing the current year's inflation data helps assess:

    - How current inflation compares to historical seasonal patterns
    - Whether recent price pressures are within normal ranges
    - If monetary policy adjustments may be needed

    This comparison is crucial for central bank communication and market expectations.
    """
    )
    return


@app.cell(hide_code=True)
def _(core_data):
    # Extract current year data for comparison with historical patterns
    current_year_data = (
        core_data
        .query('index >= "2025-01-01"')
        .reset_index()
        [['month', 'mean_core_inflation', 'ipca']]
    )
    return (current_year_data,)


@app.cell(hide_code=True)
def _(current_year_data, monthly_results, pd):
    # Merge historical monthly patterns with current year data
    combined_monthly_data = pd.merge(
        left=monthly_results, 
        right=current_year_data, 
        how='outer'
    )
    combined_monthly_data
    return (combined_monthly_data,)


@app.cell(hide_code=True)
def _(combined_monthly_data, pd):
    # Sort months in chronological order and format for display
    month_order = sorted(
        combined_monthly_data['month'].unique(), 
        key=lambda x: pd.to_datetime(x, format='%B')
    )

    # Create ordered categorical for proper sorting
    combined_monthly_data['month'] = pd.Categorical(
        combined_monthly_data['month'], 
        categories=month_order, 
        ordered=True
    )
    combined_monthly_data.sort_values(by='month', inplace=True)

    # Create abbreviated month names for charts
    combined_monthly_data['month_abbrev'] = combined_monthly_data['month'].str.slice(stop=3)

    combined_monthly_data
    return


@app.cell(hide_code=True)
def _(np):
    def calculate_cumulative_inflation(data, window: int):
        """
        Calculate cumulative inflation over rolling windows.

        This function compounds monthly inflation rates to show accumulated 
        inflation over specified periods, which is essential for:
        - Annual inflation targeting (12-month windows)
        - Policy assessment (shorter-term trends)

        Args:
            data: Series of monthly inflation rates as percentages
            window: Number of periods to accumulate over

        Returns:
            Series with cumulative inflation rates as percentages
        """
        return (((data / 100) + 1)
                .rolling(window=window)
                .apply(np.prod)
                - 1) * 100
    return (calculate_cumulative_inflation,)


@app.cell(hide_code=True)
def _(core_data, sm):
    # Perform seasonal decomposition to isolate trend from seasonal effects
    decomposition = sm.tsa.seasonal_decompose(
        core_data.mean_core_inflation, 
        model='additive'
    )

    # Create seasonally adjusted series by removing seasonal component
    core_data['core_inflation_sa'] = (
        core_data.mean_core_inflation.values - decomposition.seasonal.values
    )
    return (decomposition,)


@app.cell(hide_code=True)
def _(calculate_cumulative_inflation, core_data):
    # Calculate key inflation metrics for policy analysis
    inflation_metrics = (
        core_data
        .assign(
            # 12-month accumulated inflation (standard policy measure)
            core_12m_cumulative = lambda x: calculate_cumulative_inflation(
                x.mean_core_inflation, window=12
            ),
            # 3-month annualized moving average (shows recent momentum)
            core_3m_annualized = lambda x: (
                x.core_inflation_sa.rolling(window=3).mean() * 12
            )
        )
        .reset_index()
        .dropna()  # Remove NaN values as per project specification
    )

    inflation_metrics
    return (inflation_metrics,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Visualization - Core Inflation Trends

    **Interpreting the Chart:**

    - **Blue line (12M Cumulative)**: Shows inflation over the past 12 months
    - **Orange line (3M Annualized)**: Shows recent inflation momentum

    **Policy Implications:**

    - When lines converge: Inflation is stabilizing
    - When 3M is above 12M: Inflation is accelerating
    - When 3M is below 12M: Inflation is decelerating
    """
    )
    return


@app.cell(hide_code=True)
def _(colors, inflation_metrics, pd, plt):
    # Transform data to long format for plotting
    metrics_long = inflation_metrics[[
        'Date', 'core_12m_cumulative', 'core_3m_annualized'
    ]].melt(id_vars=['Date'])

    # Create the main inflation trends chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each metric with appropriate styling
    for variable, group_data in metrics_long.groupby('variable'):
        if variable == 'core_12m_cumulative':
            color = colors['blue']
            label = '12M Cumulative'
        else:
            color = colors['orange']
            label = '3M Annualized'

        plt.plot(
            group_data['Date'], 
            group_data['value'], 
            label=label, 
            linewidth=2,
            color=color
        )

    # Format x-axis with 2-year intervals
    date_ticks = pd.date_range(
        start=metrics_long['Date'].min(), 
        end=metrics_long['Date'].max(), 
        freq='2YS'
    )
    plt.xticks(
        ticks=date_ticks,
        labels=date_ticks.strftime('%b/%y'),
        rotation=45
    )

    # Chart formatting and labels
    plt.title("Brazil Core Inflation Average", fontsize=16, fontweight='bold')
    plt.suptitle("12-Month Cumulative and 3-Month Annualized (SAAR)", fontsize=12)
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate (%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add source note
    plt.figtext(
        0.99, 0.01, 
        "Source: IBGE and BCB | Analysis: quantitative research", 
        horizontalalignment='right', 
        fontsize=10, 
        color='gray'
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Seasonal Analysis

    **Understanding Inflation Seasonality:**

    Seasonal patterns in inflation help policymakers:

    - Distinguish between seasonal effects and underlying trends
    - Time policy announcements and interventions
    - Set appropriate expectations for specific months


    **Reading the Chart:**

    - **Gray band**: Historical range (20th-80th percentile)
    - **Black line**: Long-term monthly averages
    - **Colored lines**: Current year vs historical patterns
    """
    )
    return


@app.cell(hide_code=True)
def _(colors, combined_monthly_data, plt):
    # Set up the seasonal analysis chart
    plt.figure(figsize=(12, 6))

    # Create x-axis positions for months
    x_positions = range(len(combined_monthly_data))

    # Plot the uncertainty band (20th-80th percentile range)
    plt.fill_between(
        x_positions,
        combined_monthly_data['percentile_20'], 
        combined_monthly_data['percentile_80'], 
        color='lightgray', 
        alpha=0.5, 
        label='Historical Range (P20-P80)'
    )

    # Plot historical monthly averages
    plt.plot(
        x_positions,
        combined_monthly_data['monthly_mean'], 
        color='black', 
        linewidth=2, 
        marker='o', 
        markersize=4,
        label='Historical Average'
    )

    # Plot current year core inflation if available
    if 'mean_core_inflation' in combined_monthly_data.columns:
        current_mask = combined_monthly_data['mean_core_inflation'].notna()
        if current_mask.any():
            current_positions = [i for i, mask in enumerate(current_mask) if mask]
            plt.plot(
                current_positions,
                combined_monthly_data.loc[current_mask, 'mean_core_inflation'],
                color=colors['blue'], 
                linewidth=2.5, 
                marker='s', 
                markersize=6,
                label='2024 Core Inflation'
            )

    # Plot current year headline IPCA if available
    if 'ipca' in combined_monthly_data.columns:
        ipca_mask = combined_monthly_data['ipca'].notna()
        if ipca_mask.any():
            ipca_positions = [i for i, mask in enumerate(ipca_mask) if mask]
            plt.plot(
                ipca_positions,
                combined_monthly_data.loc[ipca_mask, 'ipca'],
                color=colors['red'], 
                linewidth=2.5, 
                marker='^', 
                markersize=6,
                label='2024 Headline IPCA'
            )

    # Chart formatting
    plt.title(
        'Monthly Core Inflation Seasonality Analysis (2002-2024)', 
        fontsize=14, 
        fontweight='bold'
    )
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Monthly Inflation Rate (%)', fontsize=12)

    # Set month labels on x-axis
    plt.xticks(
        x_positions, 
        combined_monthly_data['month_abbrev'], 
        rotation=0
    )

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Seasonal Decomposition Analysis

    **Understanding the Components:**

    This analysis shows how core inflation can be decomposed into:
    - **Original Series**: Raw core inflation data
    - **Seasonally Adjusted**: Core inflation with seasonal effects removed
    - **Seasonal Component**: The recurring seasonal pattern

    The seasonally adjusted series is crucial for policy analysis as it reveals the underlying inflation trend without seasonal noise.
    """
    )
    return


@app.cell(hide_code=True)
def _(colors, core_data, decomposition, plt):
    # Create a comprehensive decomposition chart
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Core Inflation Time Series Decomposition', fontsize=16, fontweight='bold')

    # Original series
    axes[0, 0].plot(core_data.index, core_data['mean_core_inflation'], 
                    color=colors['blue'], linewidth=1.5)
    axes[0, 0].set_title('Original Core Inflation Series')
    axes[0, 0].set_ylabel('Inflation Rate (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # Trend component
    axes[0, 1].plot(core_data.index, decomposition.trend, 
                    color=colors['green'], linewidth=1.5)
    axes[0, 1].set_title('Trend Component')
    axes[0, 1].set_ylabel('Inflation Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Seasonal component
    axes[1, 0].plot(core_data.index, decomposition.seasonal, 
                    color=colors['orange'], linewidth=1.5)
    axes[1, 0].set_title('Seasonal Component')
    axes[1, 0].set_ylabel('Seasonal Effect (%)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].grid(True, alpha=0.3)

    # Seasonally adjusted series
    axes[1, 1].plot(core_data.index, core_data['core_inflation_sa'], 
                    color=colors['purple'], linewidth=1.5)
    axes[1, 1].set_title('Seasonally Adjusted Series')
    axes[1, 1].set_ylabel('Inflation Rate (%)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)

    # Format all x-axes
    for _ax in axes.flat:
        _ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
