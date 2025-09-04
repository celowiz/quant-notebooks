import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    green_text = '<strong><span style=\"color:green;\">benefit</span></strong>'
    red_text = '<strong><span style=\"color:red;\">hurt</span></strong>'
    return green_text, red_text


@app.cell(hide_code=True)
def _(green_text, mo, red_text):
    mo.md(
        rf"""
    # BRL Sensitivity Study — Brazilian Equities vs. BRL Strength (Marimo)


    ## Goal:
    This interactive notebook analyzes how Brazilian stocks co-move with the Brazilian Real (BRL), using historical daily data from Yahoo Finance. We derive a **sensitivity metric** (beta from linear regression on returns), rolling correlations, and rank which stocks historically {green_text} or are {red_text} the most when BRL **strengthens** (i.e., **BRLUSD** rises because we invert USD/BRL).


    ## Pipeline:

    - **Fetch tickers** of B3-listed stocks (via public API fallback strategy) or load from CSV.
    - **Map tickers** to Yahoo Finance symbols by appending **`.SA`**.
    - **Download historical prices** (adjusted close) for each stock and **USD/BRL** (`BRL=X`), then compute **BRLUSD = 1/(USD/BRL)**.
    - **Visualize** rolling returns (stock vs. BRLUSD) and rolling correlations.
    - **Compute statistics**: rolling correlation (windowed), linear regression **beta** (stock returns on BRLUSD returns), **R²**, and p-values.
    - **Rank** top beneficiaries (highest beta) and most hurt (lowest beta) and **export** results to Excel.


    ## Notes:

    - This notebook is modular: each function resides in its own cell. All code comments and explanations are in English.
    - Use the controls below to customize: **lookback window**, number of plots, ranking size, data source, etc.
    - Export is written to: `root/data/sensitivity_brl/{{YYYY-MM-DD}}/`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Import Libraries & Define Variables""")
    return


@app.cell
def _():
    import os
    import math
    from dataclasses import dataclass
    from typing import List, Tuple, Dict


    import numpy as np
    import pandas as pd
    import requests
    import yfinance as yf
    from scipy import stats
    from tqdm import tqdm
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import date


    YAHOO_SUFFIX = '.SA'
    YAHOO_USDBRL = 'BRL=X' # USD per 1 unit is not BRL; here it's BRL per USD; we invert.
    BRAPI_AVAILABLE_URL = 'https://brapi.dev/api/available'
    START_DATE = '2015-01-01'
    END_DATE = date.today().strftime('%Y-%m-%d')
    OUTDIR = f'data/sensitivity_brl/{END_DATE}'
    FALLBACK_TICKERS = [
        'PETR4','VALE3','ITUB4','BBDC4','ABEV3','BBAS3','B3SA3','WEGE3','SUZB3',
        'GGBR4','EMBR3','MGLU3','LREN3','KLBN11','EQTL3', 'ELET6','PRIO3','RDOR3'
    ]
    return (
        BRAPI_AVAILABLE_URL,
        END_DATE,
        FALLBACK_TICKERS,
        OUTDIR,
        START_DATE,
        YAHOO_SUFFIX,
        YAHOO_USDBRL,
        dataclass,
        go,
        make_subplots,
        np,
        os,
        pd,
        px,
        requests,
        stats,
        tqdm,
        yf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Input Parameters""")
    return


@app.cell
def _(mo):
    source = mo.ui.dropdown(options=['brapi', 'fallback'], value='fallback', label='Ticker source')
    lookback = mo.ui.slider(21, 504, value=126, step=1, label='Rolling window (trading days)')
    max_plots = mo.ui.slider(2, 80, value=24, label='Max plots for grid (stock vs BRLUSD)')
    ncols = mo.ui.slider(1, 4, value=2, label='Columns in grid plot')
    min_obs = mo.ui.slider(60, 500, value=126, label='Min overlapping observations')
    min_r2 = mo.ui.slider(0.0, 1.0, value=0.0, step=0.01, label='Min R² filter (for ranking display)')

    mo.hstack([
        mo.vstack([source, max_plots, ncols]),
        mo.vstack([lookback, min_obs, min_r2])
    ])
    return lookback, max_plots, min_obs, min_r2, ncols, source


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Fetch Available Tickers""")
    return


@app.cell
def _(BRAPI_AVAILABLE_URL, requests):
    def fetch_tickers_brapi(timeout: int = 20):
        """Fetch available B3 tickers from brapi.dev. Filters out BDRs, fractions and funds.
        Returns a list like ['PETR4', 'VALE3', ...]."""
        try:
            r = requests.get(BRAPI_AVAILABLE_URL, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            symbols = data.get('stocks', []) or data.get('availableStocks', [])
            cleaned = []
            for s in symbols:
                s = s.strip().upper()
                if any(s.endswith(suf) for suf in ('3', '4', '5', '6', '11')) and not any(x in s for x in ('F', '11', '34')):
                    cleaned.append(s)
            cleaned += ['BOVA11', 'SMAL11', 'BPAC11', 'SANB11'] # add Units and Index ETFs
            return sorted(set(cleaned))
        except Exception:
            return []
        return cleaned

    available_tickers = fetch_tickers_brapi()
    return (fetch_tickers_brapi,)


@app.cell
def _(FALLBACK_TICKERS, fetch_tickers_brapi):
    def get_tickers(source: str = 'brapi'):
        if source.lower() == 'brapi':
            t = fetch_tickers_brapi()
            if t:
                return t
            else:
                raise RuntimeError("Failed to fetch tickers from brapi.dev")
        return FALLBACK_TICKERS
    return (get_tickers,)


@app.cell
def _(get_tickers, source, to_yahoo_symbols):
    tickers = get_tickers(source.value)
    yahoo_symbols = to_yahoo_symbols(tickers)
    print(yahoo_symbols)
    return (yahoo_symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Symbols & Downloading""")
    return


@app.cell
def _(YAHOO_SUFFIX):
    def to_yahoo_symbols(tickers):
        return [t + YAHOO_SUFFIX for t in tickers]
    return (to_yahoo_symbols,)


@app.cell
def _(pd, tqdm, yf):
    def download_close_series(symbols, batch_size: int = 50) -> dict:
        """Download adjusted close series for a list of Yahoo symbols in batches."""
        result = {}
        for i in tqdm(range(0, len(symbols), batch_size), desc="Downloading prices"):
            batch = symbols[i:i+batch_size]
            try:
                data = yf.download(batch, period="max", interval="1d", auto_adjust=True, progress=False, threads=True)
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data[('Close')]
                    for col in closes.columns:
                        s = closes[col].dropna()
                        if len(s) > 0:
                            result[col] = s
                else:
                    s = data['Close'].dropna()
                    if len(s) > 0:
                        result[batch[0]] = s
            except Exception:
                continue
        return result
    return (download_close_series,)


@app.cell
def _(download_close_series, yahoo_symbols):
    price_map = download_close_series(yahoo_symbols)
    price_map
    return (price_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Download USDBRL data & Compute Returns""")
    return


@app.cell
def _(np, pd):
    def compute_returns(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return (compute_returns,)


@app.cell
def _(END_DATE, START_DATE, YAHOO_USDBRL, compute_returns, mo, px, yf):
    fx = yf.download(YAHOO_USDBRL, start=START_DATE, end=END_DATE, interval='1d', auto_adjust=True, progress=False)
    usdbrl = fx['Close'].dropna()
    brlusd = 1.0 / usdbrl # invert to get BRL in USD terms
    brl_ret = compute_returns(brlusd)

    mo.hstack([
        mo.ui.plotly(px.line(brlusd, x=brlusd.index, y='BRL=X', title='BRL/USD')), 
        mo.ui.plotly(px.line(brl_ret, x=brl_ret.index, y='BRL=X', title='BRL Daily Returns'))
    ])
    return brl_ret, brlusd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Data Classes & Statistics""")
    return


@app.cell
def _(dataclass, pd):
    @dataclass
    class AssetData:
        symbol: str
        prices: pd.Series # adjusted close
        returns: pd.Series
    return (AssetData,)


@app.cell
def _(np, pd, stats):
    def compute_stats(assets, brlusd_ret: pd.Series, window: int = 126, min_obs: int = 126):
        """Compute rolling correlation, regression beta, R², p-value; and rolling correlations.
        Returns dict of DataFrames with columns ['beta', 'r2', 'pvalue', 'corr']."""
        rolling_results = {}

        for a in assets:
            df = pd.concat([a.returns.rename('asset'), brlusd_ret['BRL=X'].rename('brl')], axis=1).dropna()
            if len(df) < max(60, window, min_obs):
                continue

            # Calculate rolling regression metrics using a different approach
            def rolling_regression(x, y):
                """Calculate regression statistics for rolling windows"""
                try:
                    if len(x) < max(60, window, min_obs):
                        return np.nan, np.nan, np.nan
                    lr = stats.linregress(x, y)
                    return lr.slope, max(0.0, lr.rvalue ** 2), lr.pvalue
                except:
                    return np.nan, np.nan, np.nan
        
            # Apply rolling regression to each window
            beta_list = []
            r2_list = []
            pvalue_list = []
        
            for i in range(window - 1, len(df)):
                window_data = df.iloc[i - window + 1:i + 1]
                if len(window_data) >= min_obs:
                    beta, r2, pvalue = rolling_regression(window_data['brl'].values, window_data['asset'].values)
                else:
                    beta, r2, pvalue = np.nan, np.nan, np.nan
            
                beta_list.append(beta)
                r2_list.append(r2)
                pvalue_list.append(pvalue)
        
            # Create results DataFrame with proper index alignment
            result_index = df.index[window - 1:]
            regression_df = pd.DataFrame({
                'beta': beta_list,
                'r2': r2_list,
                'pvalue': pvalue_list
            }, index=result_index)
        
            # Rolling correlations
            corr_results = df['asset'].rolling(window=window).corr(df['brl'])
        
            # Combine all results into a single DataFrame for this asset
            asset_results = pd.DataFrame({
                'beta': regression_df['beta'],
                'r2': regression_df['r2'], 
                'pvalue': regression_df['pvalue'],
                'corr': corr_results
            }).dropna()
        
            rolling_results[a.symbol] = asset_results

        return rolling_results
    return (compute_stats,)


@app.cell
def _(
    AssetData,
    brl_ret,
    compute_returns,
    compute_stats,
    lookback,
    min_obs,
    price_map,
):
    # Build asset structures
    assets = []
    for sym, series in price_map.items():
        ret = compute_returns(series)
        if len(ret) < 30:
            continue
        assets.append(AssetData(sym, series, ret))

    rolling_results = compute_stats(assets, brl_ret, window=int(lookback.value), min_obs=int(min_obs.value))
    rolling_results
    return assets, rolling_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Plotting""")
    return


@app.cell
def _(go, make_subplots, np, pd):
    def plot_prices_grid_plotly(assets, brlusd: pd.Series, window: int = 126, ncols: int = 2, max_plots: int = 24):
        n = min(len(assets), max_plots)
        ncols_eff = max(1, ncols)
        nrows = int(np.ceil(n / ncols_eff))
        fig = make_subplots(rows=nrows, cols=ncols_eff, shared_xaxes=True, subplot_titles=[a.symbol for a in assets[:n]], horizontal_spacing=0.025, vertical_spacing=0.015)

        for idx in range(n):
            r = idx // ncols_eff + 1
            c = idx % ncols_eff + 1
            asset = assets[idx]
            df = pd.concat([asset.prices.rename('ASSET'), brlusd['BRL=X'].rename('BRLUSD')], axis=1).dropna()
            if df.empty:
                continue

            # Calculate rolling returns instead of using prices
            asset_rolling_returns = df['ASSET'].pct_change(window).dropna() * 100  # Convert to percentage
            brlusd_rolling_returns = df['BRLUSD'].pct_change(window).dropna() * 100  # Convert to percentage
        
            # Align the data
            rolling_df = pd.concat([asset_rolling_returns.rename('ASSET'), brlusd_rolling_returns.rename('BRLUSD')], axis=1).dropna()
        
            if rolling_df.empty:
                continue

            fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df['ASSET'], name=f'{asset.symbol} Returns', mode='lines', showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df['BRLUSD'], name=f'BRLUSD Returns ({window}d)', mode='lines', showlegend=(idx==0)), row=r, col=c)
    
        fig.update_layout(height=280*nrows, width=1100, title_text=f'Rolling Returns ({window}-day) - Asset vs BRLUSD', showlegend=False, margin=dict(l=0, r=0))
        return fig
    return (plot_prices_grid_plotly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 5.1 Grid Plots""")
    return


@app.cell
def _(assets, brlusd, lookback, max_plots, mo, ncols, plot_prices_grid_plotly):
    fig_grid = plot_prices_grid_plotly(assets, brlusd, window=int(lookback.value), ncols=int(ncols.value), max_plots=int(max_plots.value))
    mo.ui.plotly(fig_grid, config={"staticPlot": True},)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 5.2 Rolling Correlation""")
    return


@app.cell
def _():
    """
    TODO:
    1) Pegar variavel rolling_results e agrupar em diferentes dataframes linhas (ativos) x beta, r2, pvalue, corr
    2) Fazer diferentes gráficos para todos ativos em cada um
    3) Calcular as médias desse dataframe
    4) Criar os rankings (principal será por beta)
    """
    return


@app.cell
def _(go):
    def plot_rollcorr(symbol: str, series):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=f'Rolling Correlation {symbol}'))
        fig.update_layout(height=400, width=800)
        return fig
    return (plot_rollcorr,)


@app.cell
def _(mo, plot_rollcorr, roll_corrs, rolling_results, stats_df):
    examples = []
    if not rolling_results.empty:
        top_syms = rolling_results.index.tolist()[: int(10)] + stats_df.index.tolist()[- int(10):]
        for s in top_syms:
            rc = roll_corrs.get(s)
            if rc is not None and not rc.empty:
                examples.append((s, mo.ui.plotly(plot_rollcorr(s, rc))))
    mo.accordion({s: v for s, v in examples}) if examples else mo.md("No rolling correlation series available.")
    return


@app.cell
def _(mo):
    mo.md(r"""## 6. Rankings""")
    return


@app.cell
def _(pd):
    def make_rankings(df: pd.DataFrame, k: int, r2min: float):
        if df.empty:
            return df.copy(), df.copy()
        dff = df[df['r2'] >= r2min].copy()
        winners = dff.sort_values(['beta','r2','n_obs'], ascending=[False, False, False]).head(k)
        losers = dff.sort_values(['beta','r2','n_obs'], ascending=[True, False, False]).head(k)
        return winners, losers
    return (make_rankings,)


@app.cell
def _(make_rankings, min_r2, mo, stats_df):
    winners, losers = make_rankings(stats_df, 10, float(min_r2.value))


    mo.hstack([
        mo.vstack([mo.md("**Top beneficiaries (higher beta)**"), mo.ui.table(winners.drop('n_obs', axis=1).reset_index())]),
        mo.vstack([mo.md("**Most hurt (lower beta)**"), mo.ui.table(losers.drop('n_obs', axis=1).reset_index())])
    ])
    return losers, winners


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Export""")
    return


@app.cell
def _(OUTDIR, losers, os, pd, stats_df, winners):
    def export():
        os.makedirs(OUTDIR, exist_ok=True)
        xlsx_path = os.path.join(OUTDIR, "brl_sensitivity.xlsx")
        winners_path = os.path.join(OUTDIR, 'rank_beneficiadas_top10.csv')
        losers_path = os.path.join(OUTDIR, 'rank_prejudicadas_top10.csv')
        stats_path = os.path.join(OUTDIR, 'estatisticas_gerais.csv')
    
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            winners.to_excel(writer, sheet_name="winners")
            losers.to_excel(writer, sheet_name="losers")
            stats_df.to_excel(writer, sheet_name="all_stats")
        winners.to_csv(winners_path, index=True)
        losers.to_csv(losers_path, index=True)
        stats_df.to_csv(stats_path, index=True)
    

    return (export,)


@app.cell
def _(mo):
    export_btn = mo.ui.run_button(label="Export to Excel")
    export_btn
    return (export_btn,)


@app.cell
def _(export, export_btn):
    if export_btn.value:
        export()
        print("Export completed!")
    else:
        print('Click button to Export Files')
    return


if __name__ == "__main__":
    app.run()
