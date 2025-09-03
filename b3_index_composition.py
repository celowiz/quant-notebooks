import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# B3 Index Composition""")
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Library Imports""")
    return


@app.cell
def _():
    import os
    import requests
    import json
    import base64
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns
    return base64, datetime, json, np, os, pd, plt, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Available Indexes

    The index composition with all stocks that comprise them can be found on the official website of the brazilian stock exchange (B3) using the following links:

    ### Broad Indices: 
    - [Bovespa Index (Ibovespa)](https://www.b3.com.br/en_us/market-data-and-indices/indices/broad-indices/indice-ibovespa-ibovespa-composition-index-portfolio.htm)
    - [Bovespa B3 BR+ (Ibovespa B3 BR+)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/bovespa-b3-br-index-ibovespa-b3-br-composition-index-portfolio.htm)
    - [B3 BR+ Equal Weight Bovespa Index (Ibovespa B3 BR+ Equal Weight)**](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/b3-br-equal-weight-bovespa-index-ibovespa-b3-br-equal-weight-composition-index-portfolio.htm)
    - [Brazil 100 Index (IBrX 100)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/indice-brasil-100-ibrx-100-composition-index-portfolio.htm)
    - [Brazil 50 Index (IBrX 50)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/indice-brasil-50-ibrx-50-composition-index-portfolio.htm)
    - [Brazil Broad-Based Index (IBrA)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/brazil-broad-based-index-ibra-composition-index-portfolio.htm) 

    ### Indices for Segments and Sectors: 
    - [BM&FBOVESPA Basic Materials Index (IMAT)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/basic-materials-index-imat-composition-index-portfolio.htm)
    - [BM&FBOVESPA Consumer Stock Index (ICON)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/consumer-stock-index-icon-composition-index-portfolio.htm)
    - [BM&FBOVESPA Dividend Index (IDIV)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/dividend-index-idiv-composition-index-portfolio.htm)
    - [BM&FBOVESPA Electric Utilities Index (IEE)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/electric-utilities-index-iee-composition-index-portfolio.htm)
    - [BM&FBOVESPA Financials Index (IFNC)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/financials-index-ifnc-composition-index-portfolio.htm)
    - [BM&FBOVESPA Industrials Index (INDX)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/industrials-index-indx-composition-index-portfolio.htm)
    - [BM&FBOVESPA Public Utilities Index (UTIL)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/public-utilities-index-util-composition-index-portfolio.htm)
    - [BM&FBOVESPA Real Estate Index (IMOB)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/real-estate-index-imob-composition-index-portfolio.htm)
    - [SmallCap Index (SMLL)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/smallcap-index-smll-composition-index-portfolio.htm)
    - [Unsponsored BDR Index-GLOBAL (BRDX)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/unsponsored-bdr-index-global-bdrx-composition-index-portfolio.htm)
    - [Valor BM&FBOVESPA Index (IVBX 2)](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indexes-for-segments-and-sectors/valor-index-ivbx-2-composition-index-portfolio.htm)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 2. Select Index""")
    return


@app.cell
def _(base64, json, requests):
    def get_available_indexes():
        """
        Make a request to the specified URL and return a list of all available indexes.
    
        Parameters:
        -----------
        url : str
            The URL to make the request to
        
        Returns:
        --------
        list
            List of all available indexes as strings
        """

        base64.decodebytes(b"eyJwYWdlTnVtYmVyIjoxLCJwYWdlU2l6ZSI6MjB9")
        params = json.dumps({"pageNumber": 1, "pageSize": 999})
        params_enc = base64.encodebytes(bytes(params, "utf8")).decode("utf8")

        url = f"https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetStockIndex/{params_enc.strip()}"

        try:
            # Make the request
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
        
            # Parse JSON response
            data = response.json()
        
            # Extract results
            results = data.get('results', [])
        
            # Get all indixes for each company
            all_indexes = []
            for item in results:
                indexes = item['indexes'].split(',')
                all_indexes += indexes
        
            return sorted(list(set(all_indexes)))
        
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return []
        except KeyError as e:
            print(f"Error parsing response: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    return (get_available_indexes,)


@app.cell
def _(get_available_indexes, mo):
    list_index = get_available_indexes()

    callout_index = mo.ui.dropdown(
        label="Select Index", 
        options=list_index, 
        value='IBOV'
    )
    return (callout_index,)


@app.cell
def _(base64, datetime, json, os, pd, requests):
    def return_index(index:str): 

        URL_B3 = 'https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/'
        DEFAULT_PAYLOAD = {'language': 'pt-br', 'index': index.upper().strip(), 'segment': '2'}

        try:
            payload_json = json.dumps(DEFAULT_PAYLOAD)
            payload_b64 = base64.b64encode(payload_json.encode()).decode()
            url = URL_B3 + payload_b64

            response = requests.get(url)
            response.raise_for_status()

            dados_json = response.json()
            lista_acoes = dados_json.get('results')

            if not lista_acoes:
                print("key 'results' not found or empty json.")
                return None

            df = pd.DataFrame(lista_acoes)

            # casting of cols
            df['part'] = df['part'].str.replace(',', '.').astype(float)
            df['partAcum'] = df['partAcum'].str.replace(',', '.').astype(float)
            df['theoricalQty'] = df['theoricalQty'].str.replace('.', '', regex=False).astype(int)

            # sort by 'part'
            df = df.sort_values('part', ascending=False)

            # add date
            today_date = datetime.now().strftime('%Y-%m-%d')
            df['date'] = today_date

            return df.sort_values(by='part', ascending=False)[['date', 'cod', 'segment', 'part']]

        except requests.exceptions.RequestException as e:
            print(f"HTTP request errror: {e}")
            return None
        except json.JSONDecodeError:
            print("json decode error.")
            return None
        except Exception as e:
            print(f"unexpected error: {e}")
            return None

    def salvar_dados(df: pd.DataFrame, index: str):
        """
        Salva o DataFrame em arquivos CSV e Parquet.
        """
        if df is None or df.empty:
            print("O DataFrame está vazio. Nenhum arquivo será salvo.")
            return

        try:
            dir_path = os.path.join('data', f'{index.lower()} composition')
            os.makedirs(dir_path, exist_ok=True)
            today_date = datetime.now().strftime('%Y-%m-%d')
        
            caminho_csv = os.path.join(dir_path, f"{today_date}.csv")
            df.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
            print(f"\nArquivo salvo com sucesso em: {caminho_csv}")

            caminho_parquet = os.path.join(dir_path, f"{today_date}.parquet")
            df.to_parquet(caminho_parquet, index=False, engine='pyarrow')
            print(f"Arquivo salvo com sucesso em: {caminho_parquet}")

        except Exception as e:
            print(f"Erro ao salvar os arquivos: {e}")
    return return_index, salvar_dados


@app.cell
def _(callout_index, return_index):
    df = return_index(index=callout_index.value)
    return (df,)


@app.cell
def _(callout_index, df, salvar_dados):
    if df is not None:
        salvar_dados(df, callout_index.value)
    return


@app.cell
def _(callout_index, df, mo):
    mo.vstack([callout_index, df], align='stretch', gap=0)

    return


@app.cell
def _(np, pd, plt):
    def top_part(data, n=10):
        """
        Extract top n parted components and prepare data for donut chart visualization.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing 'cod' and 'part' columns
        n : int, default=10
            Number of top components to extract

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: cod, part, cum_part, ymax, ymin, label_pos, label
        """

        # Get top n components by 'part'
        top_n = (data
                 .head(n)
                 [['cod', 'part']]
                 .copy())

        # Calculate total part of top components
        total_part = top_n['part'].sum()

        # Create "Others" category for remaining part
        others = pd.DataFrame({
            'cod': ['Others'],
            'part': [100 - total_part]
        })

        # Combine top components with others
        result = pd.concat([top_n, others], ignore_index=True)

        # Calculate cumulative parts and positions for visualization
        result['cum_part'] = result['part'].cumsum()
        result['ymax'] = result['cum_part']
        result['ymin'] = np.concatenate([[0], result['cum_part'].iloc[:-1].values])
        result['label_pos'] = (result['ymax'] + result['ymin']) / 2
        result['label'] = result['cod'] + '\n' + (result['part']).round(1).astype(str) + '%'

        # Convert cod to categorical for consistent ordering
        result['cod'] = pd.Categorical(result['cod'], categories=result['cod'], ordered=True)

        return result

    def create_donut_chart(data, index_name, figsize=(10, 10)):
        """
        Create a donut chart visualization from participation data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: ymax, ymin, cod, label_pos, label (from top_part function)
        index_name : str
            Name of the index to display in the center
        figsize : tuple, default=(10, 10)
            Figure size for the plot

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Set theta direction and zero location
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')

        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))

        # Draw donut segments
        for i, row in data.iterrows():
            # Convert percentages to radians (divide by 100 to get fraction, then multiply by 2π)
            theta1 = (row['ymin'] / 100) * 2 * np.pi
            theta2 = (row['ymax'] / 100) * 2 * np.pi

            # Create wedge for donut segment
            theta = np.linspace(theta1, theta2, 100)
            r_inner = np.full_like(theta, 0.6)
            r_outer = np.full_like(theta, 1.0)

            ax.fill_between(theta, r_inner, r_outer, color=colors[i], alpha=0.8, edgecolor='white', linewidth=2)

            # Add labels - calculate middle angle for label positioning
            theta_mid = (theta1 + theta2) / 2
            r_label = 1.15

            # Only show labels for segments larger than 2% to avoid overcrowding
            if row['part'] > 2:
                ax.text(theta_mid, r_label, row['label'], 
                        ha='center', va='center', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.text(0, 0, index_name, ha='center', va='center', 
                fontsize=16, fontweight='bold', color='grey')

        # Style the plot
        ax.set_ylim(0, 1.3)
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

        plt.tight_layout()
        plt.show()

    # Example usage function
    def plot_index_composition(data, index_name, n=10, figsize=(8, 8)):
        """
        Complete workflow to create donut chart from raw part data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with 'cod' and 'part' columns
        index_name : str
            Name of the index
        n : int, default=10
            Number of top components to show
        figsize : tuple, default=(8, 8)
            Figure size

        Returns:
        --------
        matplotlib.figure.Figure
            The created donut chart figure
        """
        processed_data = top_part(data, n=n)
        return create_donut_chart(processed_data, index_name, figsize=figsize)
    return (plot_index_composition,)


@app.cell
def _(callout_index, df, plot_index_composition):
    plot_index_composition(df, index_name=callout_index.value)
    return


if __name__ == "__main__":
    app.run()
