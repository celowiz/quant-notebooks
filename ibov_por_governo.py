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
        """
    # Ibovespa Performance by the Federal Government 🇧🇷

    /// details | *References:*

    - https://wilsonfreitas.github.io/posts/variacao-do-ibovespa-por-governo-federal.html
    - https://github.com/BDonadelli/Finance-playground/blob/main/B3_Ibov_desde_68.ipynb

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 1. Library Imports""")
    return


@app.cell
def _():
    import os
    from datetime import datetime, date
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import quantstats as qs
    from bcb import sgs
    return date, datetime, np, os, pd, plt, qs, sgs, sns, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Ibovespa daily closing prices

    ### From 1968 - 1997
    The Ibovespa closing values for all trading session as of January 1998 can be accessed bt the following URL: 

    - https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/indice-ibovespa-ibovespa-historic-statistics.htm

    ### From 1998 - Nowadays
    We can get Ibovespa prices using `yfinance` library

    """
    )
    return


@app.cell(hide_code=True)
def _(datetime, os, pd):
    # Utility Function to get Ibovespa Daily Data from 1968 - 1997
    def read_ibovespa_excel(file_path):
        """
        Read Ibovespa daily closing prices from Excel file and return a DataFrame.
    
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing Ibovespa data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with date index (YYYY-MM-DD format) and 'Valor' column containing prices
        """
    
        # Portuguese month names to numbers mapping
        month_mapping = {
            'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAIO': 5, 'JUN': 6,
            'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12
        }
    
        # Read all sheet names (years)
        excel_file = pd.ExcelFile(file_path)
        all_data = []
    
        for sheet_name in excel_file.sheet_names:
            try:
                # Read the sheet
                df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, thousands='.', decimal=',', skiprows=1, skipfooter=4)
            
                # Get the year from sheet name
                year = int(sheet_name)
            
                # Iterate through each month column
                for month_name in df_sheet.columns:
                    if month_name.upper() in month_mapping:
                        month_num = month_mapping[month_name.upper()]
                    
                        # Get the month data
                        month_data = df_sheet[month_name].dropna()
                    
                        # Create dates for each day in the month
                        for day in month_data.index:
                            try:
                                # Create date
                                date = datetime(year, month_num, int(day))
                                value = month_data[day]
                            
                                # Only add if value is not NaN
                                if pd.notna(value):
                                    all_data.append({
                                        'Date': date,
                                        'Value': float(value)
                                    })
                            except (ValueError, TypeError):
                                # Skip invalid dates or values
                                continue
                            
            except (ValueError, TypeError) as e:
                print(f"Skipping sheet '{sheet_name}': {e}")
                continue
    
        # Create DataFrame
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df.set_index('Date', inplace=True)
            result_df.sort_index(inplace=True)
            return result_df
        else:
            return pd.DataFrame(columns=['Value'])

    # Usage example:
    def load_ibovespa_data():
        """
        Convenience function to load Ibovespa data from the default location.
        """
        file_path = os.path.join('data', 'IBOVDIA.XLS')
        return read_ibovespa_excel(file_path)
    return (load_ibovespa_data,)


@app.cell
def _(load_ibovespa_data):
    old_data = load_ibovespa_data()
    old_data
    return (old_data,)


@app.cell
def _(pd, yf):
    new_data = pd.DataFrame()
    new_data['Value'] = yf.download('^BVSP', period='max', auto_adjust=True)['Close']
    new_data.loc['1998-01-01':]
    return (new_data,)


@app.cell
def _(new_data, old_data, pd):
    df = pd.concat([old_data, new_data.loc['1998-01-01':]])
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Visualize Data (from 1968 - Today)""")
    return


@app.cell
def _(df):
    df.plot()
    return


@app.cell
def _(df, plt):
    fig, axes = plt.subplots(nrows=3, ncols=2 ,figsize=(15,7))

    df.loc[:'1980'].plot(ax=axes[0,0]); axes[0,0].set_title('<1980')
    df.loc['1980':'1982'].plot(ax=axes[0,1]); axes[0,1].set_title('1980-1982')
    df.loc['1983':'1985'].plot(ax=axes[1,0]); axes[1,0].set_title('1983-1985')
    df.loc['1986':'1988'].plot(ax=axes[1,1]); axes[1,1].set_title('1986-1988')
    df.loc['1989':'1991'].plot(ax=axes[2,0]); axes[2,0].set_title('1989-1991')
    df.loc['1992':'1997'].plot(ax=axes[2,1]); axes[2,1].set_title('1992-1997')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Monthly Returns""")
    return


@app.cell
def _(df):
    monthly_df = df.copy().resample('ME').ffill()
    monthly_df['Return'] = monthly_df.pct_change()
    monthly_df
    return (monthly_df,)


@app.cell
def _(monthly_df):
    returns_table = monthly_df.pivot_table(values='Return', 
                                   index=monthly_df.index.year, 
                                   columns=monthly_df.index.month, 
                                   aggfunc='mean')
    returns_table
    return (returns_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Motnhly Returns Heatmap""")
    return


@app.cell
def _(plt, returns_table, sns):
    fig_2 = plt.figure()

    ax = plt.gca()

    sns.heatmap(returns_table.fillna(0)*100.0,annot=True,annot_kws={"size": 9},cmap='RdYlGn',
                alpha=1.0,center=0.0,cbar=False,ax=ax, fmt='.4g')
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Montlhy Return (%)")
    fig_2.set_size_inches(11,20)
    plt.show()
    return


@app.cell
def _(returns_table):
    returns_table.mean().plot.bar(figsize=(6,4))
    return


@app.cell
def _(mo):
    mo.md(r"""## 6. Quantstats Report""")
    return


@app.cell
def _(df, qs):
    returns = df.pct_change()
    qs.reports.full(returns.Value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Monthly Inflation Data (IPCA)""")
    return


@app.cell
def _(mo, pd, sgs):
    ipca = sgs.get({'IPCA': 433}, start='1968-01-01', end=pd.to_datetime('today'))
    mo.hstack([ipca, ipca.plot()])
    return (ipca,)


@app.cell
def _(mo):
    mo.md(r"""## 8. Calculate Ibovespa Returns and Deflated Returns""")
    return


@app.cell
def _(ipca, mo, monthly_df, pd):
    ipca_from_1994 = ipca.loc['1994-12-01':]
    ipca_from_1994 = ipca_from_1994.resample('ME').last()

    ibov_from_1994 = monthly_df['Value'].rename("IBOV").loc['1994-12-01':]
    data = pd.concat([ipca_from_1994, ibov_from_1994], axis=1)

    data['IBOV_RET'] = data['IBOV'].diff() / data['IBOV'].shift(1)
    data['IBOV_DEFL'] = (1 + data['IBOV_RET']) / (1 + data['IPCA']/100) - 1
    mo.hstack([data, data[['IBOV_RET', 'IBOV_DEFL']].plot()])
    return (data,)


@app.cell
def _(date, pd):
    def get_president(input_date):
        """
        Maps a given date to the corresponding Brazilian president in office.
    
        Parameters:
        -----------
        input_date : datetime.date, datetime.datetime, pd.Timestamp, or datetime-like
            The date to check
        
        Returns:
        --------
        str or None
            President name and term number (if applicable), or None if date is outside covered periods
        """
    
        # Convert to date if it's a datetime, Timestamp, or string
        if hasattr(input_date, 'date'):
            input_date = input_date.date()
        elif isinstance(input_date, str):
            input_date = pd.to_datetime(input_date).date()
        elif hasattr(input_date, 'to_pydatetime'):  # Handle pandas Timestamp
            input_date = input_date.to_pydatetime().date()
    
        # New Republic (1985-present)
        if input_date >= date(1985, 3, 15) and input_date < date(1990, 3, 15):
            return 'SARNEY'
        elif input_date >= date(1990, 3, 15) and input_date < date(1992, 12, 29):
            return 'COLLOR'
        elif input_date >= date(1992, 12, 29) and input_date < date(1995, 1, 1):
            return 'ITAMAR'
        elif input_date >= date(1995, 1, 1) and input_date < date(1999, 1, 1):
            return 'FHC 1'
        elif input_date >= date(1999, 1, 1) and input_date < date(2003, 1, 1):
            return 'FHC 2'
        elif input_date >= date(2003, 1, 1) and input_date < date(2007, 1, 1):
            return 'LULA 1'
        elif input_date >= date(2007, 1, 1) and input_date < date(2011, 1, 1):
            return 'LULA 2'
        elif input_date >= date(2011, 1, 1) and input_date < date(2015, 1, 1):
            return 'DILMA 1'
        elif input_date >= date(2015, 1, 1) and input_date < date(2016, 8, 31):
            return 'DILMA 2'
        elif input_date >= date(2016, 8, 31) and input_date < date(2019, 1, 1):
            return 'TEMER'
        elif input_date >= date(2019, 1, 1) and input_date < date(2023, 1, 1):
            return 'BOLSONARO'
        elif input_date >= date(2023, 1, 1):
            return 'LULA 3'
    
        # Fourth Republic (1946-1964) - Additional coverage
        elif input_date >= date(1946, 1, 31) and input_date < date(1951, 1, 31):
            return 'DUTRA'
        elif input_date >= date(1951, 1, 31) and input_date < date(1954, 8, 24):
            return 'VARGAS 2'
        elif input_date >= date(1954, 8, 24) and input_date < date(1955, 11, 8):
            return 'CAFÉ FILHO'
        elif input_date >= date(1955, 11, 8) and input_date < date(1956, 1, 31):
            return 'NEREU RAMOS'
        elif input_date >= date(1956, 1, 31) and input_date < date(1961, 1, 31):
            return 'KUBITSCHEK'
        elif input_date >= date(1961, 1, 31) and input_date < date(1961, 8, 25):
            return 'JÂNIO QUADROS'
        elif input_date >= date(1961, 8, 25) and input_date < date(1963, 1, 23):
            return 'RANIERI MAZZILLI 1'
        elif input_date >= date(1963, 1, 23) and input_date < date(1964, 4, 1):
            return 'JOÃO GOULART'
    
        # Military Dictatorship (1964-1985)
        elif input_date >= date(1964, 4, 1) and input_date < date(1964, 4, 15):
            return 'RANIERI MAZZILLI 2'
        elif input_date >= date(1964, 4, 15) and input_date < date(1967, 3, 15):
            return 'CASTELO BRANCO'
        elif input_date >= date(1967, 3, 15) and input_date < date(1969, 8, 31):
            return 'COSTA E SILVA'
        elif input_date >= date(1969, 8, 31) and input_date < date(1974, 3, 15):
            return 'MÉDICI'
        elif input_date >= date(1974, 3, 15) and input_date < date(1979, 3, 15):
            return 'GEISEL'
        elif input_date >= date(1979, 3, 15) and input_date < date(1985, 3, 15):
            return 'FIGUEIREDO'
    
        else:
            return None
    return (get_president,)


@app.cell
def _(data, get_president):
    data['PRESIDENT'] = data.index.map(get_president)
    data
    return


@app.cell
def _(data, np):
    data_gov = data.groupby('PRESIDENT')
    data_gov_agg = data_gov[['IBOV_RET', 'IBOV_DEFL']].aggregate(lambda x: 100*(np.prod(1 + x) - 1))
    data_gov_agg
    return data_gov, data_gov_agg


@app.cell
def _(data_gov_agg):
    def sort_by_govern(dx):
        governs = ['ITAMAR', 'FHC 1', 'FHC 2', 'LULA 1', 'LULA 2', 'DILMA 1', 'DILMA 2', 'TEMER', 'BOLSONARO', 'LULA 3']
        mapping = {gov: i for i, gov in enumerate(governs)}
        key = dx.index.map(lambda x: mapping[x])
        return dx.iloc[key.argsort()]

    sorted_data_gov_agg = sort_by_govern(data_gov_agg)
    sorted_data_gov_agg
    return sort_by_govern, sorted_data_gov_agg


@app.cell
def _(sorted_data_gov_agg):
    ax2 = sorted_data_gov_agg[['IBOV_RET', 'IBOV_DEFL']].plot(kind='bar', figsize=(10,6),
                                                              title="% Variation of IBOVESPA under Federal Governments")
    ax2.set_xlabel("Govern")
    ax2.set_ylabel("Variation (%)")
    return


@app.cell
def _(data_gov, sorted_data_gov_agg):
    sorted_data_gov_agg['Months'] = data_gov['PRESIDENT'].count()
    sorted_data_gov_agg
    return


@app.cell
def _(sorted_data_gov_agg):
    sorted_data_gov_agg['IBOV_RET_ANNUAL'] = ((1 + sorted_data_gov_agg['IBOV_RET']/100)**(12./sorted_data_gov_agg['Months']) - 1)*100
    sorted_data_gov_agg['IBOV_DEFL_ANNUAL'] = ((1 + sorted_data_gov_agg['IBOV_DEFL']/100)**(12./sorted_data_gov_agg['Months']) - 1)*100
    sorted_data_gov_agg
    return


@app.cell
def _(sorted_data_gov_agg):
    ax3 = sorted_data_gov_agg[['IBOV_RET_ANNUAL', 'IBOV_DEFL_ANNUAL']].plot(kind='bar', figsize=(10,6),
                                                                          title="% Variation of IBOVESPA under Federal Governments")
    ax3.set_xlabel("Govern")
    ax3.set_ylabel("Variation (%)")
    return


@app.cell
def _(data_gov, np, sort_by_govern):
    data_infl = data_gov['IPCA'].aggregate(lambda x: 100*(np.prod(1 + x/100)**(12./len(x)) - 1))
    data_infl = sort_by_govern(data_infl)
    ax4 = data_infl.plot(kind='bar', figsize=(10,6),
                        title="IPCA under Federal Governments")
    ax4.set_xlabel("Govern")
    ax4.set_ylabel("IPCA (%)")
    return


if __name__ == "__main__":
    app.run()
