# Quantitative Finance Notebooks with Marimo 📊

This repository is dedicated to learning [Marimo](https://marimo.io/) and conducting quantitative data analysis in finance through interactive Python notebooks. The main objective is to explore financial data, test various analytical scripts, and conduct research using modern Python tools.

## 🎯 Project Objectives

- **Learn Marimo**: Explore the capabilities of Marimo as a reactive notebook environment
- **Quantitative Analysis**: Develop and test scripts for financial data analysis
- **Research**: Conduct financial research using Brazilian and international market data
- **Experimentation**: Test different approaches to financial modeling and analysis

## 📁 Project Structure

```
quant-notebooks/
├── .github/
│   └── workflows/
│       └── export_html.yml      # GitHub Actions for HTML export
├── notebooks/                   # All analysis notebooks organized by category
│   ├── ai/                     # AI and machine learning applications
│   ├── backtesting/            # Trading strategy backtesting
│   ├── data_extractor/         # Data collection and extraction tools
│   │   └── yfinance_guide.py   # Comprehensive YFinance usage guide
│   ├── data_visualization/     # Data visualization and analysis
│   │   └── b3_index_composition.py  # B3 stock index composition analysis
│   ├── macroeconomics/         # Macroeconomic analysis
│   ├── projects/               # Complete analysis projects
│   │   └── ibov_por_governo.py # Ibovespa performance by government periods
│   └── technical_indicators/   # Technical analysis indicators
│       └── stocks_momentum.py  # Stock momentum analysis
├── data/                       # Data files (Excel, CSV, etc.)
│   └── IBOVDIA.XLS            # Historical Ibovespa data (1968-1997)
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## 📊 Notebooks

### Projects

#### [Ibovespa Performance by the Federal Government 🇧🇷](./notebooks/projects/ibov_por_governo.py)
Comprehensive analysis of Ibovespa (Brazilian stock index) performance across different federal government periods. This notebook combines historical data from 1968-1997 with modern data from YFinance to provide insights into market performance under different political administrations.

**Key Features:**
- Historical data processing from Excel files (1968-1997)
- Modern data integration via YFinance (1998-present)
- Government period classification and analysis
- Performance visualization and statistical analysis
- Long-term trend analysis across political cycles

### Data Extraction

#### [Guide to Using YFinance with Python for Effective Stock Analysis](./notebooks/data_extractor/yfinance_guide.py)
Comprehensive guide demonstrating how to use the YFinance library for stock market data analysis. Covers everything from basic setup to advanced data manipulation techniques.

**Key Features:**
- Basic stock information retrieval
- Historical price data analysis
- Intraday data processing with minute-level intervals
- Multi-ticker bulk data analysis
- Returns and volatility calculations
- Dividend and split analysis
- Financial statements access
- ESG data retrieval
- Analyst recommendations
- Options data access
- Visualization examples

### Data Visualization

#### [B3 Index Composition](./notebooks/data_visualization/b3_index_composition.py)
Interactive analysis of Brazilian stock exchange (B3) index compositions. This notebook allows users to explore the composition of various B3 indices and analyze stock weightings.

**Key Features:**
- Real-time index composition data from B3 API
- Interactive index selection dropdown
- Stock weighting analysis and visualization
- Portfolio composition breakdown
- Data export capabilities (CSV and Parquet)
- Comprehensive coverage of major B3 indices (IBOV, IBrX 100, IBrX 50, etc.)

### Technical Indicators

#### [Stock Momentum Analysis](./notebooks/technical_indicators/stocks_momentum.py)
Technical analysis notebook focused on momentum indicators for stock analysis.

**Key Features:**
- Momentum indicator calculations
- Technical analysis tools
- Stock performance metrics

## 🛠️ Technologies Used

- **[Marimo](https://marimo.io/)**: Reactive Python notebook environment
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[YFinance](https://github.com/ranaroussi/yfinance)**: Stock market data retrieval
- **[Matplotlib](https://matplotlib.org/)**: Data visualization
- **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[QuantStats](https://github.com/ranaroussi/quantstats)**: Portfolio analytics
- **[python-bcb](https://github.com/wilsonfreitas/python-bcb)**: Brazilian Central Bank data access
- **[Requests](https://docs.python-requests.org/)**: HTTP library for API requests
- **[PyArrow](https://arrow.apache.org/docs/python/)**: Fast data processing and Parquet support
- **[xlrd](https://github.com/python-excel/xlrd)**: Excel file reading capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quant-notebooks
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Run Marimo notebooks:
```bash
# Run specific notebooks with uv
uv run marimo edit notebooks/projects/ibov_por_governo.py
uv run marimo edit notebooks/data_extractor/yfinance_guide.py
uv run marimo edit notebooks/data_visualization/b3_index_composition.py

# Or run any notebook in the notebooks directory
uv run marimo edit notebooks/path/to/your/notebook.py
```

Or if using pip:
```bash
marimo edit "notebooks/projects/ibov_por_governo.py"
```

## 📈 Data Sources

- **Historical Ibovespa Data (1968-1997)**: [B3 Historical Statistics](https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/indice-ibovespa-ibovespa-historic-statistics.htm)
- **Modern Stock Data**: [Yahoo Finance](https://finance.yahoo.com/) via YFinance
- **Brazilian Economic Data**: [Central Bank of Brazil (BCB)](https://www.bcb.gov.br/)

## 🤝 Contributing

This is a learning and research repository. Feel free to:

- Suggest improvements to existing analyses
- Propose new research topics
- Share interesting findings
- Report issues or bugs

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Useful Links

- [Marimo Documentation](https://docs.marimo.io/)
- [YFinance Documentation](https://github.com/ranaroussi/yfinance)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [B3 (Brazilian Stock Exchange)](https://www.b3.com.br/)

---

*Built with ❤️ for learning and financial research*