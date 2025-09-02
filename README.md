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
├── data/                    # Data files (Excel, CSV, etc.)
│   └── IBOVDIA.XLS         # Historical Ibovespa data (1968-1997)
├── ibov_por_governo.py     # Ibovespa analysis by government periods
├── yfinance_guide.py       # YFinance usage guide and examples
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## 📊 Notebooks

### [Ibovespa Performance by the Federal Government 🇧🇷](./ibov_por_governo.py)
Analysis of Ibovespa (Brazilian stock index) performance across different federal government periods. This notebook combines historical data from 1968-1997 with modern data from YFinance to provide comprehensive analysis of market performance under different political administrations.

**Key Features:**
- Historical data processing from Excel files
- Data combination from multiple sources
- Government period classification
- Performance visualization and analysis

### [Guide to Using YFinance with Python for Effective Stock Analysis](./yfinance_guide.py)
Comprehensive guide demonstrating how to use the YFinance library for stock market data analysis. Covers everything from basic setup to advanced data manipulation techniques.

**Key Features:**
- Basic stock information retrieval
- Historical price data analysis
- Intraday data processing
- Multi-ticker bulk data analysis
- Returns and volatility calculations

## 🛠️ Technologies Used

- **[Marimo](https://marimo.io/)**: Reactive Python notebook environment
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[YFinance](https://github.com/ranaroussi/yfinance)**: Stock market data retrieval
- **[Matplotlib](https://matplotlib.org/)**: Data visualization
- **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[QuantStats](https://github.com/ranaroussi/quantstats)**: Portfolio analytics

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
# Activate virtual environment if using uv
uv run marimo edit ibov_por_governo.py
# or
uv run marimo edit yfinance_guide.py
```

Or if using pip:
```bash
marimo edit ibov_por_governo.py
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