# Amazon Sales Analysis

This project provides a comprehensive analysis of Amazon sales data, including data cleaning, exploration, visualization, and business insights. It is organized for reproducible data science workflows using Python and Jupyter Notebooks.

## Project Structure
```
amazon_sales_analysis/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Cleaned/processed data
├── notebooks/              # Jupyter notebooks for each analysis step
├── src/                    # Source code (data processing, analysis, visualization)
├── reports/                # Generated reports and figures
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── main_analysis.py        # Main analysis script
```

## Setup Instructions
1. **Clone the repository**
2. **Install [uv](https://github.com/astral-sh/uv) (a fast Python package/dependency manager):**
   ```bash
   pip install uv
   ```
3. **Create and activate a virtual environment using uv:**
   ```bash
   uv venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
4. **Install dependencies with uv:**
   ```bash
   uv add install -r requirements.txt
   ```
5. **Place your raw Amazon sales data CSV in `data/raw/`**
6. **Run the main analysis:**
   ```bash
   python main_analysis.py
   ```
7. **Explore the Jupyter notebooks in the `notebooks/` folder for step-by-step analysis.**

## Features
- Data loading, cleaning, and preprocessing
- Exploratory data analysis and visualization
- Sales, product, fulfillment, geographical, and customer segmentation analysis
- Business insights and recommendations

## Requirements
- Python 3.7+
- [uv](https://github.com/astral-sh/uv) (recommended for environment and dependency management)
- See `requirements.txt` for all dependencies

## Authors
- J Likith Sagar



