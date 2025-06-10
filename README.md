# Amazon Sales Analysis

This project provides a complete pipeline for analyzing Amazon sales data, including data cleaning, exploration, visualization, and business insights. It is designed for reproducibility and modularity, making it easy to extend or adapt for similar sales analytics tasks.

## Project Structure
```
amazon_sales_analysis/
├── data/
│   ├── raw/                # Place your raw data CSV here
│   └── processed/          # Cleaned data will be saved here
├── notebooks/              # Jupyter notebooks for step-by-step analysis
├── reports/                # Generated reports and figures
├── src/                    # Source code (data processing, analysis, visualization)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── main_analysis.py        # Main analysis script (optional)
└── run_pipeline.py         # Pipeline entry point
```

## Features
- Data loading, cleaning, and preprocessing
- Exploratory data analysis and summary statistics
- Sales, product, fulfillment, geographical, and customer segmentation analysis
- Business insights and recommendations
- Automated visualizations and interactive dashboards

## Getting Started
1. **Clone this repository**
2. **Install [uv](https://github.com/astral-sh/uv) and dependencies:**
   ```bash
   pip install uv
   uv venv .venv
   # On Windows: .venv\Scripts\activate
   # On macOS/Linux: source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. **Add your raw Amazon sales CSV to `data/raw/`**
4. **Run the pipeline:**
   ```bash
   python run_pipeline.py
   ```
5. **Check outputs in `data/processed/`, `reports/`, and view visualizations.**

## Usage
- You can also run or modify the Jupyter notebooks in the `notebooks/` folder for interactive, step-by-step analysis.
- All main logic is modularized in the `src/` directory for easy reuse.

## Author
J Likith Sagar