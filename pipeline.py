import os
import pandas as pd

import plotly.io as pio
pio.renderers.default = "browser"

from src.data_processing import DataProcessor
from src.analysis_functions import generate_comprehensive_report
from src.visualization import generate_all_visualizations

RAW_DATA_PATH = "data/raw/amazon_sales_data.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_amazon_data.csv"

def main():
    print("=== DATA PROCESSING ===")
    processor = DataProcessor()
    df = processor.load_data(RAW_DATA_PATH)
    if df is None:
        print("Data loading failed. Exiting pipeline.")
        return
    processor.explore_data()
    cleaned_df = processor.clean_data()
    processor.save_cleaned_data(CLEANED_DATA_PATH)

    print("\n=== ANALYSIS ===")
    report = generate_comprehensive_report(cleaned_df)
    print("\nExecutive Summary:")
    print(report.get('executive_summary', {}))
    print("\nBusiness Insights:")
    print(report.get('business_insights', {}))

    print("\n=== VISUALIZATION ===")
    generate_all_visualizations(cleaned_df)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()