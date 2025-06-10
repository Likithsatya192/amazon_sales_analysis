import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def prepare_data_for_viz(df):
    df = df.copy()
    date_columns = ['Date', 'date', 'Order Date', 'order_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df['year'] = df[col].dt.year
            df['month'] = df[col].dt.month
            df['day_of_week'] = df[col].dt.day_name()
            break
    amount_columns = ['Amount', 'amount', 'Sales', 'sales', 'Revenue', 'revenue']
    for col in amount_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            break
    return df

def sales_overview_plots(df):
    df = prepare_data_for_viz(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Amazon Sales Overview Dashboard', fontsize=16, fontweight='bold')
    date_col = next((col for col in ['Date', 'date'] if col in df.columns), None)
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    if date_col and amount_col:
        daily_sales = df.groupby(date_col)[amount_col].sum().reset_index()
        axes[0, 0].plot(daily_sales[date_col], daily_sales[amount_col], color='blue', linewidth=2)
        axes[0, 0].set_title('Sales Trend Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales Amount')
        axes[0, 0].tick_params(axis='x', rotation=45)
    if 'month' in df.columns and amount_col:
        monthly_sales = df.groupby('month')[amount_col].sum()
        axes[0, 1].bar(monthly_sales.index, monthly_sales.values, color=sns.color_palette("viridis", len(monthly_sales)))
        axes[0, 1].set_title('Monthly Sales Distribution')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Total Sales')
    if 'day_of_week' in df.columns and amount_col:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales = df.groupby('day_of_week')[amount_col].sum().reindex(dow_order)
        axes[1, 0].bar(range(len(dow_sales)), dow_sales.values, color=sns.color_palette("coolwarm", 7))
        axes[1, 0].set_title('Sales by Day of Week')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Total Sales')
        axes[1, 0].set_xticks(range(len(dow_sales)))
        axes[1, 0].set_xticklabels([day[:3] for day in dow_sales.index], rotation=45)
    if amount_col:
        axes[1, 1].hist(df[amount_col].dropna(), bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Sales Amount Distribution')
        axes[1, 1].set_xlabel('Sales Amount')
        axes[1, 1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def product_analysis_plots(df):
    df = prepare_data_for_viz(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Product Analysis Dashboard', fontsize=16, fontweight='bold')
    category_cols = ['Category', 'category', 'Product Category', 'product_category']
    category_col = next((col for col in category_cols if col in df.columns), None)
    if category_col:
        category_counts = df[category_col].value_counts().head(10)
        axes[0, 0].barh(range(len(category_counts)), category_counts.values, color=sns.color_palette("Set2", len(category_counts)))
        axes[0, 0].set_title('Top 10 Product Categories')
        axes[0, 0].set_xlabel('Number of Orders')
        axes[0, 0].set_yticks(range(len(category_counts)))
        axes[0, 0].set_yticklabels(category_counts.index)
    size_cols = ['Size', 'size', 'Product Size', 'product_size']
    size_col = next((col for col in size_cols if col in df.columns), None)
    if size_col:
        size_counts = df[size_col].value_counts()
        axes[0, 1].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Product Size Distribution')
    qty_cols = ['Qty', 'qty', 'Quantity', 'quantity']
    qty_col = next((col for col in qty_cols if col in df.columns), None)
    if qty_col:
        axes[1, 0].hist(df[qty_col].dropna(), bins=20, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Quantity Distribution')
        axes[1, 0].set_xlabel('Quantity')
        axes[1, 0].set_ylabel('Frequency')
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    if category_col and amount_col:
        category_revenue = df.groupby(category_col)[amount_col].sum().sort_values(ascending=False).head(10)
        axes[1, 1].bar(range(len(category_revenue)), category_revenue.values, color=sns.color_palette("viridis", len(category_revenue)))
        axes[1, 1].set_title('Revenue by Product Category')
        axes[1, 1].set_xlabel('Product Category')
        axes[1, 1].set_ylabel('Total Revenue')
        axes[1, 1].set_xticks(range(len(category_revenue)))
        axes[1, 1].set_xticklabels(category_revenue.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def fulfillment_analysis_plots(df):
    df = prepare_data_for_viz(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fulfillment Analysis Dashboard', fontsize=16, fontweight='bold')
    fulfillment_cols = ['Fulfilment', 'fulfilment', 'Fulfillment', 'fulfillment']
    fulfillment_col = next((col for col in fulfillment_cols if col in df.columns), None)
    if fulfillment_col:
        fulfillment_counts = df[fulfillment_col].value_counts()
        axes[0, 0].pie(fulfillment_counts.values, labels=fulfillment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Fulfillment Method Distribution')
    status_cols = ['Status', 'status', 'Order Status', 'order_status']
    status_col = next((col for col in status_cols if col in df.columns), None)
    if status_col:
        status_counts = df[status_col].value_counts()
        axes[0, 1].bar(status_counts.index, status_counts.values, color=sns.color_palette("coolwarm", len(status_counts)))
        axes[0, 1].set_title('Order Status Distribution')
        axes[0, 1].set_xlabel('Status')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    channel_cols = ['Sales Channel', 'sales_channel', 'Channel', 'channel']
    channel_col = next((col for col in channel_cols if col in df.columns), None)
    if channel_col:
        channel_counts = df[channel_col].value_counts()
        axes[1, 0].barh(channel_counts.index, channel_counts.values, color=sns.color_palette("Set1", len(channel_counts)))
        axes[1, 0].set_title('Sales Channel Distribution')
        axes[1, 0].set_xlabel('Count')
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    if fulfillment_col and amount_col:
        fulfillment_revenue = df.groupby(fulfillment_col)[amount_col].sum()
        axes[1, 1].bar(fulfillment_revenue.index, fulfillment_revenue.values, color=sns.color_palette("plasma", len(fulfillment_revenue)))
        axes[1, 1].set_title('Revenue by Fulfillment Method')
        axes[1, 1].set_xlabel('Fulfillment Method')
        axes[1, 1].set_ylabel('Total Revenue')
        axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def geographical_analysis_plots(df):
    df = prepare_data_for_viz(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Geographical Analysis Dashboard', fontsize=16, fontweight='bold')
    state_cols = ['State', 'state', 'ship-state', 'Ship State']
    state_col = next((col for col in state_cols if col in df.columns), None)
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    if state_col:
        state_counts = df[state_col].value_counts().head(15)
        axes[0, 0].barh(range(len(state_counts)), state_counts.values, color=sns.color_palette("viridis", len(state_counts)))
        axes[0, 0].set_title('Top 15 States by Order Count')
        axes[0, 0].set_xlabel('Number of Orders')
        axes[0, 0].set_yticks(range(len(state_counts)))
        axes[0, 0].set_yticklabels(state_counts.index)
    city_cols = ['City', 'city', 'ship-city', 'Ship City']
    city_col = next((col for col in city_cols if col in df.columns), None)
    if city_col:
        city_counts = df[city_col].value_counts().head(10)
        axes[0, 1].bar(range(len(city_counts)), city_counts.values, color=sns.color_palette("coolwarm", len(city_counts)))
        axes[0, 1].set_title('Top 10 Cities by Order Count')
        axes[0, 1].set_xlabel('Cities')
        axes[0, 1].set_ylabel('Number of Orders')
        axes[0, 1].set_xticks(range(len(city_counts)))
        axes[0, 1].set_xticklabels(city_counts.index, rotation=45, ha='right')
    if state_col and amount_col:
        state_revenue = df.groupby(state_col)[amount_col].sum().sort_values(ascending=False).head(10)
        axes[1, 0].bar(range(len(state_revenue)), state_revenue.values, color=sns.color_palette("plasma", len(state_revenue)))
        axes[1, 0].set_title('Top 10 States by Revenue')
        axes[1, 0].set_xlabel('States')
        axes[1, 0].set_ylabel('Total Revenue')
        axes[1, 0].set_xticks(range(len(state_revenue)))
        axes[1, 0].set_xticklabels(state_revenue.index, rotation=45, ha='right')
    postal_cols = ['postal-code', 'Postal Code', 'ZIP', 'zip']
    postal_col = next((col for col in postal_cols if col in df.columns), None)
    if postal_col:
        postal_counts = df[postal_col].value_counts().head(15)
        axes[1, 1].barh(range(len(postal_counts)), postal_counts.values, color=sns.color_palette("Set2", len(postal_counts)))
        axes[1, 1].set_title('Top 15 Postal Codes by Order Count')
        axes[1, 1].set_xlabel('Number of Orders')
        axes[1, 1].set_yticks(range(len(postal_counts)))
        axes[1, 1].set_yticklabels(postal_counts.index)
    plt.tight_layout()
    plt.show()

def customer_segmentation_plots(df):
    df = prepare_data_for_viz(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    qty_col = next((col for col in ['Qty', 'qty', 'Quantity', 'quantity'] if col in df.columns), None)
    if amount_col:
        axes[0, 0].hist(df[amount_col].dropna(), bins=50, color='lightblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df[amount_col].mean(), color='red', linestyle='--', label=f'Mean: {df[amount_col].mean():.2f}')
        axes[0, 0].set_title('Customer Purchase Value Distribution')
        axes[0, 0].set_xlabel('Purchase Amount')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
    if amount_col and qty_col:
        axes[0, 1].scatter(df[qty_col], df[amount_col], alpha=0.6, color='green')
        axes[0, 1].set_title('Quantity vs Purchase Amount')
        axes[0, 1].set_xlabel('Quantity')
        axes[0, 1].set_ylabel('Purchase Amount')
    size_col = next((col for col in ['Size', 'size', 'Product Size'] if col in df.columns), None)
    if size_col and amount_col:
        size_amount = df.groupby(size_col)[amount_col].mean().sort_values(ascending=False)
        axes[1, 0].bar(size_amount.index, size_amount.values, color=sns.color_palette("viridis", len(size_amount)))
        axes[1, 0].set_title('Average Purchase Amount by Size')
        axes[1, 0].set_xlabel('Size')
        axes[1, 0].set_ylabel('Average Amount')
        axes[1, 0].tick_params(axis='x', rotation=45)
    if 'month' in df.columns and amount_col:
        monthly_avg = df.groupby('month')[amount_col].mean()
        axes[1, 1].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_title('Average Purchase Amount by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Amount')
        axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def interactive_dashboard(df):
    df = prepare_data_for_viz(df)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales Trend', 'Category Distribution', 'Fulfillment Methods', 'Geographical Sales'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    date_col = next((col for col in ['Date', 'date', 'Order Date'] if col in df.columns), None)
    amount_col = next((col for col in ['Amount', 'amount', 'Sales', 'sales'] if col in df.columns), None)
    if date_col and amount_col:
        daily_sales = df.groupby(date_col)[amount_col].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=daily_sales[date_col], y=daily_sales[amount_col], 
                      mode='lines+markers', name='Daily Sales'),
            row=1, col=1
        )
    category_col = next((col for col in ['Category', 'category', 'Product Category'] if col in df.columns), None)
    if category_col:
        category_counts = df[category_col].value_counts().head(8)
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"),
            row=1, col=2
        )
    fulfillment_col = next((col for col in ['Fulfilment', 'fulfilment', 'Fulfillment'] if col in df.columns), None)
    if fulfillment_col:
        fulfillment_counts = df[fulfillment_col].value_counts()
        fig.add_trace(
            go.Pie(labels=fulfillment_counts.index, values=fulfillment_counts.values, name="Fulfillment"),
            row=2, col=1
        )
    state_col = next((col for col in ['State', 'state', 'ship-state'] if col in df.columns), None)
    if state_col and amount_col:
        state_sales = df.groupby(state_col)[amount_col].sum().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=state_sales.index, y=state_sales.values, name='State Sales'),
            row=2, col=2
        )
    fig.update_layout(height=800, showlegend=True, title_text="Amazon Sales Interactive Dashboard")
    fig.show()

def generate_all_visualizations(df):
    print("Generating Sales Overview Plots...")
    sales_overview_plots(df)
    print("Generating Product Analysis Plots...")
    product_analysis_plots(df)
    print("Generating Fulfillment Analysis Plots...")
    fulfillment_analysis_plots(df)
    print("Generating Geographical Analysis Plots...")
    geographical_analysis_plots(df)
    print("Generating Customer Segmentation Plots...")
    customer_segmentation_plots(df)
    print("Generating Interactive Dashboard...")
    interactive_dashboard(df)
    print("All visualizations generated successfully!")