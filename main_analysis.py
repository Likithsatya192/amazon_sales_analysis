import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

class AmazonSalesAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the Amazon sales data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Initial data exploration"""
        print("=== DATA EXPLORATION ===")
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumn Names: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        
        # Basic statistics
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n=== DATA CLEANING ===")
        self.cleaned_df = self.df.copy()
        
        # Convert date column (assuming it exists)
        date_columns = [col for col in self.cleaned_df.columns if 'date' in col.lower()]
        for col in date_columns:
            self.cleaned_df[col] = pd.to_datetime(self.cleaned_df[col], errors='coerce')
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{self.cleaned_df.isnull().sum()}")
        
        # Remove duplicates
        duplicates = self.cleaned_df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        
        print(f"Data shape after cleaning: {self.cleaned_df.shape}")
        
    def sales_overview_analysis(self):
        """Analyze overall sales performance and trends"""
        print("\n=== SALES OVERVIEW ANALYSIS ===")
        
        # Assuming columns exist - adjust based on actual data
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        quantity_col = [col for col in self.cleaned_df.columns if 'qty' in col.lower() or 'quantity' in col.lower()][0]
        date_col = [col for col in self.cleaned_df.columns if 'date' in col.lower()][0]
        
        # Total sales metrics
        total_revenue = self.cleaned_df[amount_col].sum()
        total_orders = len(self.cleaned_df)
        total_quantity = self.cleaned_df[quantity_col].sum()
        avg_order_value = total_revenue / total_orders
        
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Total Orders: {total_orders:,}")
        print(f"Total Quantity Sold: {total_quantity:,}")
        print(f"Average Order Value: ${avg_order_value:.2f}")
        
        # Sales trends over time
        self.cleaned_df['month'] = self.cleaned_df[date_col].dt.to_period('M')
        monthly_sales = self.cleaned_df.groupby('month')[amount_col].sum().reset_index()
        monthly_sales['month'] = monthly_sales['month'].astype(str)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Monthly sales trend
        axes[0,0].plot(monthly_sales['month'], monthly_sales[amount_col])
        axes[0,0].set_title('Monthly Sales Trend')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Sales Amount')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Daily sales distribution
        daily_sales = self.cleaned_df.groupby(self.cleaned_df[date_col].dt.date)[amount_col].sum()
        axes[0,1].hist(daily_sales, bins=30, alpha=0.7)
        axes[0,1].set_title('Daily Sales Distribution')
        axes[0,1].set_xlabel('Sales Amount')
        axes[0,1].set_ylabel('Frequency')
        
        # Order status distribution
        status_col = [col for col in self.cleaned_df.columns if 'status' in col.lower()][0]
        status_counts = self.cleaned_df[status_col].value_counts()
        axes[1,0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Order Status Distribution')
        
        # Sales by day of week
        self.cleaned_df['day_of_week'] = self.cleaned_df[date_col].dt.day_name()
        dow_sales = self.cleaned_df.groupby('day_of_week')[amount_col].sum()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales = dow_sales.reindex(dow_order)
        axes[1,1].bar(dow_sales.index, dow_sales.values)
        axes[1,1].set_title('Sales by Day of Week')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/figures/sales_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def product_analysis(self):
        """Analyze product categories, sizes, and quantities"""
        print("\n=== PRODUCT ANALYSIS ===")
        
        category_col = [col for col in self.cleaned_df.columns if 'category' in col.lower()][0]
        size_col = [col for col in self.cleaned_df.columns if 'size' in col.lower()]
        quantity_col = [col for col in self.cleaned_df.columns if 'qty' in col.lower() or 'quantity' in col.lower()][0]
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        
        # Product category analysis
        category_stats = self.cleaned_df.groupby(category_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            quantity_col: ['sum', 'mean']
        }).round(2)
        
        print("Top 10 Product Categories by Revenue:")
        top_categories = self.cleaned_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False).head(10)
        print(top_categories)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top categories by revenue
        top_categories.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Top 10 Categories by Revenue')
        axes[0,0].set_xlabel('Category')
        axes[0,0].set_ylabel('Revenue')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Quantity sold by category
        qty_by_category = self.cleaned_df.groupby(category_col)[quantity_col].sum().sort_values(ascending=False).head(10)
        qty_by_category.plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Top 10 Categories by Quantity Sold')
        axes[0,1].set_xlabel('Category')
        axes[0,1].set_ylabel('Quantity')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Size distribution (if size column exists)
        if size_col:
            size_counts = self.cleaned_df[size_col[0]].value_counts().head(10)
            axes[1,0].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%')
            axes[1,0].set_title('Size Distribution')
        
        # Average order value by category
        avg_order_by_category = self.cleaned_df.groupby(category_col)[amount_col].mean().sort_values(ascending=False).head(10)
        avg_order_by_category.plot(kind='bar', ax=axes[1,1], color='green')
        axes[1,1].set_title('Average Order Value by Category')
        axes[1,1].set_xlabel('Category')
        axes[1,1].set_ylabel('Average Order Value')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/figures/product_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def fulfillment_analysis(self):
        """Analyze fulfillment methods and their effectiveness"""
        print("\n=== FULFILLMENT ANALYSIS ===")
        
        fulfillment_col = [col for col in self.cleaned_df.columns if 'fulfil' in col.lower() or 'ship' in col.lower()][0]
        status_col = [col for col in self.cleaned_df.columns if 'status' in col.lower()][0]
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        
        # Fulfillment method distribution
        fulfillment_dist = self.cleaned_df[fulfillment_col].value_counts()
        print("Fulfillment Method Distribution:")
        print(fulfillment_dist)
        
        # Fulfillment effectiveness by status
        fulfillment_status = pd.crosstab(self.cleaned_df[fulfillment_col], self.cleaned_df[status_col])
        print(f"\nFulfillment vs Status Cross-tabulation:\n{fulfillment_status}")
        
        # Revenue by fulfillment method
        fulfillment_revenue = self.cleaned_df.groupby(fulfillment_col)[amount_col].sum().sort_values(ascending=False)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Fulfillment method distribution
        fulfillment_dist.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('Fulfillment Method Distribution')
        
        # Revenue by fulfillment method
        fulfillment_revenue.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Revenue by Fulfillment Method')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Fulfillment vs Status heatmap
        sns.heatmap(fulfillment_status, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
        axes[1,0].set_title('Fulfillment Method vs Order Status')
        
        # Success rate by fulfillment method
        success_rate = self.cleaned_df.groupby(fulfillment_col)[status_col].apply(
            lambda x: (x == 'Shipped').sum() / len(x) * 100 if 'Shipped' in x.values else 0
        )
        success_rate.plot(kind='bar', ax=axes[1,1], color='green')
        axes[1,1].set_title('Success Rate by Fulfillment Method (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/figures/fulfillment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def geographical_analysis(self):
        """Analyze geographical distribution of sales"""
        print("\n=== GEOGRAPHICAL ANALYSIS ===")
        
        # Find geography columns
        state_col = [col for col in self.cleaned_df.columns if 'state' in col.lower()][0]
        city_col = [col for col in self.cleaned_df.columns if 'city' in col.lower()]
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        
        # State-wise analysis
        state_sales = self.cleaned_df.groupby(state_col)[amount_col].sum().sort_values(ascending=False)
        state_orders = self.cleaned_df.groupby(state_col).size().sort_values(ascending=False)
        
        print("Top 10 States by Revenue:")
        print(state_sales.head(10))
        
        if city_col:
            city_sales = self.cleaned_df.groupby(city_col[0])[amount_col].sum().sort_values(ascending=False)
            print("\nTop 10 Cities by Revenue:")
            print(city_sales.head(10))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top states by revenue
        state_sales.head(15).plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Top 15 States by Revenue')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Top states by number of orders
        state_orders.head(15).plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Top 15 States by Number of Orders')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        if city_col:
            # Top cities by revenue
            city_sales.head(15).plot(kind='bar', ax=axes[1,0], color='green')
            axes[1,0].set_title('Top 15 Cities by Revenue')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Revenue distribution by state (box plot)
        state_revenue_dist = []
        for state in state_sales.head(10).index:
            state_revenue_dist.append(self.cleaned_df[self.cleaned_df[state_col] == state][amount_col].values)
        
        axes[1,1].boxplot(state_revenue_dist, labels=state_sales.head(10).index)
        axes[1,1].set_title('Revenue Distribution by Top 10 States')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/figures/geographical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def customer_segmentation(self):
        """Perform customer segmentation analysis"""
        print("\n=== CUSTOMER SEGMENTATION ===")
        
        # Prepare data for clustering
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        quantity_col = [col for col in self.cleaned_df.columns if 'qty' in col.lower() or 'quantity' in col.lower()][0]
        
        # Create customer-level features (assuming there's a customer identifier)
        # If no customer ID, we'll segment by order characteristics
        segmentation_features = self.cleaned_df[[amount_col, quantity_col]].copy()
        
        # Add derived features
        segmentation_features['order_value_category'] = pd.cut(
            self.cleaned_df[amount_col], 
            bins=[0, 50, 200, 1000, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # Prepare data for clustering
        X = segmentation_features[[amount_col, quantity_col]].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        self.cleaned_df['customer_segment'] = clusters
        
        # Analyze segments
        segment_analysis = self.cleaned_df.groupby('customer_segment').agg({
            amount_col: ['mean', 'sum', 'count'],
            quantity_col: ['mean', 'sum']
        }).round(2)
        
        print("Customer Segment Analysis:")
        print(segment_analysis)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot of segments
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(4):
            cluster_data = X[clusters == i]
            axes[0,0].scatter(cluster_data[amount_col], cluster_data[quantity_col], 
                            c=colors[i], label=f'Segment {i}', alpha=0.6)
        axes[0,0].set_xlabel('Order Amount')
        axes[0,0].set_ylabel('Quantity')
        axes[0,0].set_title('Customer Segments')
        axes[0,0].legend()
        
        # Segment size distribution
        segment_sizes = pd.Series(clusters).value_counts().sort_index()
        axes[0,1].pie(segment_sizes.values, labels=[f'Segment {i}' for i in segment_sizes.index], 
                     autopct='%1.1f%%')
        axes[0,1].set_title('Segment Size Distribution')
        
        # Average order value by segment
        avg_order_by_segment = self.cleaned_df.groupby('customer_segment')[amount_col].mean()
        avg_order_by_segment.plot(kind='bar', ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('Average Order Value by Segment')
        axes[1,0].set_xlabel('Segment')
        
        # Total revenue by segment
        revenue_by_segment = self.cleaned_df.groupby('customer_segment')[amount_col].sum()
        revenue_by_segment.plot(kind='bar', ax=axes[1,1], color='lightcoral')
        axes[1,1].set_title('Total Revenue by Segment')
        axes[1,1].set_xlabel('Segment')
        
        plt.tight_layout()
        plt.savefig('reports/figures/customer_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_insights_and_recommendations(self):
        """Generate business insights and recommendations"""
        print("\n=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
        
        insights = []
        recommendations = []
        
        # Revenue insights
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        total_revenue = self.cleaned_df[amount_col].sum()
        avg_order_value = self.cleaned_df[amount_col].mean()
        
        insights.append(f"Total revenue generated: ${total_revenue:,.2f}")
        insights.append(f"Average order value: ${avg_order_value:.2f}")
        
        # Top performing categories
        category_col = [col for col in self.cleaned_df.columns if 'category' in col.lower()][0]
        top_category = self.cleaned_df.groupby(category_col)[amount_col].sum().idxmax()
        insights.append(f"Top performing category: {top_category}")
        
        # Geographic insights
        state_col = [col for col in self.cleaned_df.columns if 'state' in col.lower()][0]
        top_state = self.cleaned_df.groupby(state_col)[amount_col].sum().idxmax()
        insights.append(f"Top performing state: {top_state}")
        
        # Recommendations based on analysis
        recommendations.extend([
            "Focus marketing efforts on top-performing product categories",
            "Optimize inventory management for high-demand products",
            "Develop targeted marketing campaigns for different customer segments",
            "Improve fulfillment processes in underperforming regions",
            "Implement dynamic pricing strategies based on demand patterns",
            "Enhance customer retention programs for high-value segments"
        ])
        
        print("KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
            
        print("\nRECOMMENDations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
            
        return insights, recommendations
        
    def create_dashboard_summary(self):
        """Create a comprehensive dashboard summary"""
        print("\n=== DASHBOARD SUMMARY ===")
        
        # Key metrics
        amount_col = [col for col in self.cleaned_df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'sales' in col.lower()][0]
        quantity_col = [col for col in self.cleaned_df.columns if 'qty' in col.lower() or 'quantity' in col.lower()][0]
        
        metrics = {
            'Total Revenue': f"${self.cleaned_df[amount_col].sum():,.2f}",
            'Total Orders': f"{len(self.cleaned_df):,}",
            'Total Quantity Sold': f"{self.cleaned_df[quantity_col].sum():,}",
            'Average Order Value': f"${self.cleaned_df[amount_col].mean():.2f}",
            'Unique Products': f"{self.cleaned_df['category' if 'category' in self.cleaned_df.columns else self.cleaned_df.columns[0]].nunique():,}",
        }
        
        print("KEY PERFORMANCE METRICS:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        return metrics

# Main execution
def main():
    # Initialize analyzer
    analyzer = AmazonSalesAnalyzer('data/raw/amazon_sales_data.csv')
    
    # Load and explore data
    df = analyzer.load_data()
    if df is not None:
        analyzer.explore_data()
        analyzer.clean_data()
        
        # Perform all analyses
        analyzer.sales_overview_analysis()
        analyzer.product_analysis()
        analyzer.fulfillment_analysis()
        analyzer.geographical_analysis()
        analyzer.customer_segmentation()
        
        # Generate insights
        insights, recommendations = analyzer.generate_insights_and_recommendations()
        
        # Create dashboard summary
        metrics = analyzer.create_dashboard_summary()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)

if __name__ == "__main__":
    main()