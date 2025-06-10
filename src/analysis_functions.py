import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def prepare_data(df):
    """
    Prepare and clean data for analysis
    """
    df = df.copy()
    date_columns = ['Date', 'date', 'Order Date', 'order_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df['year'] = df[col].dt.year
            df['month'] = df[col].dt.month
            df['quarter'] = df[col].dt.quarter
            df['day_of_week'] = df[col].dt.day_name()
            df['is_weekend'] = df[col].dt.weekday >= 5
            break
    amount_columns = ['Amount', 'amount', 'Sales', 'sales', 'Revenue', 'revenue']
    for col in amount_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            break
    qty_columns = ['Qty', 'qty', 'Quantity', 'quantity']
    for col in qty_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            break
    return df

def sales_overview_analysis(df):
    amount_col = _get_column(df, ['Amount', 'amount', 'Sales', 'sales'])
    date_col = _get_column(df, ['Date', 'date', 'Order Date', 'order_date'])
    if not amount_col:
        return {"error": "No amount column found"}
    analysis = {}
    analysis['total_revenue'] = df[amount_col].sum()
    analysis['total_orders'] = len(df)
    analysis['average_order_value'] = df[amount_col].mean()
    analysis['median_order_value'] = df[amount_col].median()
    analysis['max_order_value'] = df[amount_col].max()
    analysis['min_order_value'] = df[amount_col].min()
    analysis['sales_std'] = df[amount_col].std()
    analysis['sales_variance'] = df[amount_col].var()
    analysis['sales_percentiles'] = {
        '25th': df[amount_col].quantile(0.25),
        '50th': df[amount_col].quantile(0.50),
        '75th': df[amount_col].quantile(0.75),
        '90th': df[amount_col].quantile(0.90),
        '95th': df[amount_col].quantile(0.95)
    }
    if date_col:
        monthly_sales = df.groupby('month')[amount_col].agg(['sum', 'mean', 'count']).round(2)
        analysis['monthly_performance'] = monthly_sales.to_dict()
        dow_sales = df.groupby('day_of_week')[amount_col].agg(['sum', 'mean', 'count']).round(2)
        analysis['day_of_week_performance'] = dow_sales.to_dict()
        weekend_analysis = df.groupby('is_weekend')[amount_col].agg(['sum', 'mean', 'count']).round(2)
        analysis['weekend_vs_weekday'] = weekend_analysis.to_dict()
    return analysis

def product_analysis(df):
    analysis = {}
    category_col = _get_column(df, ['Category', 'category', 'Product Category', 'product_category'])
    amount_col = _get_column(df, ['Amount', 'amount', 'Sales', 'sales'])
    qty_col = _get_column(df, ['Qty', 'qty', 'Quantity', 'quantity'])
    size_col = _get_column(df, ['Size', 'size', 'Product Size', 'product_size'])
    if category_col:
        cat_analysis = df.groupby(category_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count',
            qty_col: ['sum', 'mean'] if qty_col else 'count'
        }).round(2)
        analysis['category_performance'] = cat_analysis.to_dict()
        if amount_col:
            top_categories_revenue = df.groupby(category_col)[amount_col].sum().sort_values(ascending=False).head(10)
            analysis['top_categories_by_revenue'] = top_categories_revenue.to_dict()
        top_categories_orders = df[category_col].value_counts().head(10)
        analysis['top_categories_by_orders'] = top_categories_orders.to_dict()
    if size_col:
        size_analysis = df.groupby(size_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count',
            qty_col: ['sum', 'mean'] if qty_col else 'count'
        }).round(2)
        analysis['size_performance'] = size_analysis.to_dict()
        size_distribution = df[size_col].value_counts(normalize=True).round(3)
        analysis['size_distribution'] = size_distribution.to_dict()
    if qty_col:
        analysis['quantity_stats'] = {
            'mean': df[qty_col].mean(),
            'median': df[qty_col].median(),
            'std': df[qty_col].std(),
            'max': df[qty_col].max(),
            'min': df[qty_col].min()
        }
        bulk_threshold = df[qty_col].quantile(0.75)
        analysis['bulk_vs_single'] = {
            'bulk_orders_count': (df[qty_col] >= bulk_threshold).sum(),
            'single_orders_count': (df[qty_col] < bulk_threshold).sum(),
            'bulk_threshold': bulk_threshold
        }
    return analysis

def fulfillment_analysis(df):
    analysis = {}
    fulfillment_col = _get_column(df, ['Fulfilment', 'fulfilment', 'Fulfillment', 'fulfillment'])
    status_col = _get_column(df, ['Status', 'status', 'Order Status', 'order_status'])
    channel_col = _get_column(df, ['Sales Channel', 'sales_channel', 'Channel', 'channel'])
    amount_col = _get_column(df, ['Amount', 'amount', 'Sales', 'sales'])
    if fulfillment_col:
        fulfillment_stats = df.groupby(fulfillment_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count'
        }).round(2)
        analysis['fulfillment_performance'] = fulfillment_stats.to_dict()
        fulfillment_dist = df[fulfillment_col].value_counts(normalize=True).round(3)
        analysis['fulfillment_distribution'] = fulfillment_dist.to_dict()
    if status_col:
        status_stats = df.groupby(status_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count'
        }).round(2)
        analysis['status_performance'] = status_stats.to_dict()
        status_dist = df[status_col].value_counts(normalize=True).round(3)
        analysis['status_distribution'] = status_dist.to_dict()
        successful_statuses = ['Shipped', 'Delivered', 'shipped', 'delivered']
        successful_orders = df[status_col].isin(successful_statuses).sum()
        analysis['fulfillment_rate'] = successful_orders / len(df) if len(df) > 0 else 0
    if channel_col:
        channel_stats = df.groupby(channel_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count'
        }).round(2)
        analysis['channel_performance'] = channel_stats.to_dict()
        channel_dist = df[channel_col].value_counts(normalize=True).round(3)
        analysis['channel_distribution'] = channel_dist.to_dict()
    return analysis

def geographical_analysis(df):
    analysis = {}
    state_col = _get_column(df, ['State', 'state', 'ship-state', 'Ship State'])
    city_col = _get_column(df, ['City', 'city', 'ship-city', 'Ship City'])
    postal_col = _get_column(df, ['postal-code', 'Postal Code', 'ZIP', 'zip'])
    amount_col = _get_column(df, ['Amount', 'amount', 'Sales', 'sales'])
    if state_col:
        state_stats = df.groupby(state_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count'
        }).round(2)
        analysis['state_performance'] = state_stats.to_dict()
        if amount_col:
            top_states_revenue = df.groupby(state_col)[amount_col].sum().sort_values(ascending=False).head(10)
            analysis['top_states_by_revenue'] = top_states_revenue.to_dict()
        top_states_orders = df[state_col].value_counts().head(10)
        analysis['top_states_by_orders'] = top_states_orders.to_dict()
        analysis['geographical_diversity'] = {
            'total_states': df[state_col].nunique(),
            'state_concentration_ratio': df[state_col].value_counts().iloc[0] / len(df),
            'states_with_single_order': (df[state_col].value_counts() == 1).sum()
        }
    if city_col:
        city_stats = df.groupby(city_col).agg({
            amount_col: ['sum', 'mean', 'count'] if amount_col else 'count'
        }).round(2)
        if amount_col:
            top_cities_revenue = df.groupby(city_col)[amount_col].sum().sort_values(ascending=False).head(15)
            analysis['top_cities_by_revenue'] = top_cities_revenue.to_dict()
        top_cities_orders = df[city_col].value_counts().head(15)
        analysis['top_cities_by_orders'] = top_cities_orders.to_dict()
        analysis['city_diversity'] = {
            'total_cities': df[city_col].nunique(),
            'city_concentration_ratio': df[city_col].value_counts().iloc[0] / len(df),
            'cities_with_single_order': (df[city_col].value_counts() == 1).sum()
        }
    if postal_col:
        postal_stats = df[postal_col].value_counts().head(20)
        analysis['top_postal_codes'] = postal_stats.to_dict()
        analysis['postal_diversity'] = {
            'total_postal_codes': df[postal_col].nunique(),
            'postal_concentration_ratio': df[postal_col].value_counts().iloc[0] / len(df)
        }
    return analysis

def customer_segmentation_analysis(df):
    analysis = {}
    amount_col = _get_column(df, ['Amount', 'amount', 'Sales', 'sales'])
    qty_col = _get_column(df, ['Qty', 'qty', 'Quantity', 'quantity'])
    size_col = _get_column(df, ['Size', 'size', 'Product Size', 'product_size'])
    category_col = _get_column(df, ['Category', 'category', 'Product Category', 'product_category'])
    if amount_col:
        q75 = df[amount_col].quantile(0.75)
        q50 = df[amount_col].quantile(0.50)
        q25 = df[amount_col].quantile(0.25)
        df['customer_segment'] = pd.cut(df[amount_col], 
                                     bins=[0, q25, q50, q75, np.inf], 
                                     labels=['Low Value', 'Medium Value', 'High Value', 'Premium'])
        segment_analysis = df.groupby('customer_segment').agg({
            amount_col: ['count', 'sum', 'mean'],
            qty_col: ['sum', 'mean'] if qty_col else amount_col
        }).round(2)
        analysis['customer_segments'] = segment_analysis.to_dict()
        segment_dist = df['customer_segment'].value_counts(normalize=True).round(3)
        analysis['segment_distribution'] = segment_dist.to_dict()
    if qty_col:
        bulk_threshold = df[qty_col].quantile(0.8)
        analysis['purchase_behavior'] = {
            'bulk_buyers_percentage': (df[qty_col] >= bulk_threshold).mean(),
            'average_quantity_per_order': df[qty_col].mean(),
            'bulk_threshold': bulk_threshold
        }
    if size_col and amount_col:
        size_preference = df.groupby(size_col)[amount_col].agg(['count', 'mean', 'sum']).round(2)
        analysis['size_preferences'] = size_preference.to_dict()
    if category_col and amount_col:
        category_preference = df.groupby(category_col)[amount_col].agg(['count', 'mean', 'sum']).round(2)
        analysis['category_preferences'] = category_preference.to_dict()
    if 'month' in df.columns and amount_col:
        seasonal_behavior = df.groupby('month')[amount_col].agg(['count', 'mean', 'sum']).round(2)
        analysis['seasonal_behavior'] = seasonal_behavior.to_dict()
    return analysis

def business_insights_and_recommendations(insights):
    insights_out = {}
    recommendations = []
    if 'sales_overview' in insights:
        sales_data = insights['sales_overview']
        if 'sales_percentiles' in sales_data:
            top_10_percent_threshold = sales_data['sales_percentiles']['90th']
            insights_out['revenue_concentration'] = {
                'high_value_order_threshold': top_10_percent_threshold,
                'average_order_value': sales_data['average_order_value']
            }
            if sales_data['average_order_value'] < top_10_percent_threshold * 0.5:
                recommendations.append({
                    'area': 'Revenue Optimization',
                    'insight': 'Average order value is significantly lower than top-performing orders',
                    'recommendation': 'Implement upselling and cross-selling strategies to increase AOV',
                    'priority': 'High'
                })
    if 'product_analysis' in insights:
        product_data = insights['product_analysis']
        if 'top_categories_by_revenue' in product_data:
            top_categories = list(product_data['top_categories_by_revenue'].keys())[:3]
            insights_out['top_performing_categories'] = top_categories
            recommendations.append({
                'area': 'Product Strategy',
                'insight': f'Top 3 categories ({", ".join(top_categories)}) drive significant revenue',
                'recommendation': 'Focus marketing budget and inventory on these high-performing categories',
                'priority': 'High'
            })
    if 'geographical_analysis' in insights:
        geo_data = insights['geographical_analysis']
        if 'geographical_diversity' in geo_data:
            concentration_ratio = geo_data['geographical_diversity']['state_concentration_ratio']
            if concentration_ratio > 0.3:
                recommendations.append({
                    'area': 'Market Expansion',
                    'insight': f'High geographical concentration ({concentration_ratio:.1%} from top state)',
                    'recommendation': 'Develop targeted marketing campaigns for underperforming states',
                    'priority': 'Medium'
                })
    if 'fulfillment_analysis' in insights:
        fulfillment_data = insights['fulfillment_analysis']
        if 'fulfillment_rate' in fulfillment_data:
            fulfillment_rate = fulfillment_data['fulfillment_rate']
            if fulfillment_rate < 0.95:
                recommendations.append({
                    'area': 'Operations',
                    'insight': f'Fulfillment rate is {fulfillment_rate:.1%}',
                    'recommendation': 'Investigate and address fulfillment issues to improve customer satisfaction',
                    'priority': 'High'
                })
    if 'customer_segmentation' in insights:
        customer_data = insights['customer_segmentation']
        if 'segment_distribution' in customer_data:
            premium_percentage = customer_data['segment_distribution'].get('Premium', 0)
            if premium_percentage < 0.1:
                recommendations.append({
                    'area': 'Customer Development',
                    'insight': f'Only {premium_percentage:.1%} of customers are in premium segment',
                    'recommendation': 'Develop loyalty programs and premium offerings to increase customer value',
                    'priority': 'Medium'
                })
    if 'sales_overview' in insights and 'monthly_performance' in insights['sales_overview']:
        monthly_data = insights['sales_overview']['monthly_performance']
        if 'sum' in monthly_data:
            monthly_revenue = monthly_data['sum']
            peak_month = max(monthly_revenue, key=monthly_revenue.get)
            low_month = min(monthly_revenue, key=monthly_revenue.get)
            insights_out['seasonal_patterns'] = {
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonality_ratio': monthly_revenue[peak_month] / monthly_revenue[low_month]
            }
            recommendations.append({
                'area': 'Inventory Management',
                'insight': f'Strong seasonality with peak in month {peak_month}',
                'recommendation': 'Adjust inventory levels and marketing spend based on seasonal patterns',
                'priority': 'Medium'
            })
    business_analysis = {
        'key_insights': insights_out,
        'recommendations': recommendations,
        'priority_actions': [r for r in recommendations if r['priority'] == 'High']
    }
    return business_analysis

def generate_comprehensive_report(df):
    """
    Run all analyses and return a report dictionary
    """
    df = prepare_data(df)
    insights = {}
    insights['sales_overview'] = sales_overview_analysis(df)
    insights['product_analysis'] = product_analysis(df)
    insights['fulfillment_analysis'] = fulfillment_analysis(df)
    insights['geographical_analysis'] = geographical_analysis(df)
    insights['customer_segmentation'] = customer_segmentation_analysis(df)
    insights['business_recommendations'] = business_insights_and_recommendations(insights)
    report = {
        'executive_summary': _generate_executive_summary(insights),
        'detailed_analysis': insights,
        'business_insights': insights['business_recommendations'],
        'data_quality_report': _generate_data_quality_report(df)
    }
    return report

def _generate_executive_summary(insights):
    summary = {}
    if 'sales_overview' in insights:
        sales_data = insights['sales_overview']
        summary['total_revenue'] = sales_data.get('total_revenue', 0)
        summary['total_orders'] = sales_data.get('total_orders', 0)
        summary['average_order_value'] = sales_data.get('average_order_value', 0)
    if 'product_analysis' in insights:
        product_data = insights['product_analysis']
        if 'top_categories_by_revenue' in product_data:
            summary['top_category'] = list(product_data['top_categories_by_revenue'].keys())[0]
    if 'geographical_analysis' in insights:
        geo_data = insights['geographical_analysis']
        if 'top_states_by_revenue' in geo_data:
            summary['top_state'] = list(geo_data['top_states_by_revenue'].keys())[0]
    return summary

def _generate_data_quality_report(df):
    quality_report = {}
    missing_values = df.isnull().sum()
    quality_report['missing_values'] = missing_values[missing_values > 0].to_dict()
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    quality_report['data_completeness'] = (total_cells - missing_cells) / total_cells
    quality_report['duplicate_records'] = df.duplicated().sum()
    quality_report['data_types'] = df.dtypes.astype(str).to_dict()
    return quality_report

def _get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None