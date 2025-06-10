import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    A comprehensive data processing class for Amazon sales data
    """
    
    def __init__(self):
        self.df = None
        self.cleaned_df = None
        self.column_mapping = {}
        
    def load_data(self, file_path):
        """
        Load Amazon sales data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            pandas.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            self._identify_columns()
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _identify_columns(self):
        """
        Automatically identify column types based on common patterns
        """
        columns = self.df.columns.str.lower()
        
        # Common column patterns for Amazon data
        patterns = {
            'order_id': ['order', 'id'],
            'date': ['date', 'time'],
            'status': ['status', 'state'],
            'fulfillment': ['fulfil', 'ship', 'deliver'],
            'sales_channel': ['channel', 'platform'],
            'category': ['category', 'type', 'product'],
            'size': ['size'],
            'quantity': ['qty', 'quantity', 'count'],
            'amount': ['amount', 'price', 'cost', 'sales', 'revenue'],
            'location': ['city', 'state', 'location', 'address'],
            'customer': ['customer', 'buyer', 'user']
        }
        
        for col_type, keywords in patterns.items():
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    self.column_mapping[col_type] = col
                    break
        
        print(f"🔍 Identified columns: {self.column_mapping}")
    
    def explore_data(self):
        """
        Perform comprehensive data exploration
        
        Returns:
            dict: Exploration results
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None
        
        exploration_results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        print("=" * 60)
        print("📈 DATA EXPLORATION REPORT")
        print("=" * 60)
        print(f"📊 Dataset Shape: {exploration_results['shape']}")
        print(f"🔢 Total Records: {exploration_results['shape'][0]:,}")
        print(f"📋 Total Columns: {exploration_results['shape'][1]}")
        print(f"💾 Memory Usage: {exploration_results['memory_usage'] / 1024**2:.2f} MB")
        print(f"🔄 Duplicate Records: {exploration_results['duplicates']:,}")
        
        print(f"\n📊 Column Information:")
        for col, dtype in exploration_results['dtypes'].items():
            missing = exploration_results['missing_values'][col]
            missing_pct = (missing / len(self.df)) * 100
            print(f"  • {col}: {dtype} (Missing: {missing} / {missing_pct:.1f}%)")
        
        print(f"\n🔢 Numeric Columns: {len(exploration_results['numeric_columns'])}")
        print(f"📝 Categorical Columns: {len(exploration_results['categorical_columns'])}")
        
        return exploration_results
    
    def clean_data(self):
        """
        Comprehensive data cleaning pipeline
        
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None
        
        print("🧹 Starting data cleaning process...")
        self.cleaned_df = self.df.copy()
        
        # 1. Handle duplicates
        initial_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.cleaned_df)
        if duplicates_removed > 0:
            print(f"✅ Removed {duplicates_removed:,} duplicate rows")
        
        self._convert_date_columns()
        
        self._clean_numeric_columns()
        
        self._clean_categorical_columns()
        
        self._handle_missing_values()
        
        self._create_derived_features()
        
        print(f"✅ Data cleaning completed!")
        print(f"📊 Final dataset shape: {self.cleaned_df.shape}")
        
        return self.cleaned_df
    
    def _convert_date_columns(self):
        """Convert date columns to datetime format"""
        date_columns = [col for col in self.cleaned_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        for col in date_columns:
            try:
                self.cleaned_df[col] = pd.to_datetime(self.cleaned_df[col], errors='coerce')
                print(f"✅ Converted {col} to datetime")
            except Exception as e:
                print(f"⚠️ Could not convert {col} to datetime: {e}")
    
    def _clean_numeric_columns(self):
        """Clean and validate numeric columns"""
        numeric_columns = self.cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if any(keyword in col.lower() for keyword in ['qty', 'quantity', 'amount', 'price']):
                negative_count = (self.cleaned_df[col] < 0).sum()
                if negative_count > 0:
                    self.cleaned_df = self.cleaned_df[self.cleaned_df[col] >= 0]
                    print(f"✅ Removed {negative_count} negative values from {col}")
            
            Q1 = self.cleaned_df[col].quantile(0.25)
            Q3 = self.cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.cleaned_df[col] < lower_bound) | (self.cleaned_df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"⚠️ Found {outliers} outliers in {col} (not removed - review manually)")
    
    def _clean_categorical_columns(self):
        """Clean categorical columns"""
        categorical_columns = self.cleaned_df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if self.cleaned_df[col].dtype == 'datetime64[ns]':
                continue
                
            if self.cleaned_df[col].dtype == 'object':
                self.cleaned_df[col] = self.cleaned_df[col].astype(str).str.strip()
                
            if any(keyword in col.lower() for keyword in ['status', 'category', 'size']):
                self.cleaned_df[col] = self.cleaned_df[col].str.title()
                print(f"✅ Standardized case for {col}")
    
    def _handle_missing_values(self):
        """Handle missing values based on column type and importance"""
        missing_summary = self.cleaned_df.isnull().sum()
        missing_columns = missing_summary[missing_summary > 0]
        
        if len(missing_columns) == 0:
            print("✅ No missing values found")
            return
        
        print(f"📋 Handling missing values in {len(missing_columns)} columns:")
        
        for col in missing_columns.index:
            missing_count = missing_columns[col]
            missing_pct = (missing_count / len(self.cleaned_df)) * 100
            
            if missing_pct > 50:
                print(f"⚠️ {col}: {missing_pct:.1f}% missing - Consider dropping column")
            elif self.cleaned_df[col].dtype in ['int64', 'float64']:
                median_val = self.cleaned_df[col].median()
                self.cleaned_df[col].fillna(median_val, inplace=True)
                print(f"✅ {col}: Filled {missing_count} missing values with median ({median_val})")
            else:
                if len(self.cleaned_df[col].mode()) > 0:
                    mode_val = self.cleaned_df[col].mode()[0]
                    self.cleaned_df[col].fillna(mode_val, inplace=True)
                    print(f"✅ {col}: Filled {missing_count} missing values with mode ({mode_val})")
                else:
                    self.cleaned_df[col].fillna('Unknown', inplace=True)
                    print(f"✅ {col}: Filled {missing_count} missing values with 'Unknown'")
    
    def _create_derived_features(self):
        """Create useful derived features"""
        print("🔧 Creating derived features...")
        
        date_col = self.column_mapping.get('date')
        if date_col and date_col in self.cleaned_df.columns:
            self.cleaned_df['year'] = self.cleaned_df[date_col].dt.year
            self.cleaned_df['month'] = self.cleaned_df[date_col].dt.month
            self.cleaned_df['day_of_week'] = self.cleaned_df[date_col].dt.day_name()
            self.cleaned_df['is_weekend'] = self.cleaned_df[date_col].dt.dayofweek >= 5
            print("✅ Created date-based features")
        
        # Amount-based features
        amount_col = self.column_mapping.get('amount')
        if amount_col and amount_col in self.cleaned_df.columns:
            self.cleaned_df[amount_col] = pd.to_numeric(self.cleaned_df[amount_col], errors='coerce')
            self.cleaned_df['order_value_category'] = pd.cut(
                self.cleaned_df[amount_col],
                bins=[0, 50, 200, 500, float('inf')],
                labels=['Low', 'Medium', 'High', 'Premium']
            )
            print("✅ Created order value categories")
        
        qty_col = self.column_mapping.get('quantity')
        if qty_col and qty_col in self.cleaned_df.columns:
            self.cleaned_df['bulk_order'] = self.cleaned_df[qty_col] > self.cleaned_df[qty_col].quantile(0.8)
            print("✅ Created bulk order indicator")
    
    def get_data_quality_report(self):
        """
        Generate a comprehensive data quality report
        
        Returns:
            dict: Data quality metrics
        """
        if self.cleaned_df is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return None
        
        quality_report = {
            'total_records': len(self.cleaned_df),
            'total_columns': len(self.cleaned_df.columns),
            'missing_values': self.cleaned_df.isnull().sum().sum(),
            'duplicate_records': self.cleaned_df.duplicated().sum(),
            'data_completeness': (1 - self.cleaned_df.isnull().sum().sum() / (len(self.cleaned_df) * len(self.cleaned_df.columns))) * 100,
            'column_types': self.cleaned_df.dtypes.value_counts().to_dict()
        }
        
        print("=" * 60)
        print("📊 DATA QUALITY REPORT")
        print("=" * 60)
        print(f"📋 Total Records: {quality_report['total_records']:,}")
        print(f"📊 Total Columns: {quality_report['total_columns']}")
        print(f"❌ Missing Values: {quality_report['missing_values']:,}")
        print(f"🔄 Duplicate Records: {quality_report['duplicate_records']:,}")
        print(f"✅ Data Completeness: {quality_report['data_completeness']:.2f}%")
        
        return quality_report
    
    def save_cleaned_data(self, file_path):
        """
        Save cleaned data to CSV file
        
        Args:
            file_path (str): Path to save the cleaned data
        """
        if self.cleaned_df is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return
        
        try:
            self.cleaned_df.to_csv(file_path, index=False)
            print(f"✅ Cleaned data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def get_column_summary(self):
        """
        Get detailed summary of each column
        
        Returns:
            pandas.DataFrame: Column summary
        """
        if self.cleaned_df is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return None
        
        summary_data = []
        
        for col in self.cleaned_df.columns:
            col_info = {
                'Column': col,
                'Data_Type': str(self.cleaned_df[col].dtype),
                'Non_Null_Count': self.cleaned_df[col].count(),
                'Null_Count': self.cleaned_df[col].isnull().sum(),
                'Null_Percentage': (self.cleaned_df[col].isnull().sum() / len(self.cleaned_df)) * 100,
                'Unique_Values': self.cleaned_df[col].nunique(),
                'Memory_Usage_MB': self.cleaned_df[col].memory_usage(deep=True) / 1024**2
            }
            
            if self.cleaned_df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'Min': self.cleaned_df[col].min(),
                    'Max': self.cleaned_df[col].max(),
                    'Mean': self.cleaned_df[col].mean(),
                    'Median': self.cleaned_df[col].median(),
                    'Std': self.cleaned_df[col].std()
                })
                
            summary_data.append(col_info)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df