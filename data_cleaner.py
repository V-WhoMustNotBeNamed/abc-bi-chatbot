import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime

class DataCleaner:
    """Handles data cleaning, column standardization, and type inference for Google Sheets data"""
    
    def __init__(self):
        # Keywords for identifying column types
        self.numeric_keywords = [
            'amount', 'quantity', 'revenue', 'cost', 'price', 'value', 
            'spent', 'profit', 'total', 'sum', 'count', 'number_of'
        ]
        
        self.date_keywords = [
            'date', 'time', 'created', 'updated', 'modified', 'timestamp',
            'year', 'month', 'day', 'when'
        ]
        
        self.id_keywords = [
            'id', 'number', 'code', 'reference', 'ref', 'invoice', 'order',
            'transaction', 'customer_id', 'product_id', 'sale_id'
        ]
        
        self.descriptive_keywords = [
            'name', 'type', 'category', 'status', 'description', 'comment',
            'notes', 'title', 'label', 'tag'
        ]
        
        self.boolean_keywords = [
            'is_', 'has_', 'in_stock', 'active', 'enabled', 'available'
        ]
        
    def clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean a dataframe and return the cleaned version with metadata
        
        Returns:
            - Cleaned DataFrame
            - Cleaning metadata including column mappings and type conversions
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Initialize metadata
        metadata = {
            'sheet_name': sheet_name,
            'original_columns': list(df.columns),
            'column_mapping': {},
            'type_conversions': {},
            'cleaning_notes': []
        }
        
        # Step 1: Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Step 2: Clean and standardize column names
        df_clean, column_mapping = self._standardize_column_names(df_clean)
        metadata['column_mapping'] = column_mapping
        
        # Step 3: Infer and convert data types
        df_clean, type_conversions = self._infer_and_convert_types(df_clean)
        metadata['type_conversions'] = type_conversions
        
        # Step 4: Additional cleaning
        df_clean = self._additional_cleaning(df_clean, metadata)
        
        # Generate summary
        metadata['final_columns'] = list(df_clean.columns)
        metadata['final_dtypes'] = df_clean.dtypes.to_dict()
        metadata['row_count'] = len(df_clean)
        
        return df_clean, metadata
    
    def _standardize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Convert column names to snake_case and handle duplicates"""
        column_mapping = {}
        new_columns = []
        seen_columns = set()
        
        for col in df.columns:
            # Convert to string and strip whitespace
            col_str = str(col).strip()
            
            # Convert to snake_case
            # Replace spaces and special characters with underscores
            clean_col = re.sub(r'[^\w\s]', '', col_str)
            clean_col = re.sub(r'\s+', '_', clean_col)
            clean_col = clean_col.lower()
            
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            
            # Handle empty column names
            if not clean_col:
                clean_col = 'column'
            
            # Handle duplicates
            original_clean_col = clean_col
            counter = 2
            while clean_col in seen_columns:
                clean_col = f"{original_clean_col}_{counter}"
                counter += 1
            
            seen_columns.add(clean_col)
            new_columns.append(clean_col)
            column_mapping[col] = clean_col
        
        df.columns = new_columns
        return df, column_mapping
    
    def _infer_and_convert_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Infer and convert data types based on column content and names"""
        type_conversions = {}
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Skip if column is already numeric or datetime
            if df[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
                type_conversions[col] = f"{original_dtype} → {original_dtype} (no change)"
                continue
            
            # Check if column should be numeric based on name
            if self._should_be_numeric(col):
                df[col], converted = self._convert_to_numeric(df[col])
                if converted:
                    type_conversions[col] = f"{original_dtype} → float64"
                else:
                    type_conversions[col] = f"{original_dtype} → object (failed numeric conversion)"
            
            # Check if column should be datetime
            elif self._should_be_datetime(col):
                df[col], converted = self._convert_to_datetime(df[col])
                if converted:
                    type_conversions[col] = f"{original_dtype} → datetime64"
                else:
                    type_conversions[col] = f"{original_dtype} → object (failed date conversion)"
            
            # Check if column should be boolean
            elif self._should_be_boolean(col, df[col]):
                df[col], converted = self._convert_to_boolean(df[col])
                if converted:
                    type_conversions[col] = f"{original_dtype} → bool"
                else:
                    type_conversions[col] = f"{original_dtype} → object (failed boolean conversion)"
            
            # Check if column should be string (for IDs and descriptive fields)
            elif self._should_be_string(col):
                df[col] = df[col].astype(str).replace('nan', '')
                type_conversions[col] = f"{original_dtype} → object (enforced string)"
            
            else:
                # For mixed-type columns, convert to string for consistency
                if self._is_mixed_type(df[col]):
                    df[col] = df[col].astype(str).replace('nan', '')
                    type_conversions[col] = f"{original_dtype} → object (mixed types)"
                else:
                    type_conversions[col] = f"{original_dtype} → {original_dtype} (no change)"
        
        return df, type_conversions
    
    def _should_be_numeric(self, col_name: str) -> bool:
        """Check if column should be numeric based on its name"""
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in self.numeric_keywords)
    
    def _should_be_datetime(self, col_name: str) -> bool:
        """Check if column should be datetime based on its name"""
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in self.date_keywords)
    
    def _should_be_string(self, col_name: str) -> bool:
        """Check if column should be string based on its name"""
        col_lower = col_name.lower()
        return (any(keyword in col_lower for keyword in self.id_keywords) or
                any(keyword in col_lower for keyword in self.descriptive_keywords))
    
    def _should_be_boolean(self, col_name: str, series: pd.Series) -> bool:
        """Check if column should be boolean based on name and content"""
        col_lower = col_name.lower()
        
        # Check name patterns
        if any(col_lower.startswith(keyword) for keyword in self.boolean_keywords):
            return True
        
        # Check content
        unique_values = series.dropna().astype(str).str.lower().unique()
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
        
        return len(unique_values) <= 2 and all(val in boolean_values for val in unique_values)
    
    def _convert_to_numeric(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Convert series to numeric, handling currency symbols and commas"""
        try:
            # Remove currency symbols and commas
            if series.dtype == 'object':
                series_clean = series.astype(str).str.replace('$', '', regex=False)
                series_clean = series_clean.str.replace(',', '', regex=False)
                series_clean = series_clean.str.strip()
            else:
                series_clean = series
            
            # Convert to numeric
            numeric_series = pd.to_numeric(series_clean, errors='coerce')
            
            # Check if conversion was successful for most values
            non_null_count = series.notna().sum()
            if non_null_count > 0:
                success_rate = numeric_series.notna().sum() / non_null_count
                if success_rate > 0.8:  # 80% threshold
                    return numeric_series, True
            
            return series, False
            
        except Exception:
            return series, False
    
    def _convert_to_datetime(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Convert series to datetime"""
        try:
            # Try to parse dates with format='mixed' to avoid warnings
            date_series = pd.to_datetime(series, errors='coerce', format='mixed')
            
            # Check if conversion was successful for most values
            non_null_count = series.notna().sum()
            if non_null_count > 0:
                success_rate = date_series.notna().sum() / non_null_count
                if success_rate > 0.8:  # 80% threshold
                    return date_series, True
            
            return series, False
            
        except Exception:
            return series, False
    
    def _convert_to_boolean(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Convert series to boolean"""
        try:
            # Create mapping for boolean values
            bool_map = {
                'true': True, 'false': False,
                'yes': True, 'no': False,
                '1': True, '0': False,
                't': True, 'f': False,
                'y': True, 'n': False
            }
            
            # Convert to lowercase string for mapping
            series_lower = series.astype(str).str.lower()
            
            # Apply mapping
            bool_series = series_lower.map(bool_map)
            
            # Check if all non-null values were mapped
            if series_lower.notna().sum() == bool_series.notna().sum():
                return bool_series, True
            
            return series, False
            
        except Exception:
            return series, False
    
    def _is_mixed_type(self, series: pd.Series) -> bool:
        """Check if a series contains mixed data types"""
        if series.dtype != 'object':
            return False
        
        # Sample the series
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Check if values can be converted to different types
        numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
        date_count = pd.to_datetime(sample, errors='coerce', format='mixed').notna().sum()
        
        # If some values are numeric and some are not, it's mixed
        if 0 < numeric_count < len(sample):
            return True
        
        # If some values are dates and some are not, it's mixed
        if 0 < date_count < len(sample):
            return True
        
        return False
    
    def _additional_cleaning(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Perform additional cleaning operations"""
        # Trim string values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' strings with empty strings
                df[col] = df[col].replace('nan', '')
        
        # Add cleaning note
        metadata['cleaning_notes'].append("Trimmed whitespace from string columns")
        metadata['cleaning_notes'].append("Replaced 'nan' strings with empty strings")
        
        return df
    
    def generate_schema_summary(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of all schemas after cleaning"""
        summary = {
            'tables': {},
            'total_rows': 0,
            'cleaning_timestamp': datetime.now().isoformat()
        }
        
        for metadata in metadata_list:
            sheet_name = metadata['sheet_name']
            summary['tables'][sheet_name] = {
                'row_count': metadata['row_count'],
                'columns': metadata['final_columns'],
                'column_mapping': metadata['column_mapping'],
                'data_types': metadata['final_dtypes'],
                'type_conversions': metadata['type_conversions']
            }
            summary['total_rows'] += metadata['row_count']
        
        return summary