import gspread
import pandas as pd
from typing import Dict, List, Optional

class SheetsConnector:
    """Handles connection and data extraction from Google Sheets"""
    
    def __init__(self, credentials_file: str):
        self.credentials_file = credentials_file
        self.gc = None
        self.workbook = None
        self.dataframes = {}
        
    def connect(self, sheet_url: str) -> bool:
        """Connect to Google Sheets and load all worksheets"""
        try:
            # Authenticate with Google Sheets
            self.gc = gspread.service_account(filename=self.credentials_file)
            
            # Open the workbook
            self.workbook = self.gc.open_by_url(sheet_url)
            
            # Load all worksheets into DataFrames
            self.dataframes = {}
            for worksheet in self.workbook.worksheets():
                try:
                    # Get all records from the worksheet
                    records = worksheet.get_all_records()
                    if records:  # Only process if there's data
                        df = pd.DataFrame(records)
                        
                        # Basic data cleaning
                        df = self._clean_dataframe(df)
                        
                        self.dataframes[worksheet.title] = df
                except Exception as e:
                    print(f"Warning: Could not load worksheet '{worksheet.title}': {e}")
                    continue
                    
            return len(self.dataframes) > 0
            
        except Exception as e:
            print(f"Error connecting to sheets: {e}")
            return False
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning for the DataFrame"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns that might be stored as strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If we can convert most values, keep the numeric version
                    if numeric_series.notna().sum() / len(df) > 0.5:
                        df[col] = numeric_series
        
        return df
    
    def get_sheet_info(self) -> Dict:
        """Get basic information about the connected sheet"""
        if not self.workbook:
            return {}
            
        return {
            'title': self.workbook.title,
            'worksheets': list(self.dataframes.keys()),
            'total_rows': sum(len(df) for df in self.dataframes.values()),
            'worksheet_info': {
                name: {
                    'rows': len(df),
                    'columns': list(df.columns)
                }
                for name, df in self.dataframes.items()
            }
        }
    
    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all loaded DataFrames"""
        return self.dataframes
    
    def refresh_data(self) -> bool:
        """Refresh data from the connected sheet"""
        if not self.workbook:
            return False
            
        try:
            # Reload all worksheets
            self.dataframes = {}
            for worksheet in self.workbook.worksheets():
                try:
                    records = worksheet.get_all_records()
                    if records:
                        df = pd.DataFrame(records)
                        df = self._clean_dataframe(df)
                        self.dataframes[worksheet.title] = df
                except Exception as e:
                    print(f"Warning: Could not refresh worksheet '{worksheet.title}': {e}")
                    continue
                    
            return True
            
        except Exception as e:
            print(f"Error refreshing data: {e}")
            return False
    
    def get_sample_data(self, worksheet_name: str, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """Get sample data from a specific worksheet"""
        if worksheet_name in self.dataframes:
            return self.dataframes[worksheet_name].head(n_rows)
        return None