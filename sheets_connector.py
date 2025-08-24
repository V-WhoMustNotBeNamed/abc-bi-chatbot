import gspread
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from data_cleaner import DataCleaner

class SheetsConnector:
    """Handles connection and data extraction from Google Sheets"""
    
    def __init__(self, credentials_file: str):
        self.credentials_file = credentials_file
        self.gc = None
        self.workbook = None
        self.dataframes = {}
        self.cleaning_metadata = {}
        self.data_cleaner = DataCleaner()
        
    def connect(self, sheet_url: str) -> bool:
        """Connect to Google Sheets and load all worksheets"""
        try:
            # Authenticate with Google Sheets
            self.gc = gspread.service_account(filename=self.credentials_file)
            
            # Open the workbook
            self.workbook = self.gc.open_by_url(sheet_url)
            
            # Load all worksheets into DataFrames
            self.dataframes = {}
            self.cleaning_metadata = {}
            
            for worksheet in self.workbook.worksheets():
                try:
                    # Get all records from the worksheet
                    records = worksheet.get_all_records()
                    if records:  # Only process if there's data
                        df = pd.DataFrame(records)
                        
                        # Apply comprehensive data cleaning
                        df_clean, metadata = self.data_cleaner.clean_dataframe(df, worksheet.title)
                        
                        self.dataframes[worksheet.title] = df_clean
                        self.cleaning_metadata[worksheet.title] = metadata
                except Exception as e:
                    print(f"Warning: Could not load worksheet '{worksheet.title}': {e}")
                    continue
                    
            return len(self.dataframes) > 0
            
        except Exception as e:
            print(f"Error connecting to sheets: {e}")
            return False
    
    
    def get_sheet_info(self) -> Dict:
        """Get basic information about the connected sheet including schema details"""
        if not self.workbook:
            return {}
            
        worksheet_info = {}
        for name, df in self.dataframes.items():
            metadata = self.cleaning_metadata.get(name, {})
            worksheet_info[name] = {
                'rows': len(df),
                'columns': list(df.columns),
                'original_columns': metadata.get('original_columns', []),
                'column_mapping': metadata.get('column_mapping', {}),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                'type_conversions': metadata.get('type_conversions', {})
            }
            
        return {
            'title': self.workbook.title,
            'worksheets': list(self.dataframes.keys()),
            'total_rows': sum(len(df) for df in self.dataframes.values()),
            'worksheet_info': worksheet_info,
            'cleaning_summary': self.get_cleaning_summary()
        }
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get a summary of all data cleaning operations"""
        if not self.cleaning_metadata:
            return {}
        
        return self.data_cleaner.generate_schema_summary(list(self.cleaning_metadata.values()))
    
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
            self.cleaning_metadata = {}
            
            for worksheet in self.workbook.worksheets():
                try:
                    records = worksheet.get_all_records()
                    if records:
                        df = pd.DataFrame(records)
                        
                        # Apply comprehensive data cleaning
                        df_clean, metadata = self.data_cleaner.clean_dataframe(df, worksheet.title)
                        
                        self.dataframes[worksheet.title] = df_clean
                        self.cleaning_metadata[worksheet.title] = metadata
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