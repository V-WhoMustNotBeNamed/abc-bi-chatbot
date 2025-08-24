import streamlit as st
import pandas as pd
from sheets_connector import SheetsConnector
from analytics_engine import AnalyticsEngine
import io
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os, tempfile
from dotenv import load_dotenv
import time
import json
import sqlparse

# Load environment variables from .env file
load_dotenv()

# Open AI API Key
openai_api_key = os.getenv("openaitestkey")

creds_json = os.environ["CREDENTIALS_JSON"]

# Create a temp file that behaves like "credentials.json"
with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
    f.write(creds_json)
    temp_creds_path = f.name

# Set seaborn style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def extract_source_columns(sql_query):
    """Extract source column names from SQL query (not aliases)"""
    try:
        # Parse the SQL query
        parsed = sqlparse.parse(sql_query)[0]
        
        # Extract column names using regex patterns
        columns = set()
        
        # Pattern to match column names (word characters, underscores)
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Remove SQL keywords, functions, and aliases
        sql_keywords = {
            'select', 'from', 'where', 'group', 'by', 'order', 'having', 'limit',
            'join', 'inner', 'left', 'right', 'outer', 'on', 'and', 'or', 'not',
            'in', 'between', 'like', 'as', 'asc', 'desc', 'distinct', 'all',
            'sum', 'count', 'avg', 'max', 'min', 'strftime', 'date', 'cast',
            'case', 'when', 'then', 'else', 'end', 'is', 'null', 'table'
        }
        
        # Extract all potential column names
        potential_columns = re.findall(column_pattern, sql_query.lower())
        
        # Filter out SQL keywords and numeric values
        for col in potential_columns:
            if col not in sql_keywords and not col.isdigit():
                # Skip alias names (words that come after 'as')
                sql_lower = sql_query.lower()
                if f'as {col}' not in sql_lower:
                    # Also check if it's not a table name (appears after FROM or JOIN)
                    if not (f'from {col}' in sql_lower or f'join {col}' in sql_lower):
                        columns.add(col)
        
        # Special handling for date columns in strftime
        strftime_matches = re.findall(r"strftime\([^,]+,\s*([a-zA-Z_][a-zA-Z0-9_]*)\)", sql_query, re.IGNORECASE)
        columns.update(strftime_matches)
        
        return sorted(list(columns))
    except Exception as e:
        # If parsing fails, return empty list
        return []

def format_numbers_in_dataframe(df):
    """Format numbers with proper currency and comma formatting"""
    df_formatted = df.copy()
    
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['int64', 'float64']:
            # Check if it's revenue/amount related
            if any(keyword in col.lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit', 'value', 'spent']):
                # Format as currency
                df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else x)
            else:
                # Format as number with commas
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else x)
    
    return df_formatted

def format_dates_in_dataframe(df):
    """Post-process DataFrame to ensure proper date formatting"""
    for col in df.columns:
        if 'month' in col.lower() or 'date' in col.lower():
            # If it's in YYYY-MM format, convert to "Month YYYY"
            if df[col].dtype == 'object':
                try:
                    # Handle YYYY-MM format
                    if df[col].str.match(r'^\d{4}-\d{2}$').any():
                        df[col] = pd.to_datetime(df[col] + '-01').dt.strftime('%B %Y')
                except:
                    pass  # Keep original if conversion fails
    return df

def sort_month_data(df):
    """Sort DataFrame by month chronologically"""
    if len(df.columns) >= 2:
        date_col = df.columns[0]
        try:
            # Try to convert month names back to sortable dates
            if 'month' in date_col.lower():
                # Create a temporary sort column
                df_sorted = df.copy()
                try:
                    df_sorted['_sort_date'] = pd.to_datetime(df_sorted[date_col], format='%B %Y')
                    df_sorted = df_sorted.sort_values('_sort_date')
                    df_sorted = df_sorted.drop('_sort_date', axis=1)
                    return df_sorted
                except:
                    # If that fails, try with YYYY-MM format
                    try:
                        df_sorted['_sort_date'] = pd.to_datetime(df_sorted[date_col])
                        df_sorted = df_sorted.sort_values('_sort_date')
                        df_sorted = df_sorted.drop('_sort_date', axis=1)
                        return df_sorted
                    except:
                        pass
        except:
            pass
    return df

def should_create_chart(df, question, sql_query=None):
    """Determine if a chart should be created based on data and question"""
    if len(df.columns) < 2:
        return False
    
    question_lower = question.lower()
    
    # Explicit chart request keywords
    chart_keywords = ['plot', 'chart', 'graph', 'visualize', 'show me', 'trend', 'compare', 'distribution']
    if any(keyword in question_lower for keyword in chart_keywords):
        return True
    
    # Time-series detection
    time_keywords = ['month', 'year', 'date', 'time', 'trend', 'over time', 'by month', 'by year', 'monthly', 'yearly', 'daily', 'weekly']
    if any(keyword in question_lower for keyword in time_keywords):
        return True
    
    # Check if first column is date-like
    first_col = df.columns[0].lower()
    if any(keyword in first_col for keyword in ['month', 'date', 'year', 'time', 'period']):
        return True
    
    # Check SQL for time-based grouping
    if sql_query:
        sql_lower = sql_query.lower()
        if 'group by' in sql_lower and any(func in sql_lower for func in ['strftime', 'date_trunc', 'extract']):
            return True
    
    # Aggregation queries (top N, comparisons)
    agg_keywords = ['top', 'best', 'worst', 'highest', 'lowest', 'most', 'least', 'ranking', 'by category', 'by type', 'by product', 'by customer']
    if any(keyword in question_lower for keyword in agg_keywords) and len(df) <= 20:
        return True
    
    # Numeric analysis keywords
    analysis_keywords = ['revenue', 'sales', 'profit', 'cost', 'amount', 'quantity', 'count', 'total', 'sum', 'average']
    second_col = df.columns[1].lower() if len(df.columns) > 1 else ''
    if any(keyword in second_col for keyword in analysis_keywords):
        # Check if it's a meaningful aggregation (not too many rows)
        if len(df) <= 20:
            return True
    
    return False

def create_chart_and_save(df, question):
    """Create a chart using seaborn and return it as base64 encoded image"""
    if len(df.columns) >= 2:
        try:
            # Create figure with seaborn style
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Extract numeric values from formatted strings
            y_values = []
            for val in df.iloc[:, 1]:
                if isinstance(val, str):
                    # Remove $ and commas
                    numeric_val = re.sub(r'[$,]', '', val)
                    try:
                        y_values.append(float(numeric_val))
                    except:
                        y_values.append(0)
                else:
                    y_values.append(val)
            
            # Create DataFrame for seaborn
            chart_df = pd.DataFrame({
                df.columns[0]: df.iloc[:, 0],
                df.columns[1]: y_values
            })
            
            # Determine chart type based on data and question
            question_lower = question.lower()
            first_col_lower = df.columns[0].lower()
            
            # Use line chart for time series data
            is_time_series = (
                any(keyword in first_col_lower for keyword in ['month', 'date', 'year', 'time', 'period']) or
                any(keyword in question_lower for keyword in ['trend', 'over time', 'monthly', 'yearly', 'daily', 'weekly'])
            )
            
            if is_time_series and len(chart_df) > 1:
                # Create line plot for time series
                ax.plot(range(len(chart_df)), y_values, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
                ax.fill_between(range(len(chart_df)), y_values, alpha=0.3, color='#2E86AB')
                ax.set_xticks(range(len(chart_df)))
                ax.set_xticklabels(chart_df.iloc[:, 0])
                
                # Add value labels on points
                for i, (x, y) in enumerate(zip(range(len(chart_df)), y_values)):
                    if any(keyword in df.columns[1].lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit']):
                        label = f'${y:,.0f}'
                    else:
                        label = f'{y:,.0f}'
                    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            else:
                # Create bar plot for comparisons
                bars = sns.barplot(
                    data=chart_df, 
                    x=df.columns[0], 
                    y=df.columns[1],
                    hue=df.columns[0],
                    palette="viridis",
                    legend=False,
                    ax=ax
                )
                
                # Add value labels on bars
                for i, bar in enumerate(bars.patches):
                    height = bar.get_height()
                    if any(keyword in df.columns[1].lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit']):
                        label = f'${height:,.0f}'
                    else:
                        label = f'{height:,.0f}'
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           label, ha='center', va='bottom', fontsize=10)
            
            # Customize the chart
            ax.set_title(f"Analysis: {question[:50]}{'...' if len(question) > 50 else ''}", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(df.columns[0], fontsize=12, fontweight='bold')
            ax.set_ylabel(df.columns[1], fontsize=12, fontweight='bold')
            
            # Format y-axis with currency if needed
            if any(keyword in df.columns[1].lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit']):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Rotate x-axis labels if needed
            if len(chart_df) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
        except Exception as e:
            print(f"Chart creation failed: {e}")
            return None
    return None

def export_to_excel(df, question, sql_query=None, chart_base64=None):
    """Export a single DataFrame to Excel with formatting, SQL, and chart"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        from openpyxl.styles import Font
        
        # Write the raw data for calculations
        df_raw = df.copy()
        # Remove formatting for raw data
        for col in df_raw.columns:
            if df_raw[col].dtype == 'object':
                try:
                    numeric_series = pd.to_numeric(df_raw[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
                    df_raw[col] = numeric_series.fillna(df_raw[col])
                except:
                    pass
        
        # Use simple sheet name
        sheet_name = "Data"
        df_raw.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add question as title in A1
        worksheet['A1'] = f"Query: {question}"
        worksheet['A1'].font = Font(bold=True, size=14)
        
        # Format the data with proper number formatting
        for col_idx, col in enumerate(df_raw.columns, 1):
            col_letter = chr(64 + col_idx)
            
            if any(keyword in col.lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit', 'value', 'spent']):
                for row in range(3, len(df_raw) + 3):
                    cell = worksheet[f'{col_letter}{row}']
                    cell.number_format = '$#,##0.00'
            else:
                if df_raw[col].dtype in ['int64', 'float64']:
                    for row in range(3, len(df_raw) + 3):
                        cell = worksheet[f'{col_letter}{row}']
                        cell.number_format = '#,##0'

        # Auto-fit all column widths
        for col_idx, col in enumerate(df_raw.columns, 1):
            col_letter = chr(64 + col_idx)
            column = worksheet.column_dimensions[col_letter]
            
            # Calculate max width needed
            max_length = 0
            for row in range(1, len(df_raw) + 3):  # Include header and title
                cell_value = str(worksheet[f'{col_letter}{row}'].value or '')
                max_length = max(max_length, len(cell_value))
            
            # Set width with some padding, max 50 chars
            column.width = min(max_length + 2, 50)

        # Also auto-fit the question title width
        worksheet.column_dimensions['A'].width = max(len(f"Query: {question}") + 2, worksheet.column_dimensions['A'].width)
        # Add metadata sheet
        metadata_rows = [
            ['Question', question],
            ['Generated Date', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Number of Rows', len(df)],
            ['Columns', ', '.join(df.columns)]
        ]
        
        if sql_query:
            metadata_rows.append(['SQL Query', sql_query])
        
        metadata_df = pd.DataFrame(metadata_rows, columns=['Field', 'Value'])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        metadata_sheet = writer.sheets['Metadata']
        metadata_sheet.column_dimensions['B'].width = 100
        
        if sql_query:
            sql_df = pd.DataFrame({'SQL_Query': [sql_query]})
            sql_df.to_excel(writer, sheet_name='SQL_Query', index=False)
            sql_sheet = writer.sheets['SQL_Query']
            sql_sheet.column_dimensions['A'].width = 100
            
        if chart_base64:
            try:
                from openpyxl.drawing.image import Image as OpenpyxlImage
                chart_data = base64.b64decode(chart_base64)
                chart_buffer = io.BytesIO(chart_data)
                img = OpenpyxlImage(chart_buffer)
                img.width = 600
                img.height = 400
                worksheet.add_image(img, f'A{len(df_raw) + 5}')
            except Exception as e:
                print(f"Could not embed chart in Excel: {e}")
    
    buffer.seek(0)
    return buffer

def export_all_results_to_excel(chat_history):
    """Export all query results to a multi-sheet Excel file with charts"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        from openpyxl.styles import Font
        from openpyxl.drawing.image import Image as OpenpyxlImage
        
        summary_data = []
        sheet_counter = 1
        
        for i, chat in enumerate(chat_history):
            if chat['result']['success'] and chat['result']['data']:
                sheet_name = f"Query_{sheet_counter}"
                
                summary_data.append({
                    'Query_Number': sheet_counter,
                    'Question': chat['question'],
                    'Sheet_Name': sheet_name,
                    'Rows_Returned': len(chat['result']['data']),
                    'Timestamp': chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'SQL_Query': chat['result'].get('sql', 'N/A')
                })
                
                if isinstance(chat['result']['data'][0], (list, tuple)):
                    df_result = pd.DataFrame(
                        chat['result']['data'], 
                        columns=chat['result'].get('columns', [f'Col_{j}' for j in range(len(chat['result']['data'][0]))])
                    )
                    
                    df_raw = df_result.copy()
                    for col in df_raw.columns:
                        if df_raw[col].dtype == 'object':
                            try:
                                numeric_series = pd.to_numeric(df_raw[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
                                df_raw[col] = numeric_series.fillna(df_raw[col])
                            except:
                                pass
                    
                    df_raw.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                    
                    worksheet = writer.sheets[sheet_name]
                    worksheet['A1'] = f"Query {sheet_counter}: {chat['question']}"
                    worksheet['A1'].font = Font(bold=True, size=14)
                    
                    for col_idx, col in enumerate(df_raw.columns, 1):
                        col_letter = chr(64 + col_idx)
                        
                        if any(keyword in col.lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit', 'value', 'spent']):
                            for row in range(3, len(df_raw) + 3):
                                cell = worksheet[f'{col_letter}{row}']
                                cell.number_format = '$#,##0.00'
                        elif df_raw[col].dtype in ['int64', 'float64']:
                            for row in range(3, len(df_raw) + 3):
                                cell = worksheet[f'{col_letter}{row}']
                                cell.number_format = '#,##0'

                    # Auto-fit all column widths
                    for col_idx, col in enumerate(df_raw.columns, 1):
                        col_letter = chr(64 + col_idx)
                        column = worksheet.column_dimensions[col_letter]
                        
                        # Calculate max width needed
                    
                        max_length = 0
                        for row in range(1, len(df_raw) + 3):  # Include header and title
                            cell_value = str(worksheet[f'{col_letter}{row}'].value or '')
                            max_length = max(max_length, len(cell_value))
                        
                    
                        # Set width with some padding, max 50 chars
                        column.width = min(max_length + 2, 50)

                    # Also auto-fit the question title width
                    worksheet.column_dimensions['A'].width = max(len(f"Query {sheet_counter}: {chat['question']}") + 2, worksheet.column_dimensions['A'].width)
                    
                    # Add chart if available
                    if chat.get('chart_base64'):
                        try:
                            chart_data = base64.b64decode(chat['chart_base64'])
                            chart_buffer = io.BytesIO(chart_data)
                            img = OpenpyxlImage(chart_buffer)
                            img.width = 600
                            img.height = 400
                            worksheet.add_image(img, f'A{len(df_raw) + 5}')
                        except Exception as e:
                            print(f"Could not embed chart in Excel for query {sheet_counter}: {e}")
                    
                    sheet_counter += 1
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            summary_sheet = writer.sheets['Summary']
            summary_sheet.column_dimensions['F'].width = 80
            summary_sheet.column_dimensions['B'].width = 50
    
    buffer.seek(0)
    return buffer

def export_error_log(error_log):
    """Export error log as a formatted text file for support tickets"""
    buffer = io.StringIO()
    
    buffer.write("=== ERROR REPORT FOR DEVELOPMENT TEAM ===\n")
    buffer.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buffer.write(f"Total Errors: {len(error_log)}\n\n")
    
    for i, error in enumerate(error_log, 1):
        buffer.write(f"--- ERROR {i} ---\n")
        buffer.write(f"Timestamp: {error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        buffer.write(f"User Question: {error['question']}\n")
        buffer.write(f"Error Message: {error['error']}\n")
        buffer.write(f"SQL Attempted: {error['sql_attempted']}\n")
        buffer.write("\n" + "="*50 + "\n\n")
    
    return buffer.getvalue().encode('utf-8')

# Page config
st.set_page_config(
    page_title="AI BI Analyst",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'sheets_connector' not in st.session_state:
    st.session_state.sheets_connector = None
if 'analytics_engine' not in st.session_state:
    st.session_state.analytics_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'error_log' not in st.session_state:
    st.session_state.error_log = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'query_log' not in st.session_state:
    st.session_state.query_log = []
if 'selected_faq' not in st.session_state:
    st.session_state.selected_faq = None

def main():
    st.title("🍺 AI BI Analyst for Google Sheets")
    st.markdown("Ask questions about your business data in natural language!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input with session state persistence
        # openai_api_key = st.text_input(
        #     "OpenAI API Key:",
        #     value=st.session_state.openai_api_key,
        #     type="password",
        #     placeholder="sk-...",
        #     help="Enter your OpenAI API key for smart SQL generation",
        #     key="api_key_input"
        # )
        
        # Update session state when key changes
        if openai_api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            # Reinitialize analytics engine with new key if already connected
            if st.session_state.sheets_connector is not None:
                analytics = AnalyticsEngine(openai_api_key if openai_api_key else None)
                analytics.load_data(st.session_state.sheets_connector.get_dataframes(), 
                                  st.session_state.sheets_connector.cleaning_metadata)
                st.session_state.analytics_engine = analytics
        
        # Show API key status
        if st.session_state.openai_api_key:
            st.success("API Key: Active")
        else:
            st.warning("API Key: Not Set")
        
        # Google Sheets URL input
        sheet_url = st.text_input(
            "Google Sheets URL:",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="Paste your Google Sheets URL here"
        )
        
        # Connect button
        if st.button("Connect to Sheets", type="primary"):
            if sheet_url:
                try:
                    with st.spinner("Connecting to Google Sheets..."):
                        # Initialize sheets connector
                        connector = SheetsConnector(temp_creds_path)
                        success = connector.connect(sheet_url)
                        
                        if success:
                            st.session_state.sheets_connector = connector
                            
                            # Initialize analytics engine with persisted OpenAI key
                            analytics = AnalyticsEngine(st.session_state.openai_api_key if st.session_state.openai_api_key else None)
                            analytics.load_data(connector.get_dataframes(), connector.cleaning_metadata)
                            st.session_state.analytics_engine = analytics
                            
                            # Show AI status
                            if st.session_state.openai_api_key:
                                st.success("Connected with AI-powered queries!")
                            else:
                                st.success("Connected with pattern-based queries!")
                                st.info("Add OpenAI API key above for smarter query generation")
                            
                        else:
                            st.error("Failed to connect to sheets")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a Google Sheets URL")
        
        # Always show sheet info if connected (persistence fix)
        if st.session_state.sheets_connector is not None:
            st.markdown("---")
            st.subheader("Sheet Information")
            sheet_info = st.session_state.sheets_connector.get_sheet_info()
            st.write(f"**Title:** {sheet_info['title']}")
            st.write(f"**Worksheets:** {', '.join(sheet_info['worksheets'])}")
            
            # Show connection status
            if st.session_state.analytics_engine is not None:
                if st.session_state.analytics_engine.ai_generator is not None:
                    st.success("🤖 AI-Powered Queries Active")
                else:
                    st.info("🔧 Pattern-Based Queries Active")
                    st.caption("Add OpenAI API key above for smarter queries")
            
            # Show schema information
            if 'cleaning_summary' in sheet_info:
                with st.expander("📋 Data Schema & Cleaning Info"):
                    cleaning_summary = sheet_info['cleaning_summary']
                    st.write(f"**Total rows processed:** {cleaning_summary.get('total_rows', 0)}")
                    st.write("**Tables and Column Mappings:**")
                    
                    for table_name, table_info in cleaning_summary.get('tables', {}).items():
                        st.write(f"\n**{table_name}** ({table_info['row_count']} rows)")
                        
                        # Show column mappings if any changes were made
                        col_mapping = table_info['column_mapping']
                        has_changes = any(orig != clean for orig, clean in col_mapping.items())
                        
                        if has_changes:
                            st.write("Column name changes:")
                            mapping_df = pd.DataFrame([
                                {'Original Name': orig, 'Cleaned Name': clean}
                                for orig, clean in col_mapping.items()
                                if orig != clean
                            ])
                            st.dataframe(mapping_df, use_container_width=True)
                        
                        # Show type conversions if any
                        type_conversions = table_info.get('type_conversions', {})
                        if type_conversions:
                            st.write("Data type conversions:")
                            for col, conversion in type_conversions.items():
                                st.write(f"  - `{col}`: {conversion}")
            
            # Show data preview
            for worksheet_name in sheet_info['worksheets']:
                with st.expander(f"Preview: {worksheet_name}"):
                    df = st.session_state.sheets_connector.dataframes[worksheet_name]
                    st.write(f"Shape: {df.shape}")
                    
                    # Show column info
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': [str(dtype) for dtype in df.dtypes]
                    })
                    st.write("**Column Information:**")
                    st.dataframe(col_info, use_container_width=True)
                    
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
    
    # Main chat interface
    if st.session_state.analytics_engine is not None:
        st.markdown("---")
        st.subheader("💬 Ask Your BI Questions")
        
        # FAQ with clickable questions
        with st.expander("💡 FAQ - Frequently Asked Questions"):
            st.markdown("**Click any question below to use it:**")
            
            faq_questions = [
                "What are our top 5 selling cultivars",
                "Who are our best customers by revenue?",
                "What's our monthly revenue trend?",
                "What was the revenue generated in Q3 2024 month over month by salesperson?",
                "Show me customers who spent more than $25,000",
                "What are the top 5 price per clones by cultivar",
                "Plot monthly revenue for 2025",
                "Chart the top 10 customers by revenue"
            ]
            
            # Create clickable buttons for each FAQ
            for i, question in enumerate(faq_questions):
                if st.button(f"📌 {question}", key=f"faq_{i}"):
                    st.session_state.selected_faq = question
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            **📊 Chart Generation Tips:**
            - Use keywords like "plot", "chart", "visualize" to ensure charts are created
            - Time-series queries automatically generate charts (monthly/yearly trends)
            - Top N queries (top 5, best 10) will create bar charts
            """)
        
        # Chat input with Enter key support via form
        with st.form(key="question_form", clear_on_submit=False):
            # Check if FAQ was selected
            default_value = ""
            if st.session_state.selected_faq:
                default_value = st.session_state.selected_faq
            
            user_question = st.text_input(
                "Ask a question about your data:",
                value=default_value,
                placeholder="e.g., What are our top selling products this year?",
                key="user_input_form"
            )
            
            # Single submit button for form (Enter key support)
            submitted = st.form_submit_button("🔍 Analyze", type="primary")
        
        # Process question when form submitted
        if submitted and user_question:
            # Reset selected_faq after processing
            st.session_state.selected_faq = None
            
            # Check if this exact question was just processed (prevent duplicates)
            if (not st.session_state.chat_history or 
                st.session_state.chat_history[-1]['question'] != user_question):
                
                try:
                    with st.spinner("Analyzing your data..."):
                        # Track query timing
                        start_time = time.time()
                        
                        # Get answer from analytics engine
                        result = st.session_state.analytics_engine.answer_question(user_question)
                        
                        # Calculate response time
                        response_time = time.time() - start_time
                        
                        # Display result
                        st.markdown("### Analysis Result")
                        st.markdown(f"**Question:** {user_question}")
                        st.markdown("---")
                        
                        if result['success']:
                            # Show Answer Methodology (explanation of how the query works)
                            if st.session_state.analytics_engine.ai_generator:
                                with st.expander("Answer Methodology"):
                                    explanation = st.session_state.analytics_engine.ai_generator.explain_sql_query(
                                        result['sql'], user_question
                                    )
                                    st.write(explanation)
                            
                            # Show results
                            if result['data']:
                                # Convert to DataFrame for better display
                                if isinstance(result['data'][0], (list, tuple)):
                                    df_result = pd.DataFrame(
                                        result['data'], 
                                        columns=result.get('columns', [f'Col_{i}' for i in range(len(result['data'][0]))])
                                    )
                                    
                                    # Post-process date formatting
                                    df_result = format_dates_in_dataframe(df_result)
                                    df_result = sort_month_data(df_result)
                                    
                                    # Create formatted version for display
                                    df_display = format_numbers_in_dataframe(df_result)
                                    
                                    # Check if chart should be created
                                    chart_applicable = should_create_chart(df_result, user_question, result.get('sql'))
                                    chart_base64 = None
                                    
                                    # Generate chart if applicable (for both display and Excel export)
                                    if chart_applicable:
                                        try:
                                            chart_base64 = create_chart_and_save(df_display, user_question)
                                        except Exception as e:
                                            st.warning(f"Could not create chart: {e}")
                                            chart_applicable = False
                                    
                                    # Create tabs for Data and Chart (Data first)
                                    if chart_applicable and chart_base64:
                                        tab1, tab2 = st.tabs(["📋 Data", "📊 Chart"])
                                        
                                        with tab1:
                                            st.dataframe(df_display, use_container_width=True)
                                        
                                        with tab2:
                                            st.image(f"data:image/png;base64,{chart_base64}")
                                    else:
                                        # If no chart, just show the data without tabs
                                        tab1, tab2 = st.tabs(["📋 Data", "📊 Chart"])
                                        
                                        with tab1:
                                            st.dataframe(df_display, use_container_width=True)
                                        
                                        with tab2:
                                            st.info("No visualization available for this query")
                                    
                                    # Show SQL query used (moved to bottom)
                                    with st.expander("SQL Query Used"):
                                        st.code(result['sql'], language='sql')
                                    
                                    # Add to chat history with full result data including chart
                                    chat_entry = {
                                        'question': user_question,
                                        'result': result,
                                        'timestamp': pd.Timestamp.now(),
                                        'chart_base64': chart_base64,
                                        'formatted_df': df_display
                                    }
                                    st.session_state.chat_history.append(chat_entry)
                                    
                                    # Add to query log for analytics
                                    log_entry = {
                                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'question': user_question,
                                        'status': 'success',
                                        'sql_query': result.get('sql', ''),
                                        'results': result['data'],
                                        'columns': result.get('columns', []),
                                        'source_columns': extract_source_columns(result.get('sql', '')),
                                        'response_time_seconds': round(response_time, 2),
                                        'error': None
                                    }
                                    st.session_state.query_log.append(log_entry)
                                    
                                    # Individual export removed - use bulk export instead
                                    
                                else:
                                    st.write(result['data'])
                                    # Add to chat history for non-tabular results too
                                    chat_entry = {
                                        'question': user_question,
                                        'result': result,
                                        'timestamp': pd.Timestamp.now(),
                                        'chart_base64': None,
                                        'formatted_df': None
                                    }
                                    st.session_state.chat_history.append(chat_entry)
                                    
                                    # Add to query log
                                    log_entry = {
                                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'question': user_question,
                                        'status': 'success',
                                        'sql_query': result.get('sql', ''),
                                        'results': result['data'],
                                        'columns': [],
                                        'source_columns': extract_source_columns(result.get('sql', '')),
                                        'response_time_seconds': round(response_time, 2),
                                        'error': None
                                    }
                                    st.session_state.query_log.append(log_entry)
                            else:
                                st.info("Query executed successfully but returned no data.")
                                # Add to chat history for no-data results
                                chat_entry = {
                                    'question': user_question,
                                    'result': result,
                                    'timestamp': pd.Timestamp.now(),
                                    'chart_base64': None,
                                    'formatted_df': None
                                }
                                st.session_state.chat_history.append(chat_entry)
                                
                                # Add to query log
                                log_entry = {
                                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'question': user_question,
                                    'status': 'success',
                                    'sql_query': result.get('sql', ''),
                                    'results': [],
                                    'columns': [],
                                    'source_columns': extract_source_columns(result.get('sql', '')),
                                    'response_time_seconds': round(response_time, 2),
                                    'error': None
                                }
                                st.session_state.query_log.append(log_entry)
                        else:
                            # Log error and show user-friendly message
                            error_entry = {
                                'timestamp': pd.Timestamp.now(),
                                'question': user_question,
                                'error': result['error'],
                                'sql_attempted': result.get('sql', 'No SQL generated')
                            }
                            st.session_state.error_log.append(error_entry)
                            
                            # Add to query log as failure
                            log_entry = {
                                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'question': user_question,
                                'status': 'failure',
                                'sql_query': result.get('sql', 'No SQL generated'),
                                'results': [],
                                'columns': [],
                                'source_columns': extract_source_columns(result.get('sql', '')),
                                'response_time_seconds': round(response_time, 2),
                                'error': result['error']
                            }
                            st.session_state.query_log.append(log_entry)
                            
                            # User-friendly error message
                            st.error("I couldn't process that question properly.")
                            st.markdown("""
                            **Try rephrasing your question differently:**
                            - Be more specific about what you want to analyze
                            - Use simpler language 
                            - Check column names in the data preview above
                            
                            **Examples of clear questions:**
                            - "Show me total revenue by month"
                            - "Which customers spent the most money?"
                            - "What are our top selling products?"
                            """)
                            
                            # Show technical details in expander
                            with st.expander("Technical Details (for developers)"):
                                st.code(f"Error: {result['error']}", language="text")
                                if result.get('sql'):
                                    st.code(result['sql'], language='sql')
                            
                except Exception as e:
                    # Log unexpected errors
                    error_entry = {
                        'timestamp': pd.Timestamp.now(),
                        'question': user_question,
                        'error': str(e),
                        'sql_attempted': 'System error'
                    }
                    st.session_state.error_log.append(error_entry)
                    
                    st.error("Something went wrong! Please try rephrasing your question.")
                    
                    with st.expander("Technical Details (for developers)"):
                        st.code(f"System Error: {str(e)}", language="text")
        elif submitted and not user_question:
            st.warning("Please enter a question first.")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            
            # Header with export options
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📝 Questions Asked in this Session")
            
            with col2:
                # Direct download without two-step process
                all_results_buffer = export_all_results_to_excel(st.session_state.chat_history)
                st.download_button(
                    label="📥 Export All Results",
                    data=all_results_buffer,
                    file_name=f"all_query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="export_all_btn"
                )
            
            # Show recent queries with cached results
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.expander(f"Q: {chat['question'][:60]}..."):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Asked:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if chat['result']['success'] and chat['result']['data']:
                        st.write(f"**Results:** {len(chat['result']['data'])} rows")
                        
                        # Show cached formatted results
                        if chat.get('formatted_df') is not None:
                            st.dataframe(chat['formatted_df'], use_container_width=True)
                        elif isinstance(chat['result']['data'][0], (list, tuple)):
                            df_cached = pd.DataFrame(
                                chat['result']['data'], 
                                columns=chat['result'].get('columns', [f'Col_{j}' for j in range(len(chat['result']['data'][0]))])
                            )
                            
                            # Apply formatting to cached results
                            df_cached = format_dates_in_dataframe(df_cached)
                            df_cached = sort_month_data(df_cached)
                            df_cached = format_numbers_in_dataframe(df_cached)
                            
                            st.dataframe(df_cached, use_container_width=True)
                        
                        # Show cached chart if available
                        if chat.get('chart_base64'):
                            st.markdown("**Chart:**")
                            st.image(f"data:image/png;base64,{chat['chart_base64']}")
                        
                        # Individual export removed - use bulk export instead
                        
                        # Show SQL used
                        with st.expander("🔍 SQL Query"):
                            st.code(chat['result']['sql'], language='sql')
                    else:
                        st.write("❌ Query failed")
                        if chat['result'].get('error'):
                            st.code(chat['result']['error'], language='text')
            
            # Query Analytics Log Section
            if st.session_state.query_log:
                st.markdown("---")
                st.subheader("📊 Query Analytics Log")
                
                # Export options for query log
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Total Queries:** {len(st.session_state.query_log)} | "
                              f"**Success Rate:** {sum(1 for q in st.session_state.query_log if q['status'] == 'success') / len(st.session_state.query_log) * 100:.1f}%")
                
                with col2:
                    # Export query log as JSON
                    log_json = json.dumps(st.session_state.query_log, indent=2)
                    st.download_button(
                        label="💾 Export Log (JSON)",
                        data=log_json,
                        file_name=f"query_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="export_log_btn"
                    )
                
                # Display all log entries from this session
                st.markdown(f"**All Query Analytics (This Session - {len(st.session_state.query_log)} queries):**")
                
                # Show all logs, reversed to show newest first
                all_logs = st.session_state.query_log[::-1]  # All logs, reversed
                
                for i, log in enumerate(all_logs):
                    status_icon = "✅" if log['status'] == 'success' else "❌"
                    with st.expander(f"{status_icon} {log['question'][:60]}... ({log['response_time_seconds']}s)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Timestamp:** {log['timestamp']}")
                            st.write(f"**Status:** {log['status']}")
                            st.write(f"**Response Time:** {log['response_time_seconds']} seconds")
                        
                        with col2:
                            if log['results']:
                                st.write(f"**Rows Returned:** {len(log['results'])}")
                            else:
                                st.write("**Rows Returned:** 0")
                        
                        if log['sql_query']:
                            st.markdown("**SQL Query:**")
                            st.code(log['sql_query'], language='sql')
                        
                        if log['error']:
                            st.markdown("**Error:**")
                            st.error(log['error'])
                        
                        # Show first few rows of results if available
                        if log['results'] and len(log['results']) > 0:
                            st.markdown("**Sample Results (First 5 rows):**")
                            sample_df = pd.DataFrame(
                                log['results'][:5],
                                columns=log['columns'] if log['columns'] else None
                            )
                            st.dataframe(sample_df, use_container_width=True)
    
    else:
        # Instructions when not connected
        st.markdown("---")
        st.info("👈 Please connect to your Google Sheets using the sidebar to get started!")
        
        # Demo section
        st.markdown("### 🎯 What this tool does:")
        st.markdown("""
        - **Connect** to your Google Sheets (Sales, Customers, Products, etc.)
        - **Ask questions** in plain English about your business
        - **Get instant insights** with SQL-powered analytics
        - **See visualizations** of your key metrics
        
        Perfect for SMBs running their business on Google Sheets! 📊
        """)

if __name__ == "__main__":
    main()