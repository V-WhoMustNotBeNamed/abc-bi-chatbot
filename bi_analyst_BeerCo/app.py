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
            
            # Create seaborn bar plot - fix deprecation warning
            bars = sns.barplot(
                data=chart_df, 
                x=df.columns[0], 
                y=df.columns[1],
                hue=df.columns[0],
                palette="viridis",
                legend=False,
                ax=ax
            )
            
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
            
            # Add value labels on bars
            for i, bar in enumerate(bars.patches):
                height = bar.get_height()
                if any(keyword in df.columns[1].lower() for keyword in ['revenue', 'amount', 'price', 'cost', 'profit']):
                    label = f'${height:,.0f}'
                else:
                    label = f'{height:,.0f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       label, ha='center', va='bottom', fontsize=10)
            
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
    """Export all query results to a multi-sheet Excel file"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        from openpyxl.styles import Font
        
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
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            st.markdown("""
            **Try asking questions like:**
            - What are our top 5 selling products?
            - Who are our best customers by revenue?
            - What's our monthly revenue trend?
            - Which beer types perform best?
            - What's our profit margin?
            - Which products need restocking?
            - What was our revenue in March 2023?
            - Show me customers who spent more than $1000
            - Which products have the highest profit margin?
            
            *Just type your question in natural language below!*
            """)
        
        # Chat input with Enter key support via form
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., What are our top selling products this year?",
                key="user_input_form"
            )
            
            # Single submit button for form (Enter key support)
            submitted = st.form_submit_button("🔍 Analyze", type="primary")
        
        # Process question when form submitted
        if submitted and user_question:
            # Check if this exact question was just processed (prevent duplicates)
            if (not st.session_state.chat_history or 
                st.session_state.chat_history[-1]['question'] != user_question):
                
                try:
                    with st.spinner("Analyzing your data..."):
                        # Get answer from analytics engine
                        result = st.session_state.analytics_engine.answer_question(user_question)
                        
                        # Display result
                        st.markdown("### Analysis Result")
                        st.markdown(f"**Question:** {user_question}")
                        st.markdown("---")
                        
                        if result['success']:
                            # Show SQL query used
                            with st.expander("SQL Query Used"):
                                st.code(result['sql'], language='sql')
                            
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
                                    
                                    st.dataframe(df_display, use_container_width=True)
                                    
                                    # Create and show chart
                                    chart_base64 = None
                                    if len(df_result.columns) >= 2:
                                        try:
                                            if 'revenue' in df_result.columns[1].lower() or 'amount' in df_result.columns[1].lower() or 'quantity' in df_result.columns[1].lower():
                                                chart_base64 = create_chart_and_save(df_display, user_question)
                                                if chart_base64:
                                                    st.markdown("### Visualization")
                                                    st.image(f"data:image/png;base64,{chart_base64}")
                                        except Exception as e:
                                            st.warning(f"Could not create chart: {e}")
                                    
                                    # Add to chat history with full result data including chart
                                    chat_entry = {
                                        'question': user_question,
                                        'result': result,
                                        'timestamp': pd.Timestamp.now(),
                                        'chart_base64': chart_base64,
                                        'formatted_df': df_display
                                    }
                                    st.session_state.chat_history.append(chat_entry)
                                    
                                    # Add export button for this result
                                    if st.button("Export to Excel", key=f"export_{len(st.session_state.chat_history)}"):
                                        excel_buffer = export_to_excel(df_result, user_question, result['sql'], chart_base64)
                                        st.download_button(
                                            label="Download Excel File",
                                            data=excel_buffer,
                                            file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    
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
                        else:
                            # Log error and show user-friendly message
                            error_entry = {
                                'timestamp': pd.Timestamp.now(),
                                'question': user_question,
                                'error': result['error'],
                                'sql_attempted': result.get('sql', 'No SQL generated')
                            }
                            st.session_state.error_log.append(error_entry)
                            
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
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader("📝 Query History")
            
            with col2:
                if st.button("📥 Export All Results"):
                    all_results_buffer = export_all_results_to_excel(st.session_state.chat_history)
                    st.download_button(
                        label="💾 Download All Results",
                        data=all_results_buffer,
                        file_name=f"all_query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                if st.session_state.error_log and st.button("🐛 Download Error Log"):
                    error_log_buffer = export_error_log(st.session_state.error_log)
                    st.download_button(
                        label="📋 Get Error Report",
                        data=error_log_buffer,
                        file_name=f"error_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
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
                        
                        # Individual export for this query
                        if isinstance(chat['result']['data'][0], (list, tuple)):
                            df_export = pd.DataFrame(
                                chat['result']['data'], 
                                columns=chat['result'].get('columns', [f'Col_{j}' for j in range(len(chat['result']['data'][0]))])
                            )
                            excel_buffer = export_to_excel(df_export, chat['question'], chat['result']['sql'], chat.get('chart_base64'))
                            st.download_button(
                                label="📥 Export This Result",
                                data=excel_buffer,
                                file_name=f"query_{i}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"export_history_{i}"
                            )
                        
                        # Show SQL used
                        with st.expander("🔍 SQL Query"):
                            st.code(chat['result']['sql'], language='sql')
                    else:
                        st.write("❌ Query failed")
                        if chat['result'].get('error'):
                            st.code(chat['result']['error'], language='text')
    
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