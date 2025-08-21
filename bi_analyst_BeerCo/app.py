import streamlit as st
import pandas as pd
from sheets_connector import SheetsConnector
from analytics_engine import AnalyticsEngine

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

def main():
    st.title("🍺 AI BI Analyst for Google Sheets")
    st.markdown("Ask questions about your business data in natural language!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key for smart SQL generation"
        )
        
        # Google Sheets URL input
        sheet_url = st.text_input(
            "Google Sheets URL:",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="Paste your Google Sheets URL here"
        )
        
        # Connect button
        if st.button("🔗 Connect to Sheets", type="primary"):
            if sheet_url:
                try:
                    with st.spinner("Connecting to Google Sheets..."):
                        # Initialize sheets connector
                        connector = SheetsConnector('credentials.json')
                        success = connector.connect(sheet_url)
                        
                        if success:
                            st.session_state.sheets_connector = connector
                            
                            # Initialize analytics engine with OpenAI key
                            analytics = AnalyticsEngine(openai_api_key if openai_api_key else None)
                            analytics.load_data(connector.get_dataframes())
                            st.session_state.analytics_engine = analytics
                            
                            # Show AI status
                            if openai_api_key:
                                st.success("✅ Connected with AI-powered queries!")
                            else:
                                st.success("✅ Connected with pattern-based queries!")
                                st.info("💡 Add OpenAI API key for smarter query generation")
                            
                            # Show sheet info
                            st.subheader("📊 Sheet Information")
                            sheet_info = connector.get_sheet_info()
                            st.write(f"**Title:** {sheet_info['title']}")
                            st.write(f"**Worksheets:** {', '.join(sheet_info['worksheets'])}")
                            
                            # Show data preview
                            for worksheet_name in sheet_info['worksheets']:
                                with st.expander(f"Preview: {worksheet_name}"):
                                    df = connector.dataframes[worksheet_name]
                                    st.write(f"Shape: {df.shape}")
                                    st.dataframe(df.head(), use_container_width=True)
                        else:
                            st.error("❌ Failed to connect to sheets")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a Google Sheets URL")
    
    # Main chat interface
    if st.session_state.analytics_engine is not None:
        st.markdown("---")
        st.subheader("💬 Ask Your BI Questions")
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What are our top 5 selling products?",
                "Who are our best customers by revenue?",
                "What's our monthly revenue trend?",
                "Which beer types perform best?",
                "What's our profit margin?",
                "Which products need restocking?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                col = cols[i % 2]
                if col.button(question, key=f"sample_{i}"):
                    st.session_state.current_question = question
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What are our top selling products this year?",
            key="user_input"
        )
        
        # Process question
        if st.button("🔍 Analyze", type="primary") or user_question:
            question_to_process = user_question or st.session_state.get('current_question', '')
            
            if question_to_process:
                try:
                    with st.spinner("Analyzing your data..."):
                        # Get answer from analytics engine
                        result = st.session_state.analytics_engine.answer_question(question_to_process)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question_to_process,
                            'result': result
                        })
                        
                        # Display result
                        st.markdown("### 📊 Analysis Result")
                        
                        if result['success']:
                            # Show SQL query used
                            with st.expander("🔍 SQL Query Used"):
                                st.code(result['sql'], language='sql')
                            
                            # Show results
                            if result['data']:
                                # Convert to DataFrame for better display
                                if isinstance(result['data'][0], (list, tuple)):
                                    df_result = pd.DataFrame(
                                        result['data'], 
                                        columns=result.get('columns', [f'Col_{i}' for i in range(len(result['data'][0]))])
                                    )
                                    st.dataframe(df_result, use_container_width=True)
                                    
                                    # Try to create a simple visualization
                                    if len(df_result.columns) >= 2:
                                        try:
                                            # Simple bar chart for top results
                                            if 'revenue' in df_result.columns[1].lower() or 'amount' in df_result.columns[1].lower():
                                                st.bar_chart(df_result.set_index(df_result.columns[0])[df_result.columns[1]])
                                        except:
                                            pass  # Skip visualization if it fails
                                else:
                                    st.write(result['data'])
                            else:
                                st.info("Query executed successfully but returned no data.")
                        else:
                            st.error(f"❌ Error: {result['error']}")
                            
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📝 Query History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.write(f"**Question:** {chat['question']}")
                    if chat['result']['success'] and chat['result']['data']:
                        st.write(f"**Results:** {len(chat['result']['data'])} rows returned")
                        st.code(chat['result']['sql'], language='sql')
    
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