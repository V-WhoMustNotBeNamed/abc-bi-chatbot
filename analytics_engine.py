import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional
import re
from ai_query_generator import AIQueryGenerator

class AnalyticsEngine:
    """Handles DuckDB analytics and query generation"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.conn = None
        self.tables = {}
        self.table_schemas = {}
        self.column_mappings = {}  # Store original to cleaned column mappings
        self.ai_generator = AIQueryGenerator(openai_api_key) if openai_api_key else None
        
    def load_data(self, dataframes: Dict[str, pd.DataFrame], cleaning_metadata: Optional[Dict[str, Dict]] = None) -> bool:
        """Load DataFrames into DuckDB tables with schema information"""
        try:
            # Create new in-memory DuckDB connection
            self.conn = duckdb.connect(':memory:')
            self.tables = {}
            self.table_schemas = {}
            self.column_mappings = {}
            
            for name, df in dataframes.items():
                if df is not None and not df.empty:
                    # Clean table name (remove spaces, special chars)
                    table_name = self._clean_table_name(name)
                    
                    # Create table from DataFrame
                    self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                    
                    # Store table info
                    self.tables[table_name] = name  # mapping from clean name to original
                    
                    # Get column mapping from cleaning metadata if available
                    if cleaning_metadata and name in cleaning_metadata:
                        metadata = cleaning_metadata[name]
                        column_mapping = metadata.get('column_mapping', {})
                        type_conversions = metadata.get('type_conversions', {})
                    else:
                        column_mapping = {col: col for col in df.columns}
                        type_conversions = {}
                    
                    self.table_schemas[table_name] = {
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'sample_data': df.head(3).to_dict('records'),
                        'column_mapping': column_mapping,
                        'type_conversions': type_conversions
                    }
                    
                    self.column_mappings[table_name] = column_mapping
                    
            return len(self.tables) > 0
            
        except Exception as e:
            print(f"Error loading data into DuckDB: {e}")
            return False
    
    def _clean_table_name(self, name: str) -> str:
        """Clean table name for SQL compatibility"""
        # Convert to lowercase and replace spaces/special chars with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading/trailing underscores
        return clean_name.strip('_')
    
    def get_schema_info(self) -> str:
        """Get schema information for AI query generation"""
        schema_info = "Available tables and their schemas:\n\n"
        
        for table_name, original_name in self.tables.items():
            schema = self.table_schemas[table_name]
            schema_info += f"Table: {table_name} (original sheet: {original_name})\n"
            schema_info += "Columns with original names:\n"
            
            # Get column mapping
            column_mapping = schema.get('column_mapping', {})
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            
            for col in schema['columns']:
                dtype = schema['dtypes'][col]
                original_col = reverse_mapping.get(col, col)
                if original_col != col:
                    schema_info += f"  - {col} (originally: '{original_col}', type: {dtype})\n"
                else:
                    schema_info += f"  - {col} (type: {dtype})\n"
            
            # Show type conversions if any
            type_conversions = schema.get('type_conversions', {})
            if type_conversions:
                schema_info += "Type conversions applied:\n"
                for col, conversion in type_conversions.items():
                    schema_info += f"  - {col}: {conversion}\n"
            
            schema_info += "Sample data:\n"
            for i, row in enumerate(schema['sample_data']):
                if i < 2:  # Show only 2 sample rows
                    schema_info += f"  {row}\n"
            schema_info += "\n"
            
        return schema_info
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Convert natural language question to SQL and execute"""
        try:
            # Generate SQL query from the question
            sql_query = self._generate_sql_query(question)
            
            if not sql_query:
                return {
                    'success': False,
                    'error': 'Could not generate SQL query from question',
                    'sql': None,
                    'data': None
                }
            
            # Execute the query
            result = self.conn.execute(sql_query).fetchall()
            
            # Get column names
            columns = [desc[0] for desc in self.conn.description]
            
            return {
                'success': True,
                'sql': sql_query,
                'data': result,
                'columns': columns,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'sql': sql_query if 'sql_query' in locals() else None,
                'data': None
            }
    
    def _generate_sql_query(self, question: str) -> Optional[str]:
        """Generate SQL query from natural language question"""
        
        # Try AI generation first if available
        if self.ai_generator:
            schema_info = self.get_schema_info()
            result = self.ai_generator.generate_sql(question, schema_info)
            
            if result['success']:
                sql = result['sql']
                
                # Validate SQL safety
                if self.ai_generator.validate_sql_safety(sql):
                    return sql.rstrip(';')  # Remove trailing semicolon for DuckDB
                else:
                    print("Generated SQL failed safety validation")
            else:
                print(f"AI SQL generation failed: {result['error']}")
        
        # Fallback to pattern matching if AI fails or unavailable
        question_lower = question.lower()
        
        # Top selling products
        if any(phrase in question_lower for phrase in ['top selling', 'best selling', 'top products']):
            if 'sales' in self.tables or 'salestransactions' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"""
                SELECT 
                    product_name,
                    SUM(total_amount) as revenue,
                    SUM(quantity) as units_sold
                FROM {sales_table}
                GROUP BY product_name
                ORDER BY revenue DESC
                LIMIT 10
                """
        
        # Best customers
        elif any(phrase in question_lower for phrase in ['best customers', 'top customers', 'customer revenue']):
            if 'sales' in self.tables or 'salestransactions' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"""
                SELECT 
                    customer_name,
                    SUM(total_amount) as total_spent,
                    COUNT(sale_id) as num_purchases
                FROM {sales_table}
                GROUP BY customer_name
                ORDER BY total_spent DESC
                LIMIT 10
                """
        
        # Revenue by beer type
        elif any(phrase in question_lower for phrase in ['beer type', 'beer types', 'category performance']):
            if ('sales' in self.tables or 'salestransactions' in self.tables) and 'products' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"""
                SELECT 
                    p.beer_type,
                    SUM(s.total_amount) as revenue,
                    COUNT(s.sale_id) as transactions
                FROM {sales_table} s
                JOIN products p ON s.product_id = p.product_id
                GROUP BY p.beer_type
                ORDER BY revenue DESC
                """
        
        # Monthly revenue trend
        elif any(phrase in question_lower for phrase in ['monthly', 'revenue trend', 'sales trend']):
            if 'sales' in self.tables or 'salestransactions' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"""
                SELECT 
                    strftime('%Y-%m', CAST(sale_date AS DATE)) as month,
                    SUM(total_amount) as revenue,
                    COUNT(sale_id) as transactions
                FROM {sales_table}
                GROUP BY month
                ORDER BY month
                """
        
        # Profit analysis
        elif any(phrase in question_lower for phrase in ['profit', 'margin', 'profitability']):
            if ('sales' in self.tables or 'salestransactions' in self.tables) and 'products' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"""
                SELECT 
                    SUM(s.total_amount) as total_revenue,
                    SUM(s.quantity * p.cost) as total_cost,
                    SUM(s.total_amount) - SUM(s.quantity * p.cost) as gross_profit,
                    ROUND((SUM(s.total_amount) - SUM(s.quantity * p.cost)) / SUM(s.total_amount) * 100, 2) as profit_margin_pct
                FROM {sales_table} s
                JOIN products p ON s.product_id = p.product_id
                """
        
        # Inventory levels
        elif any(phrase in question_lower for phrase in ['inventory', 'stock', 'restock']):
            if 'products' in self.tables:
                return """
                SELECT 
                    product_name,
                    stock_quantity,
                    in_stock,
                    price
                FROM products
                WHERE stock_quantity < 100 OR in_stock = 'FALSE'
                ORDER BY stock_quantity ASC
                """
        
        # Customer count
        elif any(phrase in question_lower for phrase in ['how many customers', 'customer count', 'total customers']):
            if 'customers' in self.tables:
                return "SELECT COUNT(*) as total_customers FROM customers"
        
        # Product count
        elif any(phrase in question_lower for phrase in ['how many products', 'product count', 'total products']):
            if 'products' in self.tables:
                return "SELECT COUNT(*) as total_products FROM products"
        
        # Total revenue
        elif any(phrase in question_lower for phrase in ['total revenue', 'total sales']):
            if 'sales' in self.tables or 'salestransactions' in self.tables:
                sales_table = 'sales' if 'sales' in self.tables else 'salestransactions'
                return f"SELECT SUM(total_amount) as total_revenue FROM {sales_table}"
        
        # Default: return None if no pattern matches
        return None
    
    def execute_custom_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute a custom SQL query"""
        try:
            result = self.conn.execute(sql_query).fetchall()
            columns = [desc[0] for desc in self.conn.description]
            
            return {
                'success': True,
                'sql': sql_query,
                'data': result,
                'columns': columns,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'sql': sql_query,
                'data': None
            }