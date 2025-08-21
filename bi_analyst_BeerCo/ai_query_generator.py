import openai
from typing import Optional, Dict, Any
import os
import re

class AIQueryGenerator:
    """Generates SQL queries from natural language using OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
    def generate_sql(self, question: str, schema_info: str) -> Dict[str, Any]:
        """Generate SQL query from natural language question"""
        try:
            # Create the prompt for SQL generation
            prompt = self._create_sql_prompt(question, schema_info)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a SQL expert specializing in business intelligence queries. Generate only the SQL query without any explanations or markdown formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=500
            )
            
            # Extract SQL from response
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            return {
                'success': True,
                'sql': sql_query,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'sql': None,
                'error': str(e)
            }
    
    def _create_sql_prompt(self, question: str, schema_info: str) -> str:
        """Create a detailed prompt for SQL generation"""
        prompt = f"""
Given the following database schema, generate a SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_info}

IMPORTANT GUIDELINES:
1. Use only the tables and columns that exist in the schema above
2. For date operations, cast string dates to DATE type: CAST(sale_date AS DATE)
3. For month/year formatting, use this format: strftime('%B %Y', CAST(sale_date AS DATE)) to get "March 2023" format
4. For date ranges, always add ORDER BY date ASC for chronological order
5. Use proper JOINs when combining tables
6. ALWAYS use table aliases and fully qualify column names in JOINs (e.g., s.product_id = p.product_id)
7. Include appropriate GROUP BY, ORDER BY, and LIMIT clauses
8. For revenue/sales analysis, use SUM(total_amount)
9. For customer analysis, join sales with customers table
10. For product analysis, join sales with products table
11. For profit/margin calculations, use (price - cost) or (revenue - cost)
12. Use meaningful table aliases: s for sales, c for customers, p for products
13. Return only the SQL query, no explanations

EXAMPLE OF PROPER DATE FORMATTING AND ORDERING:
SELECT strftime('%B %Y', CAST(sale_date AS DATE)) as month_year, SUM(total_amount) as revenue
FROM salestransactions
WHERE CAST(sale_date AS DATE) BETWEEN '2023-03-01' AND '2023-08-31'
GROUP BY strftime('%Y-%m', CAST(sale_date AS DATE)), strftime('%B %Y', CAST(sale_date AS DATE))
ORDER BY strftime('%Y-%m', CAST(sale_date AS DATE)) ASC

USER QUESTION: {question}

SQL QUERY:"""
        
        return prompt
    
    def _clean_sql_query(self, sql: str) -> str:
        """Clean and validate the generated SQL query"""
        # Remove markdown code blocks if present
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove extra whitespace
        sql = sql.strip()
        
        # Ensure query ends with semicolon (optional)
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def validate_sql_safety(self, sql: str) -> bool:
        """Basic validation to ensure SQL is safe (read-only)"""
        sql_lower = sql.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'insert', 'update', 'alter', 
            'create', 'truncate', 'exec', 'execute'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        # Must start with SELECT
        if not sql_lower.startswith('select'):
            return False
            
        return True