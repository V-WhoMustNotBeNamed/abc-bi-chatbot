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
                max_tokens=1000
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
    
    def explain_sql_query(self, sql_query: str, question: str) -> str:
        """Generate a natural language explanation of what the SQL query does"""
        try:
            # Create prompt for SQL explanation
            prompt = f"""
Explain the following SQL query in simple, non-technical language. 
Focus on what the query is doing to answer the user's question.

CRITICAL REQUIREMENTS - Your explanation MUST include ALL of the following:

1. **Date Column Specification**: If ANY date filtering or grouping is used, you MUST explicitly state which date column is being used (e.g., "using the date_received column" or "filtering by the sale_date column")

2. **Source Column Names**: Always mention the actual column names from the database, NOT aliases. For example:
   - Say "counting rows in the cultivar column" NOT "counting total_sold"
   - Say "summing the total_amount column" NOT "calculating revenue"
   - Say "grouping by the product_name column" NOT "grouping by product"

3. **Calculation Details**: For every aggregation (COUNT, SUM, AVG, etc.), specify:
   - The exact operation (counting, summing, averaging)
   - The exact column being operated on
   - What each calculation represents

4. **Multiple Date Columns**: If the database has multiple date columns and one was chosen, briefly explain which one (e.g., "using date_received rather than date_shipped")

5. **Be Explicit**: Never use vague terms like "the date field" or "a column" - always name the specific column

User's Question: {question}

SQL Query:
{sql_query}

Provide a clear, detailed explanation that explicitly names every column used:
"""
            
            # Call OpenAI API for explanation
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that explains SQL queries in simple, business-friendly language."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            # Fallback to a simple explanation if API fails
            return "This query analyzes your data to answer the question by filtering, grouping, and calculating the relevant information from your database tables."
    
    def _create_sql_prompt(self, question: str, schema_info: str) -> str:
        """Create a detailed prompt for SQL generation"""
        prompt = f"""
Given the following database schema, generate a SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_info}

IMPORTANT GUIDELINES:
1. Use only the tables and columns that exist in the schema above
2. CRITICAL: Column names have been standardized to snake_case. When you see "originally: 'Original Name'" in the schema, the user might refer to the original name, but you MUST use the standardized column name in your SQL query.
3. For example, if the schema shows: sales_contact (originally: 'Sales Contact'), use 'sales_contact' in the SQL even if the user mentions "Sales Contact"
4. Data types have been cleaned and standardized - the schema shows the actual data types after conversion
5. Numeric columns (amounts, quantities, prices) are already converted to float/numeric types - no casting needed
6. Date columns are already in datetime format - you can use them directly in date operations
7. For month/year formatting, use this format: strftime('%B %Y', sale_date) to get "March 2023" format
8. For date ranges, always add ORDER BY date ASC for chronological order
9. Use proper JOINs when combining tables
10. ALWAYS use table aliases and fully qualify column names in JOINs (e.g., s.product_id = p.product_id)
11. Include appropriate GROUP BY, ORDER BY, and LIMIT clauses
12. For revenue/sales analysis, use SUM(total_amount) or similar amount columns
13. For customer analysis, join sales with customers table
14. For product analysis, join sales with products table
15. For profit/margin calculations, use (price - cost) or (revenue - cost)
16. Use meaningful table aliases: s for sales, c for customers, p for products
17. Return only the SQL query, no explanations unless there's an error

CRITICAL BUSINESS LOGIC RULES:
18. **VOLUME means QUANTITY, not PRICE**: 
    - When user asks about "volume" or "sales volume", use quantity/qty/units columns
    - "Sales volume" = COUNT or SUM of quantity fields
    - "Sales value" or "revenue" = SUM of price/amount fields
    - Examples:
      * "What is the sales volume?" → Use COUNT(*) or SUM(quantity)
      * "What is the sales value?" → Use SUM(total_amount) or SUM(price)
      * "Top products by volume" → ORDER BY quantity or COUNT
      * "Top products by value" → ORDER BY amount or revenue
19. Always prioritize quantity fields for volume-related queries
20. Only use price/amount fields when explicitly asked about revenue, value, or money

COLUMN NAME MAPPING EXAMPLES:
- If user asks about "Sales Contact", use column: sales_contact
- If user asks about "Total Amount", use column: total_amount
- If user asks about "Product Name", use column: product_name

CRITICAL GROUP BY RULE:
21. **GROUP BY expressions MUST match SELECT expressions exactly**:
    - If SELECT has strftime('%B %Y', date), GROUP BY must also have strftime('%B %Y', date)
    - NEVER mix different date formats between SELECT and GROUP BY
    - Alternative: Use MIN/MAX wrapper in SELECT if GROUP BY uses different format

CORRECT DATE FORMATTING EXAMPLES:
Example 1 - Matching expressions:
SELECT strftime('%B %Y', sale_date) as month_year, SUM(total_amount) as revenue
FROM salestransactions
WHERE sale_date BETWEEN '2023-03-01' AND '2023-08-31'
GROUP BY strftime('%B %Y', sale_date)
ORDER BY MIN(sale_date) ASC

Example 2 - Using MIN wrapper:
SELECT MIN(strftime('%B %Y', sale_date)) as month_year, SUM(total_amount) as revenue
FROM salestransactions
WHERE sale_date BETWEEN '2023-03-01' AND '2023-08-31'
GROUP BY strftime('%Y-%m', sale_date)
ORDER BY strftime('%Y-%m', sale_date) ASC

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