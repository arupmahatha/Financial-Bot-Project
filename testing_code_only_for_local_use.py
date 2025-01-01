# IGNORE THIS FILE, This is for my temporary working and testing on my local machine
# This is of no use to any deployement work

import os
from typing import Dict, List, Optional, TypedDict, Literal, Union, Annotated
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import sqlite3
import re
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.tools import Tool
from functools import lru_cache
import hashlib
import pickle
import uuid
from datetime import datetime

# Initialize memory for state management
memory = {}  # Using a simple dictionary for in-memory storage

# Part 2: Type Definitions and Base Classes
class QueryType(Enum):
    DIRECT_SQL = "direct_sql"  # For direct SQL queries
    ANALYSIS = "analysis"      # For complex analysis requiring multiple queries

@dataclass
class QueryClassification:
    type: QueryType
    explanation: str
    raw_response: str
    confidence: float = 1.0

class AnalysisState(TypedDict):
    user_query: str              # The original user question
    query_classification: Dict    # How the query should be processed
    decomposed_questions: List[str]  # Breaking complex queries into parts
    sql_results: Dict            # Results from SQL queries
    analysis: str                # Analysis of the results
    final_output: Dict          # Final formatted output
    processing_time: float       # Time taken to process
    agent_states: Dict          # State tracking for agents
    raw_responses: Dict         # Raw responses from agents
    messages: List[AnyMessage]  # Conversation history
    conversation_history: List[Dict]  # Add this line

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

@dataclass
class Config:
    db_path: str = "final_working_database.db"
    sqlite_path: str = "sqlite:///final_working_database.db"
    model_name: str = "claude-3-sonnet-20240229"
    api_key: str = ""  # Anthropic API key
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    cache_ttl: int = 86400  # Cache TTL in seconds (24 hours)

# Add new classes for metadata management
class ColumnDefinition:
    def __init__(self, description: str, hierarchy_level=None, distinct_values=None):
        self.description = description
        self.hierarchy_level = hierarchy_level
        self.distinct_values = distinct_values or []

class TableDefinition:
    def __init__(self, description, key_purposes, common_queries, relationships, columns=None):
        self.description = description
        self.key_purposes = key_purposes
        self.common_queries = common_queries
        self.relationships = relationships
        self.columns = columns or {}

class FinancialTableMetadata:
    def __init__(self):
        self.tables = {
            "final_income_sheet_new_seq": TableDefinition(
                description="Tracks revenue, expenses, and profitability over a period",
                key_purposes=[
                    "Monitor revenue and expenses",
                    "Track profitability",
                    "Compare actual vs budget"
                ],
                common_queries=[
                    "Revenue by type",
                    "Expense analysis",
                    "Profit margins",
                    "Common questions about the analysis"
                ],
                relationships={
                    "balance_sheet": "Impacts through P&L",
                    "forecast_sheet": "Actual vs forecast comparison"
                },
                columns = {
                    "Operator": ColumnDefinition(
                        description="Name of the operating entity or organization",
                        hierarchy_level=1,
                        distinct_values=['Marriott', 'HHM', 'Remington', '24/7']
                    ),
                    "SQL_Property": ColumnDefinition(
                        description="List of hotel properties in the portfolio, including various brands and locations across the United States",
                        hierarchy_level=2,
                        distinct_values=['AC Wailea', 'Courtyard LA Pasadena Old Town', 'Courtyard Washington DC Dupont Circle', 'Hilton Garden Inn Bethesda', 'Marriott Crystal City', 'Moxy Washington DC Downtown', 'Residence Inn Pasadena', 'Residence Inn Westshore Tampa', 'Skyrock Inn Sedona', 'Steward\\xa0Santa\\xa0Barbara', 'Surfrider Malibu']
                    ),
                    "SQL_Account_Name": ColumnDefinition(
                        description="Main accounting categories for hotel operations including revenue, expenses, and profit metrics",
                        hierarchy_level=3,
                        distinct_values=['Operational Data', 'Replacement Reserve', 'Net Operating Income after Reserve', 'Revenue', 'Department Expenses', 'Department Profit (Loss', 'Undistributed Expenses', 'Gross Operating Profit', 'Management Fees', 'Income Before Non-Operating Inc & Exp', 'Non-Operating Income & Expenses', 'Total Non-Operating Income & Expenses', 'EBITDA', '-']
                    ),
                    "SQL_Account_Category_Order": ColumnDefinition(
                        description="Detailed breakdown of hotel performance metrics, ordered from operational statistics through financial results including occupancy, revenue, expenses, and profitability measures",
                        hierarchy_level=4,
                        distinct_values=['Available Rooms', 'Rooms Sold', 'Occupancy %', 'Average Rate', 'RevPar', 'Replacement Reserve', 'NOI after Reserve', 'NOI Margin', 'Room Revenue', 'F&B Revenue', 'Other Revenue', 'Miscellaneous Income', 'Total Operating Revenue', 'Room Expense', 'F&B Expense', 'Other Expense', 'Total Department Expense', 'Department Profit (Loss)', 'A&G Expense', 'Information & Telecommunications', 'Sales & Marketing', 'Maintenance', 'Utilities', 'Total Undistributed Expenses', 'GOP', 'GOP Margin', 'Management Fees', 'Income Before Non-Operating Inc & Exp', 'Property & Other Taxes', 'Insurance', 'Other (Non-Operating I&E)', 'Total Non-Operating Income & Expenses', 'EBITDA']
                    ),
                    "Sub_Account_Category_Order": ColumnDefinition(
                        description="Detailed breakdown of hotel performance metrics, ordered from operational statistics through financial results including occupancy, revenue, expenses, and profitability measures",
                        hierarchy_level=5,
                        distinct_values=['-', 'Replacement Reserve', 'EBITDA less REPLACEMENT RESERVE', 'Rooms', 'Food & Beverage', 'Other', 'Market', 'Rooms Other', 'Benefits/Bonus % Wages', 'Overtime Premium', 'Hourly Wages', 'Management Wages', 'FTG InRoom Services', 'Walked Guest', 'TA Commission', 'Cluster Reservation Cost', 'Comp F&B', 'Guest Supplies', 'Suite Supplies', 'Laundry', 'Cleaning Supplies', 'Linen', 'F&B Other', 'Service Charge Distribution', 'Beverage Cost', 'Food Cost', 'Other Sales Expense', 'Market Expense', 'A&G Other', 'Uniforms', 'Program Services Contribution', 'Transportation/Van Expense', 'Chargebacks', 'Employee Relations', 'Training', 'Postage', 'Bad Debt', 'Credit and Collection', 'Travel', 'Office Supplies', 'Pandemic Preparedness', 'Outside Labor Services', 'TOTAL I&TS CONT.', 'IT Compliance', 'FTG Internet', 'Guest Communications', 'Sales & Mkt. Other', 'Revenue Management', 'BT Booking Cost', 'Sales Shared Services', 'Loyalty', 'Marketing & eCommerce', 'Marketing Fund', 'PO&M Other', 'Cluster Engineering', 'PO&M NonContract', 'PO&M Contract', 'UTILITIES', 'Gross Operating Profit', 'Management Fees', 'Real Estate Tax', 'Over/Under Sales Tax', 'Property Insurance', 'Casualty Insurance', 'Other Investment Factors', 'Gain Loss Fx', 'Prior Year Adjustment', 'Lease Payments', 'Chain Services', 'Land Rent', 'Guest Accidents', 'Franchise Fees', 'System Fees', 'EBITDA', 'NOI after Reserve', 'Net Income', 'Other Operated Departments', 'Administrative & General', 'ADMINISTRATIVE & GENERAL', 'INFORMATION & TELECOMM.', 'Information & Telecommunications', 'FRANCHISE FEES', 'Sales & Marketing', 'Available Rooms', 'Property Operations & Maintenance', 'Utilities', 'Property & Other Taxes', 'Real Estate Property Tax', 'Personal Property Tax', 'Business Tax', 'Insurance - Property', 'Insurance General', 'Cyber Insurance', 'Employment Practices Insurance', 'Insurance', 'Professional Services', 'Legal & Accounting', 'Interest', 'Interest Expense-other', 'Lease Income', 'Total Food and Beverage', 'Total Other Operated Departments', 'Miscellaneous Income', 'Minor Ops', 'Franchise Taxes Owner', 'Other Expense', 'Total Other Operated Departments Expense', 'Miscellaneous Expense', 'Information & Telecommunications Sys.', 'MANAGEMENT FEE', 'REAL ESTATE/OTHER TAXES', 'HOTEL BED TAX CONTR', 'Property & Other taxes', 'Income', 'Rent & Leases', 'FFE Replacement Exp', 'Ownership Expense Owner', 'Depreciation and Amortization', 'Owner Expenses', 'EXTERNAL AUDIT FEES', 'DEFERRED MAINT. PRE-OPENING', 'COMMON AREA', 'Rent', 'RENT BASE', 'RENT VARIABLE', 'TRS LATE FEE', 'RATELOCK EXPENSE', 'BUDGET VARIANCE', 'CORPORATE OVERHEAD', 'OFFICE BLDG CASH FL', 'PROF SVCS-LEGAL', 'PROF SVCS', 'PROF SVCS-ENVIRONMENTAL', 'PROF SVCS-ACCOUNTING', 'PROF SVCS-OTHER', 'BAD DEBT EXPENSE', 'INCENTIVE MANAGEMENT FEE', 'PRE-OPENING EXPENSE', 'AMORTIZATION EXPENSE', 'OID W/O', 'PROCEEDS FROM CONVERSION', 'BASIS OF N/R', 'LONG TERM CAPITAL GAIN', 'OVERHEAD ALLOCATION', 'INTEREST EXPENSE', 'Asset Management Fee', 'Rent & Other Property/Equipment', 'Marketing Training', 'Prior Year Adj Tax', 'Property Tax', 'ASSET MANAGEMENT FEES', 'Management Fee Expense', 'NET OPERATING INCOME', 'ROOMS', 'FOOD & BEVERAGE', 'OTHER INCOME', 'SALES & MARKETING', 'REPAIRS & MAINTENANCE', 'PROPERTY TAX', 'PERSONAL PROPERTY TAX', 'LIABILITY INSURANCE', 'EQUIPMENT LEASES', "OWNER'S EXPENSE", 'LOAN INTEREST', 'ASSET MANAGEMENT FEE', 'REPLACEMENT RESERVES', 'Minibar', 'Mini Bar', 'Info & Telecom Systems', 'Property Operations', 'Interest Expense', 'Owner Expense', 'Reserve for Replacement']
                    ),
                    "SQL_Account_Group_Name": ColumnDefinition(
                        description=" ",
                        hierarchy_level=6,
                        distinct_values=['-', 'EBITDA less REPLACEMENT RESERVE', 'Rooms', 'Food & Beverage', 'Other', 'Guest Communications', 'Market', 'Rooms Other', 'Incentive Expense', 'Payroll Taxes', "Workers' Comp", 'Bonus', 'Medical', 'Overtime Premium', 'Hourly Wages', 'Management Wages', 'FTG InRoom Services', 'Walked Guest', 'TA Commission', 'Cluster Reservation Cost', 'Comp F&B', 'Guest Supplies', 'Suite Supplies', 'Laundry', 'Cleaning Supplies', 'Linen', 'F&B Other', 'Service Charge Distribution', 'Beverage Cost', 'Food Cost', 'Other Sales Expense', 'Market Expense', 'Uniforms', 'CAS System Support', 'Over/Short', 'A&G Other', 'Program Services Contribution', 'Transportation/Van Expense', 'Chargebacks', 'Employee Relations', 'Training', 'Postage', 'Bad Debt', 'Credit and Collection', 'Travel', 'Office Supplies', 'Pandemic Preparedness', 'Outside Labor Services', 'TOTAL I&TS CONT.', 'IT Compliance', 'FTG Internet', 'Sales Executive Share', 'Sales Exec Overhead Dept', 'Revenue Management', 'BT Booking Cost', 'Sales Shared Services', 'Loyalty', 'Marketing & eCommerce', 'Marketing Fund', 'PO&M Other', 'Cluster Engineering', 'PO&M NonContract', 'PO&M Contract', 'UTILITIES', 'Water/Sewer', 'Gas', 'Electricity', 'Gross Operating Profit', 'Real Estate Tax', 'Over/Under Sales Tax', 'Property Insurance', 'Casualty Insurance', 'Other Investment Factors', '71132 Common Area Chgs', 'Gain Loss Fx', 'Prior Year Adjustment', 'Lease Payments', 'Chain Services', 'Land Rent', 'Guest Accidents', 'Franchise Fees', 'System Fees', 'EBITDA', 'Marketing Training', 'Prior Year Adj Tax', 'Property Tax']
                    ),
                    "Current_Actual_Month": ColumnDefinition(
                        description=" "
                    ),
                    "YoY_Change": ColumnDefinition(
                        description=" "
                    ),
                    "Month": ColumnDefinition(
                        description="Time period for the data in YYYY-MM-DD format. When querying specific months (e.g., 'June 2024'), use format '2024-06-01' in SQL. Supports dates from January 2021 through October 2024. For month-specific queries, use strftime or date functions to match the format.",
                        distinct_values=['2024-10-01', '2024-08-01', '2024-09-01', '2024-07-01', '2024-06-01', '2024-04-01', '2022-11-01', '2024-05-01', '2022-05-01', '2022-03-01', '2022-02-01', '2021-12-01', '2023-03-01', '2023-01-01', '2023-04-01', '2023-02-01', '2024-01-01', '2023-12-01', '2024-02-01', '2022-12-01', '2022-10-01', '2023-10-01', '2023-09-01', '2023-08-01', '2023-11-01', '2022-08-01', '2022-06-01', '2022-04-01', '2022-07-01', '2022-09-01', '2022-01-01', '2021-11-01', '2021-10-01', '2021-08-01', '2023-06-01', '2023-05-01', '2023-07-01', '2021-09-01', '2024-03-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-03-01', '2021-04-01', '2021-02-01', '2021-01-01']
                    )
                }
            )
        }

    def get_table_info(self, table_name):
        return self.tables.get(table_name)

    def get_column_info(self, table_name, column_name):
        table = self.tables.get(table_name)
        if table:
            return table.columns.get(column_name)
        return None

    def get_metadata_prompt(self) -> str:
        """Generate a prompt with metadata information for the SQL agent"""
        prompt = "Database Schema Information:\n\n"
        for table_name, table in self.tables.items():
            prompt += f"Table: {table_name}\n"
            prompt += f"Description: {table.description}\n"
            prompt += "Key Purposes:\n" + "\n".join(f"- {purpose}" for purpose in table.key_purposes) + "\n"
            prompt += "Common Queries:\n" + "\n".join(f"- {query}" for query in table.common_queries) + "\n"
            prompt += "Relationships:\n"
            for related_table, relationship in table.relationships.items():
                prompt += f"- {related_table}: {relationship}\n"
            prompt += "\n"
        return prompt

# Part 4: Main DatabaseAnalyst Class
class DatabaseAnalyst:
    def __init__(self, config: Config):
        print("Initializing DatabaseAnalyst...")
        self.llm = ChatAnthropic(
            model=config.model_name,
            temperature=0,
            api_key=config.api_key,
            max_tokens=4096
        )
        
        try:
            print("Connecting to database...")
            self.db_connection = sqlite3.connect(config.db_path)
            self.db = SQLDatabase.from_uri(
                config.sqlite_path,
                sample_rows_in_table_info=0,
                view_support=False,
                indexes_in_table_info=False
            )
            print("Database connection successful")
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            raise ConfigError(f"Database connection failed: {str(e)}")
        
        print("Loading metadata...")
        self.metadata = FinancialTableMetadata()
        
        # Initialize cache
        self.query_cache = {}
        self.cache_ttl = config.cache_ttl
        self.cache_enabled = config.cache_enabled
        self.cache_dir = config.cache_dir
        
        if self.cache_enabled:
            print("Initializing cache...")
            os.makedirs(self.cache_dir, exist_ok=True)
            self.load_cache()
            print("Cache initialized")

        # Load only the financial tables
        self.financial_tables = self.metadata.tables.keys()  # Get the keys of the financial tables

    def get_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query"""
        # Create a hash of the query to use as the cache key
        return hashlib.md5(query.encode()).hexdigest()

    def load_cache(self) -> None:
        """Load the query cache from disk"""
        if not self.cache_enabled:
            return
            
        cache_file = os.path.join(self.cache_dir, "query_cache.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Filter out expired cache entries
                    current_time = time.time()
                    self.query_cache = {
                        k: v for k, v in cached_data.items()
                        if current_time - v.get('timestamp', 0) < self.cache_ttl
                    }
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.query_cache = {}

    def save_cache(self) -> None:
        """Save the query cache to disk"""
        if not self.cache_enabled:
            return
            
        cache_file = os.path.join(self.cache_dir, "query_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def clear_cache(self) -> None:
        """Clear the query cache"""
        self.query_cache = {}
        if self.cache_enabled:
            cache_file = os.path.join(self.cache_dir, "query_cache.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)

    def get_cached_result(self, query: str) -> Optional[Dict]:
        """Get a cached result if it exists and is not expired"""
        if not self.cache_enabled:
            return None
            
        cache_key = self.get_cache_key(query)
        cached_data = self.query_cache.get(cache_key)
        
        if cached_data:
            # Check if cache entry is expired
            if time.time() - cached_data.get('timestamp', 0) < self.cache_ttl:
                return cached_data.get('result')
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
                
        return None

    def cache_result(self, query: str, result: Dict) -> None:
        """Cache a query result"""
        if not self.cache_enabled:
            return
            
        cache_key = self.get_cache_key(query)
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        self.save_cache()

    def _select_relevant_tables(self, query: str) -> List[str]:
        """Step 1: Select relevant tables based on query and table descriptions"""
        print(f"\nStep 1: Selecting relevant tables for query: {query}")
        tables_prompt = (
            "Given the user question and the available financial tables, identify which tables are needed:\n\n"
            f"User Question: {query}\n\n"
            "Available Financial Tables:\n"
        )
        
        for table_name in self.financial_tables:
            table = self.metadata.tables[table_name]
            tables_prompt += (
                f"\nTable: {table_name}\n"
                f"Description: {table.description}\n"
                "Common Queries:\n"
                + "\n".join(f"- {q}" for q in table.common_queries) + "\n"
            )
        
        tables_prompt += "\nReturn ONLY a JSON array of table names needed. Example: [\"table1\", \"table2\"]"
        
        print("Prompt sent to LLM:")
        print(tables_prompt)
        
        response = self.llm.invoke([HumanMessage(content=tables_prompt)])
        try:
            match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if match:
                selected_tables = json.loads(match.group())
                print(f"Selected tables: {selected_tables}")
                return selected_tables 
            print("No tables selected")
            return []
        except Exception as e:
            print(f"Error parsing table selection: {e}")
            return []

    def _analyze_column_requirements(self, query: str, selected_tables: List[str]) -> Dict[str, List[str]]:
        """Step 2: Analyze which columns are needed based on hierarchy and descriptions."""
        print(f"\nStep 2: Analyzing column requirements for tables: {selected_tables}")
        columns_prompt = f"Given this user question: {query}\n\n"
        columns_prompt += "And these table columns with their descriptions:\n\n"
        
        for table_name in selected_tables:
            table = self.metadata.tables.get(table_name)
            if table and table.columns:
                columns_prompt += f"\nTable: {table_name}\n"
                for col_name, col_def in table.columns.items():
                    columns_prompt += f"\nColumn: {col_name}\n"
                    columns_prompt += f"Description: {col_def.description}\n"
                    # Only include hierarchy level if it exists and is relevant
                    if col_def.hierarchy_level and col_def.hierarchy_level != 'none':
                        columns_prompt += f"Hierarchy Level: {col_def.hierarchy_level}\n"
                    # Only include a sample of distinct values if they exist
                    if col_def.distinct_values and len(col_def.distinct_values) > 0:
                        sample_size = min(5, len(col_def.distinct_values))
                        samples = col_def.distinct_values[:sample_size]
                        columns_prompt += f"Example Values: {', '.join(str(v) for v in samples)}\n"
        
        columns_prompt += "\nReturn a JSON object mapping table names to needed columns. Example:"
        columns_prompt += '\n{"table1": ["column1", "column2"], "table2": ["column3"]}'
        
        print("Sending column analysis prompt to LLM...")
        response = self.llm.invoke([HumanMessage(content=columns_prompt)])
        try:
            match = re.search(r'{.*}', response.content, re.DOTALL)
            if match:
                columns = json.loads(match.group())
                print(f"Selected columns: {columns}")
                return columns
            print("No columns selected")
            return {}
        except Exception as e:
            print(f"Error parsing column requirements: {e}")
            return {}

    def _generate_sql_query(self, query: str, table_columns: Dict[str, List[str]]) -> str:
        """Step 3: Generate SQL query based on selected tables and columns. Ensure that your SQL code uses only the distinct values present in the specified columns.  If a value is not present in the column, do not write WHERE clause using that value and that column."""
        print(f"\nStep 3: Generating SQL query for: {query}")
        
        # Create a more concise prompt
        sql_prompt = f"""Generate a SQL query for: {query}

Tables and columns to use:
{json.dumps(table_columns, indent=2)}

Important Notes:
1. Month format: YYYY-MM-DD (e.g., '2024-06-01' for June 2024)
2. Month matching:
   - "June 2024" → Month = '2024-06-01'
   - "June" → strftime('%m', Month) = '06'
   - "2024" → strftime('%Y', Month) = '2024'

Generate SQL with correct date handling.
"""
        
        print("Sending SQL generation prompt to LLM...")
        response = self.llm.invoke([HumanMessage(content=sql_prompt)])
        sql = self._extract_sql(response.content)
        print(f"Generated SQL: {sql}")
        return sql

    def _analyze_results(self, query: str, results: List[Dict]) -> str:
        """Step 4: Analyze results with a more concise prompt"""
        print(f"\nStep 4: Analyzing results for query: {query}")
        
        # Limit the number of results included in the prompt
        max_results = 10
        truncated_results = results[:max_results] if len(results) > max_results else results
        
        analysis_prompt = (
            f"Analyze these results for: {query}\n\n"
            f"Results (showing {len(truncated_results)} of {len(results)} rows):\n"
            f"{json.dumps(truncated_results, indent=2)}\n\n"
            "Provide a clear, concise analysis of key insights."
        )
        
        print("Sending analysis prompt to LLM...")
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        print(f"Analysis generated: {response.content[:100]}...")
        return response.content

    def process_query(self, query: str) -> Dict:
        """Main method to process user query"""
        print(f"\nProcessing query: {query}")
        
        # Check cache first
        cached_result = self.get_cached_result(query)
        if cached_result:
            print("Found cached result")
            return cached_result

        try:
            # Step 1: Select relevant tables
            selected_tables = self._select_relevant_tables(query)
            if not selected_tables:
                print("Error: Could not determine relevant tables")
                return {"success": False, "error": "Could not determine relevant tables"}

            # Step 2: Analyze column requirements
            table_columns = self._analyze_column_requirements(query, selected_tables)
            if not table_columns:
                print("Error: Could not determine required columns")
                return {"success": False, "error": "Could not determine required columns"}

            # Step 3: Generate and execute SQL query
            sql_query = self._generate_sql_query(query, table_columns)
            if not sql_query:
                print("Error: Could not generate SQL query")
                return {"success": False, "error": "Could not generate SQL query"}

            # Execute query
            print("Executing SQL query...")
            cursor = self.db_connection.cursor()
            cursor.execute("PRAGMA timeout = 5000")
            cursor.execute(sql_query)
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            if not results:
                print("No results found")
                return {
                    "success": True,
                    "metrics": {
                        "Summary": "No data found for your query.",
                        "Data": []
                    }
                }

            # Step 4: Analyze results
            print("Analyzing results...")
            analysis = self._analyze_results(query, results)

            result = {
                "success": True,
                "metrics": {
                    "Summary": analysis,
                    "Data": results,
                    "Tables_Used": selected_tables,
                    "Columns_Used": table_columns
                }
            }

            # Cache the result
            print("Caching results...")
            self.cache_result(query, result)
            print("Query processing complete")
            return result

        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from text with improved pattern matching and validation"""
        # First try to find SQL within code blocks
        patterns = [
            r"```sql\n(.*?)\n```",     # Standard markdown SQL blocks
            r"```\n(.*?)\n```",        # Generic code blocks
            r"```(.*?)```",            # Single-line code blocks
            r"SELECT[\s\S]*?;",        # Direct SQL statements (multiline)
            r"WITH[\s\S]*?;",          # CTE-style queries (multiline)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                sql = match.group(1) if '```' in pattern else match.group(0)
                sql = sql.strip()
                
                # Basic validation that it's actually SQL
                if any(keyword.upper() in sql.upper() for keyword in ['SELECT', 'WITH']):
                    # Clean up any remaining markdown artifacts
                    sql = sql.replace('```sql', '').replace('```', '').strip()
                    return sql
        
        # If no valid SQL found in code blocks, try to find direct SQL statements
        if 'SELECT' in text.upper() or 'WITH' in text.upper():
            # Extract everything from SELECT/WITH to the next semicolon
            match = re.search(r'(?:SELECT|WITH)[\s\S]*?;', text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return ""

    def _execute_sql_safely(self, query: str) -> Dict:
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query)
            
            metrics = {}
            for row in cursor.fetchall():
                metrics[str(row[0])] = row[1] if len(row) > 1 else row[0]
            
            return {"metrics": metrics, "success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze(self, query: str) -> Dict:
        if query in self.query_cache:
            return self.query_cache[query]
        
        state = AnalysisState(user_query=query)
        result = self.process_query(state)
        self.query_cache[query] = result
        return result

    def process_follow_up(self, follow_up_question: str, previous_context: Dict) -> Dict:
        # Add the context to the prompt
        context_prompt = f"""
        Previous context: {previous_context.get('metrics', {})}
        Follow-up question: {follow_up_question}
        
        Please answer the follow-up question using the context provided.
        """
        
        # Process the follow-up using the context
        result = self.process_query(context_prompt)
        return result

    def redo_analysis(self, query: str, clear_cache: bool = True) -> Dict:
        """Redo analysis with optimized prompts"""
        if clear_cache:
            cache_key = self.get_cache_key(query)
            if cache_key in self.query_cache:
                del self.query_cache[cache_key]
        
        try:
            return self.process_query(query)
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

    def _get_relevant_columns_metadata(self, selected_tables: List[str]) -> str:
        """Get detailed column metadata only for selected tables"""
        columns_metadata = ""
        for table_name in selected_tables:
            table = self.metadata.tables.get(table_name)
            if table and hasattr(table, 'columns'):
                columns_metadata += f"\nTable: {table_name}\n"
                columns_metadata += f"Description: {table.description}\n"
                if table.columns:
                    columns_metadata += "Columns:\n"
                    for col_name, col_def in table.columns.items():
                        columns_metadata += f"- {col_name}:\n"
                        if col_def.hierarchy_level:
                            columns_metadata += f"  Hierarchy Level: {col_def.hierarchy_level}\n"
                        if col_def.distinct_values:
                            columns_metadata += f"  Possible Values: {', '.join(str(v) for v in col_def.distinct_values)}\n"
                        columns_metadata += "\n"
        return columns_metadata

    def _handle_sql_extraction_error(self, response_text: str) -> str:
        """Attempt to recover SQL from malformed response"""
        # Try various cleanup patterns
        patterns = [
            (r'````sql\s*(.*?)\s*````', r'```sql\n\1\n```'),
            (r'```sql\s*(.*?)\s*```', r'```sql\n\1\n```'),
            (r'SELECT\s+.*?;', lambda m: f'```sql\n{m.group(0)}\n```'),
            (r'WITH\s+.*?;', lambda m: f'```sql\n{m.group(0)}\n```')
        ]
        
        text = response_text
        for pattern, replacement in patterns:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
                break
        
        return text

def format_output(results: Dict) -> str:
    output = []
    output.append("=== Database Analysis Results ===")
    output.append(f"\nQuery: {results.get('user_query', 'N/A')}")
    
    if "error" in results:
        output.append(f"\nError: {results['error']}")
        return "\n".join(output)
    
    if results.get('query_type') == 'direct_sql':
        output.append(f"\nSQL Query: {results.get('sql_query', 'N/A')}")
        output.append("\nResults:")
        if isinstance(results.get('results'), list):
            df = pd.DataFrame(results['results'])
            output.append(str(df))
        else:
            output.append(str(results.get('results', 'No results available')))
    else:
        output.append("\nAnalysis:")
        if isinstance(results.get('sql_results'), dict):
            output.append(f"\nSQL Results: {results['sql_results'].get('result', 'No results available')}")
        else:
            output.append(f"\nSQL Results: {str(results.get('sql_results', 'No results available'))}")
        
        output.append("\nDetailed Analysis:")
        output.append(str(results.get('analysis', 'No analysis available')))
    
    return "\n".join(output)

def analyze_query(query: str) -> str:
    try:
        config = Config()
        analyst = DatabaseAnalyst(config)
        results = analyst.analyze(query)
        
        if results and "error" not in results:
            formatted_output = format_output(results)
            
            # Save the analysis to a JSON file
            output_data = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": formatted_output
            }
            
            # Create a safe filename from the query
            filename = f"{hashlib.md5(query.encode()).hexdigest()[:10]}_analysis.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            except Exception as save_error:
                return f"{formatted_output}\n\nWarning: Could not save results to file: {str(save_error)}"
                
            return formatted_output + f"\n\nDetailed results saved to {filename}"
        else:
            return f"Error: {results.get('error', 'Unknown error occurred')}"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Add new class for chat management
class ChatManager:
    def __init__(self):
        self.chats_dir = "saved_chats"
        os.makedirs(self.chats_dir, exist_ok=True)
        
    def save_chat(self, chat_id: str, messages: list):
        # Get chat title from first user message
        title = None
        first_query = None
        for message in messages:
            if message["role"] == "user":
                first_query = message["content"]
                # Clean and truncate the title
                clean_title = first_query.replace("\n", " ").strip()
                title = clean_title[:50] + "..." if len(clean_title) > 50 else clean_title
                break
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        chat_data = {
            "id": chat_id,
            "title": title,
            "first_query": first_query,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = os.path.join(self.chats_dir, f"{chat_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
    def load_chats(self) -> dict:
        chats = {}
        for filename in os.listdir(self.chats_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.chats_dir, filename), encoding='utf-8') as f:
                    chat_data = json.load(f)
                    chats[chat_data['id']] = chat_data
        return dict(sorted(chats.items(), key=lambda x: x[1]['timestamp'], reverse=True))
        
    def delete_chat(self, chat_id: str):
        """Delete a saved chat by its ID"""
        filename = os.path.join(self.chats_dir, f"{chat_id}.json")
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False

# Streamlit UI
def main():
    st.set_page_config(
        page_title="SQL Database Analyst",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("SQL Database Analysis Assistant")
    
    # Initialize chat manager and session state
    chat_manager = ChatManager()
    
    # Improved session state initialization
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'messages': [],
            'current_context': None,
            'current_chat_id': str(uuid.uuid4()),
            'chats': chat_manager.load_chats(),
            'last_query': None,
            'new_chat_clicked': False,
            'api_key_set': False
        })

    # Sidebar configuration with improved error handling
    with st.sidebar:
        st.title("Configuration Settings")
        api_key = st.text_input("Anthropic API Key:", type="password")
        
        if api_key:
            st.session_state.api_key_set = True
        
        st.title("Conversation Management")
        
        # New chat button with improved handling
        if st.button("Start New Analysis", type="primary", key="new_chat_button", disabled=not st.session_state.api_key_set):
            if not st.session_state.new_chat_clicked:
                st.session_state.new_chat_clicked = True
                
                # Save current chat if it exists
                if st.session_state.messages:
                    chat_manager.save_chat(
                        st.session_state.current_chat_id,
                        st.session_state.messages
                    )
                
                # Reset states with new chat
                new_chat_id = str(uuid.uuid4())
                st.session_state.update({
                    'current_chat_id': new_chat_id,
                    'messages': [],
                    'current_context': None,
                    'last_query': None
                })
                
                # Add welcome message
                welcome_message = {
                    "role": "assistant",
                    "content": """👋 Hello! I'm your SQL Database Analysis Assistant. 

I can help you analyze your database by:
- Running SQL queries
- Providing data insights
- Answering follow-up questions about the results

Please ask me any question about your database!"""
                }
                st.session_state.messages.append(welcome_message)
                st.rerun()
        
        # Display chat history with delete option
        st.title("Chat History")
        chats = chat_manager.load_chats()
        
        for chat_id, chat_data in chats.items():
            with st.expander(f"📝 {chat_data['title']}", expanded=False):
                st.write(f"Created: {datetime.fromisoformat(chat_data['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                
                # Create two columns for the buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Load Chat", key=f"load_{chat_id}"):
                        # Save current chat before loading
                        if st.session_state.messages:
                            chat_manager.save_chat(
                                st.session_state.current_chat_id,
                                st.session_state.messages
                            )
                        # Load selected chat
                        st.session_state.messages = chat_data['messages']
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{chat_id}", type="secondary"):
                        # Delete the chat
                        chat_manager.delete_chat(chat_id)
                        # If the deleted chat was the current one, start a new chat
                        if chat_id == st.session_state.current_chat_id:
                            st.session_state.current_chat_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        # Refresh the chat list
                        st.session_state.chats = chat_manager.load_chats()
                        st.rerun()

    # Reset new_chat_clicked flag
    if st.session_state.new_chat_clicked:
        st.session_state.new_chat_clicked = False

    # API key validation
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to proceed.")
        return

    # Initialize analyst with error handling
    try:
        config = Config(api_key=api_key)
        analyst = DatabaseAnalyst(config)
    except Exception as e:
        st.error(f"Failed to initialize database analyst: {str(e)}")
        return

    # Chat interface with improved message display, redo option, and edit query
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                # Add edit functionality for user queries
                if f"edit_mode_{idx}" not in st.session_state:
                    st.session_state[f"edit_mode_{idx}"] = False

                if st.session_state[f"edit_mode_{idx}"]:
                    # Show edit interface
                    edited_query = st.text_area(
                        "Edit your query:",
                        value=message["content"],
                        key=f"edit_query_{idx}"
                    )
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Save & Run", key=f"save_edit_{idx}"):
                            # Update the query
                            st.session_state.messages[idx]["content"] = edited_query
                            
                            # Remove the old response
                            if idx + 1 < len(st.session_state.messages) and st.session_state.messages[idx + 1]["role"] == "assistant":
                                st.session_state.messages.pop(idx + 1)
                            
                            # Run new analysis
                            with st.spinner("Running new analysis..."):
                                try:
                                    result = analyst.process_query(edited_query)
                                    if result["success"]:
                                        response = "🎯 Results for edited query:\n\n"
                                        for metric, value in result['metrics'].items():
                                            response += f"**{metric}**: {value}\n"
                                    else:
                                        response = f"❌ Error: {result.get('error', 'Unknown error')}"
                                    
                                    # Insert new response
                                    st.session_state.messages.insert(idx + 1, {
                                        "role": "assistant",
                                        "content": response
                                    })
                                    
                                    # Save chat
                                    chat_manager.save_chat(
                                        st.session_state.current_chat_id,
                                        st.session_state.messages
                                    )
                                    
                                    # Exit edit mode
                                    st.session_state[f"edit_mode_{idx}"] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error processing edited query: {str(e)}")
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_edit_{idx}"):
                            st.session_state[f"edit_mode_{idx}"] = False
                            st.rerun()
                else:
                    # Show normal query with edit button
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(message["content"])
                    with col2:
                        if st.button("✏️ Edit", key=f"edit_{idx}"):
                            st.session_state[f"edit_mode_{idx}"] = True
                            st.rerun()
                st.caption("Question")
            else:
                # Show assistant messages normally
                st.markdown(message["content"])
                
                if message["role"] == "assistant":
                    # Create columns for action buttons
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if st.button("🔄 Redo", key=f"redo_{idx}", help="Redo this analysis with fresh results"):
                            # Get the corresponding user query
                            user_query = None
                            for i in range(idx, -1, -1):
                                if st.session_state.messages[i]["role"] == "user":
                                    user_query = st.session_state.messages[i]["content"]
                                    break
                            
                            if user_query:
                                with st.spinner("🔄 Redoing analysis..."):
                                    try:
                                        # Redo analysis with cleared cache
                                        result = analyst.redo_analysis(user_query, clear_cache=True)
                                        
                                        if result["success"]:
                                            # Store the original response
                                            original_response = message["content"]
                                            
                                            # Create new response
                                            new_response = "🔄 **New Analysis Results:**\n\n"
                                            for metric, value in result['metrics'].items():
                                                new_response += f"**{metric}**: {value}\n"
                                            new_response += f"\n\n_Reanalyzed at: {datetime.now().strftime('%H:%M:%S')}_"
                                            
                                            # Create combined response with original and new results
                                            combined_response = (
                                                "📊 **Original Analysis:**\n\n"
                                                f"{original_response}\n\n"
                                                "---\n\n"  # Separator
                                                f"{new_response}"
                                            )
                                            
                                            # Update the message content
                                            st.session_state.messages[idx]["content"] = combined_response
                                            
                                            # Add comparison note if results differ
                                            if original_response != new_response:
                                                st.info("💡 The results have changed from the original analysis.")
                                            
                                            # Save chat after reanalysis
                                            chat_manager.save_chat(
                                                st.session_state.current_chat_id,
                                                st.session_state.messages
                                            )
                                            st.rerun()
                                        else:
                                            st.error(f"Redo failed: {result.get('error', 'Unknown error')}")
                                    except Exception as e:
                                        st.error(f"Error during reanalysis: {str(e)}")
                    
                    with col2:
                        # Show analysis information
                        if "Original Analysis:" in message["content"]:
                            st.caption("Contains original and reanalyzed results")
                
                if idx > 0 and st.session_state.messages[idx-1]["role"] == "user":
                    st.caption("Follow-up available ↓")

    # Add a redo button for the entire conversation with comparison
    if st.session_state.messages and len(st.session_state.messages) > 1:
        if st.sidebar.button("🔄 Redo Entire Analysis", help="Redo all queries in this conversation"):
            with st.spinner("Redoing entire analysis..."):
                try:
                    new_messages = [st.session_state.messages[0]]  # Keep the welcome message
                    
                    # Redo each query in the conversation
                    for idx, message in enumerate(st.session_state.messages[1:]):
                        if message["role"] == "user":
                            # Add the user message
                            new_messages.append(message)
                            
                            # Get original response if it exists
                            original_response = None
                            if idx + 2 < len(st.session_state.messages):
                                original_response = st.session_state.messages[idx + 2]["content"]
                            
                            # Redo the analysis
                            result = analyst.redo_analysis(message["content"], clear_cache=True)
                            
                            if result["success"]:
                                if original_response:
                                    response = (
                                        "📊 **Original Analysis:**\n\n"
                                        f"{original_response}\n\n"
                                        "---\n\n"
                                        "🔄 **New Analysis Results:**\n\n"
                                    )
                                else:
                                    response = "🔄 **Analysis Results:**\n\n"
                                
                                for metric, value in result['metrics'].items():
                                    response += f"**{metric}**: {value}\n"
                                response += f"\n\n_Reanalyzed at: {datetime.now().strftime('%H:%M:%S')}_"
                            else:
                                response = f"❌ Error: {result.get('error', 'Unknown error')}"
                            
                            # Add the assistant response
                            new_messages.append({
                                "role": "assistant",
                                "content": response
                            })
                    
                    # Update messages and save
                    st.session_state.messages = new_messages
                    chat_manager.save_chat(
                        st.session_state.current_chat_id,
                        st.session_state.messages
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during complete reanalysis: {str(e)}")

    # Chat input with automatic saving
    if prompt := st.chat_input("Ask me about your database...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Process query with context
                    context = None
                    if len(st.session_state.messages) > 2:  # If there's previous conversation
                        context = st.session_state.messages[-3]["content"] if len(st.session_state.messages) >= 3 else None
                    
                    result = analyst.process_query(prompt)
                    
                    if result["success"]:
                        response = "🎯 Here are the results:\n\n"
                        for metric, value in result['metrics'].items():
                            response += f"**{metric}**: {value}\n"
                    else:
                        response = f"❌ Error: {result.get('error', 'Unknown error')}"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save chat after each interaction
                    chat_manager.save_chat(
                        st.session_state.current_chat_id,
                        st.session_state.messages
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()