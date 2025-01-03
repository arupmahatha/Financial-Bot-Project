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
from langchain.memory import ConversationSummaryMemory
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.tools import Tool
from functools import lru_cache
import hashlib
import pickle
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple, Any
import torch
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import nltk
import itertools
import ssl
import traceback
import diskcache

# Add this after your imports
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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
                        description="Name of the operating entity or organization. Every operator manages multiple properties. hierarchy_level=1",
                        distinct_values=['Marriott', 'HHM', 'Remington', '24/7']
                    ),
                    "SQL_Property": ColumnDefinition(
                        description="List of hotel properties in the portfolio, including various brands and locations across the United States. Every property is managed by an operator. hierarchy_level=2",
                        distinct_values=['AC Wailea', 'Courtyard LA Pasadena Old Town', 'Courtyard Washington DC Dupont Circle', 'Hilton Garden Inn Bethesda', 'Marriott Crystal City', 'Moxy Washington DC Downtown', 'Residence Inn Pasadena', 'Residence Inn Westshore Tampa', 'Skyrock Inn Sedona', 'Steward Santa Barbara', 'Surfrider Malibu']
                    ),
                    "SQL_Account_Name": ColumnDefinition(
                        description="This column categorizes financial data into various account types, including operational data, reserves, income, expenses, profits, and fees. It also includes categories for non-operating income and expenses, as well as EBITDA. hierarchy_level=3",
                        distinct_values=['Operational Data', 'Replacement Reserve', 'Net Operating Income after Reserve', 'Revenue', 'Department Expenses', 'Department Profit (Loss', 'Undistributed Expenses', 'Gross Operating Profit', 'Management Fees', 'Income Before Non-Operating Inc & Exp', 'Non-Operating Income & Expenses', 'Total Non-Operating Income & Expenses', 'EBITDA', '-']
                    ),
                    "SQL_Account_Category_Order": ColumnDefinition(
                        description="Breakdown of (SQL_Account_Name) into more specific categories. Example: Under (Department Expense) from (SQL_Account_Name) there are 4 sub-categories in SQL_Account_Category_Order. hierarchy_level=4",
                        distinct_values=['Available Rooms', 'Rooms Sold', 'Occupancy %', 'Average Rate', 'RevPar', 'Replacement Reserve', 'NOI after Reserve', 'NOI Margin', 'Room Revenue', 'F&B Revenue', 'Other Revenue', 'Miscellaneous Income', 'Total Operating Revenue', 'Room Expense', 'F&B Expense', 'Other Expense', 'Total Department Expense', 'Department Profit (Loss)', 'A&G Expense', 'Information & Telecommunications', 'Sales & Marketing', 'Maintenance', 'Utilities', 'Total Undistributed Expenses', 'GOP', 'GOP Margin', 'Management Fees', 'Income Before Non-Operating Inc & Exp', 'Property & Other Taxes', 'Insurance', 'Other (Non-Operating I&E)', 'Total Non-Operating Income & Expenses', 'EBITDA']
                    ),
                    "Sub_Account_Category_Order": ColumnDefinition(
                        description="Breakdown of (SQL_Account_Category_Order) into more granular categorie. hierarchy_level=5",
                        distinct_values=['-', 'Replacement Reserve', 'EBITDA less REPLACEMENT RESERVE', 'Rooms', 'Food & Beverage', 'Other', 'Market', 'Rooms Other', 'Benefits/Bonus % Wages', 'Overtime Premium', 'Hourly Wages', 'Management Wages', 'FTG InRoom Services', 'Walked Guest', 'TA Commission', 'Cluster Reservation Cost', 'Comp F&B', 'Guest Supplies', 'Suite Supplies', 'Laundry', 'Cleaning Supplies', 'Linen', 'F&B Other', 'Service Charge Distribution', 'Beverage Cost', 'Food Cost', 'Other Sales Expense', 'Market Expense', 'A&G Other', 'Uniforms', 'Program Services Contribution', 'Transportation/Van Expense', 'Chargebacks', 'Employee Relations', 'Training', 'Postage', 'Bad Debt', 'Credit and Collection', 'Travel', 'Office Supplies', 'Pandemic Preparedness', 'Outside Labor Services', 'TOTAL I&TS CONT.', 'IT Compliance', 'FTG Internet', 'Guest Communications', 'Sales & Mkt. Other', 'Revenue Management', 'BT Booking Cost', 'Sales Shared Services', 'Loyalty', 'Marketing & eCommerce', 'Marketing Fund', 'PO&M Other', 'Cluster Engineering', 'PO&M NonContract', 'PO&M Contract', 'UTILITIES', 'Gross Operating Profit', 'Management Fees', 'Real Estate Tax', 'Over/Under Sales Tax', 'Property Insurance', 'Casualty Insurance', 'Other Investment Factors', 'Gain Loss Fx', 'Prior Year Adjustment', 'Lease Payments', 'Chain Services', 'Land Rent', 'Guest Accidents', 'Franchise Fees', 'System Fees', 'EBITDA', 'NOI after Reserve', 'Net Income', 'Other Operated Departments', 'Administrative & General', 'ADMINISTRATIVE & GENERAL', 'INFORMATION & TELECOMM.', 'Information & Telecommunications', 'FRANCHISE FEES', 'Sales & Marketing', 'Available Rooms', 'Property Operations & Maintenance', 'Utilities', 'Property & Other Taxes', 'Real Estate Property Tax', 'Personal Property Tax', 'Business Tax', 'Insurance - Property', 'Insurance General', 'Cyber Insurance', 'Employment Practices Insurance', 'Insurance', 'Professional Services', 'Legal & Accounting', 'Interest', 'Interest Expense-other', 'Lease Income', 'Total Food and Beverage', 'Total Other Operated Departments', 'Miscellaneous Income', 'Minor Ops', 'Franchise Taxes Owner', 'Other Expense', 'Total Other Operated Departments Expense', 'Miscellaneous Expense', 'Information & Telecommunications Sys.', 'MANAGEMENT FEE', 'REAL ESTATE/OTHER TAXES', 'HOTEL BED TAX CONTR', 'Property & Other taxes', 'Income', 'Rent & Leases', 'FFE Replacement Exp', 'Ownership Expense Owner', 'Depreciation and Amortization', 'Owner Expenses', 'EXTERNAL AUDIT FEES', 'DEFERRED MAINT. PRE-OPENING', 'COMMON AREA', 'Rent', 'RENT BASE', 'RENT VARIABLE', 'TRS LATE FEE', 'RATELOCK EXPENSE', 'BUDGET VARIANCE', 'CORPORATE OVERHEAD', 'OFFICE BLDG CASH FL', 'PROF SVCS-LEGAL', 'PROF SVCS', 'PROF SVCS-ENVIRONMENTAL', 'PROF SVCS-ACCOUNTING', 'PROF SVCS-OTHER', 'BAD DEBT EXPENSE', 'INCENTIVE MANAGEMENT FEE', 'PRE-OPENING EXPENSE', 'AMORTIZATION EXPENSE', 'OID W/O', 'PROCEEDS FROM CONVERSION', 'BASIS OF N/R', 'LONG TERM CAPITAL GAIN', 'OVERHEAD ALLOCATION', 'INTEREST EXPENSE', 'Asset Management Fee', 'Rent & Other Property/Equipment', 'Marketing Training', 'Prior Year Adj Tax', 'Property Tax', 'ASSET MANAGEMENT FEES', 'Management Fee Expense', 'NET OPERATING INCOME', 'ROOMS', 'FOOD & BEVERAGE', 'OTHER INCOME', 'SALES & MARKETING', 'REPAIRS & MAINTENANCE', 'PROPERTY TAX', 'PERSONAL PROPERTY TAX', 'LIABILITY INSURANCE', 'EQUIPMENT LEASES', "OWNER'S EXPENSE", 'LOAN INTEREST', 'ASSET MANAGEMENT FEE', 'REPLACEMENT RESERVES', 'Minibar', 'Mini Bar', 'Info & Telecom Systems', 'Property Operations', 'Interest Expense', 'Owner Expense', 'Reserve for Replacement']
                    ),
                    "SQL_Account_Group_Name": ColumnDefinition(
                        description="Further division of (Sub_Account_Category_Order). hierarchy_level=6",
                        distinct_values=['-', 'EBITDA less REPLACEMENT RESERVE', 'Rooms', 'Food & Beverage', 'Other', 'Guest Communications', 'Market', 'Rooms Other', 'Incentive Expense', 'Payroll Taxes', "Workers' Comp", 'Bonus', 'Medical', 'Overtime Premium', 'Hourly Wages', 'Management Wages', 'FTG InRoom Services', 'Walked Guest', 'TA Commission', 'Cluster Reservation Cost', 'Comp F&B', 'Guest Supplies', 'Suite Supplies', 'Laundry', 'Cleaning Supplies', 'Linen', 'F&B Other', 'Service Charge Distribution', 'Beverage Cost', 'Food Cost', 'Other Sales Expense', 'Market Expense', 'Uniforms', 'CAS System Support', 'Over/Short', 'A&G Other', 'Program Services Contribution', 'Transportation/Van Expense', 'Chargebacks', 'Employee Relations', 'Training', 'Postage', 'Bad Debt', 'Credit and Collection', 'Travel', 'Office Supplies', 'Pandemic Preparedness', 'Outside Labor Services', 'TOTAL I&TS CONT.', 'IT Compliance', 'FTG Internet', 'Sales Executive Share', 'Sales Exec Overhead Dept', 'Revenue Management', 'BT Booking Cost', 'Sales Shared Services', 'Loyalty', 'Marketing & eCommerce', 'Marketing Fund', 'PO&M Other', 'Cluster Engineering', 'PO&M NonContract', 'PO&M Contract', 'UTILITIES', 'Water/Sewer', 'Gas', 'Electricity', 'Gross Operating Profit', 'Real Estate Tax', 'Over/Under Sales Tax', 'Property Insurance', 'Casualty Insurance', 'Other Investment Factors', '71132 Common Area Chgs', 'Gain Loss Fx', 'Prior Year Adjustment', 'Lease Payments', 'Chain Services', 'Land Rent', 'Guest Accidents', 'Franchise Fees', 'System Fees', 'EBITDA', 'Marketing Training', 'Prior Year Adj Tax', 'Property Tax']
                    ),
                    "Current_Actual_Month": ColumnDefinition(
                        description="Actual financial performance for the month (income sheet). When a question is asked form the income sheet, use this column to do the aggregation/calculation for answering the queries"
                    ),
                    "YoY_Change": ColumnDefinition(
                        description="Percentage change compared to the same month in the prior year, computed for trend analysis"
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
        
        # Initialize caches
        self.query_cache = diskcache.Cache('.query_cache')
        self.prompt_cache = diskcache.Cache('.prompt_cache')
        
        # Initialize conversation memory with summary
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize database connection
        try:
            print("Initializing database connection...")
            self.db_connection = sqlite3.connect(config.db_path)
            self.db_connection.row_factory = sqlite3.Row
            print("Database connection established")
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            raise
        
        # Initialize metadata and query decomposer
        try:
            print("Loading metadata...")
            self.metadata = FinancialTableMetadata()
            
            # Initialize query decomposer with LLM only
            print("Initializing query decomposer...")
            self.query_decomposer = QueryDecomposer(self.llm)
            
            print("Database Analyst initialization complete")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def __del__(self):
        """Cleanup method to close database connection"""
        if hasattr(self, 'db_connection'):
            try:
                self.db_connection.close()
                print("Database connection closed")
            except Exception as e:
                print(f"Error closing database connection: {str(e)}")

    def _execute_sql(self, sql: str) -> Tuple[bool, Any, str]:
        """Execute SQL query with error handling"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return True, results, ""
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, None, error_msg

    def _process_single_query(self, query_info: Union[str, Dict]) -> Dict:
        """Process a single query or sub-query"""
        try:
            # Standardize query_info structure
            if isinstance(query_info, str):
                query_text = query_info
                entities = self._extract_entities(query_text)
            else:
                query_text = query_info.get('query', '')
                entities = query_info.get('entities', [])

            print(f"\nProcessing query: {query_text}")
            
            # Generate SQL using entity information
            query_data = {'query': query_text, 'entities': entities}
            sql_query = self._generate_sql_query(query_data)
            
            # Execute SQL
            print("\nExecuting SQL...")
            success, results, error = self._execute_sql(sql_query)
            
            if success and results:
                metrics = {}
                print("\nQuery Results:")
                
                # Convert results to a list of dictionaries
                formatted_results = []
                for row in results:
                    if isinstance(row, sqlite3.Row):
                        row_dict = dict(row)
                        formatted_results.append(row_dict)
                        print(f"• {row_dict}")
                    else:
                        row_dict = {"value": row[0]} if len(row) == 1 else {str(row[0]): row[1]}
                        formatted_results.append(row_dict)
                        print(f"• {row_dict}")
                
                # Store both the raw metrics and formatted results
                metrics = {
                    "raw_data": formatted_results,
                    "summary": {
                        "total_records": len(formatted_results),
                        "columns": list(formatted_results[0].keys()) if formatted_results else []
                    }
                }
                
                # Generate analysis
                analysis_text = self._analyze_results(query_text, formatted_results)
                
                return {
                    "success": True,
                    "query": query_text,
                    "entities": entities,
                    "sql_query": sql_query,
                    "metrics": metrics,
                    "formatted_results": formatted_results,
                    "analysis": analysis_text
                }
            else:
                return {
                    "success": False,
                    "error": error or "No results returned",
                    "query": query_text,
                    "entities": entities,
                    "sql_query": sql_query,
                    "metrics": {},
                    "formatted_results": [],
                    "analysis": ""
                }
                
        except Exception as e:
            error_msg = f"Query processing error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "query": query_text,
                "metrics": {},
                "formatted_results": [],
                "analysis": ""
            }

    def _combine_results(self, results: List[Dict], original_query: str) -> Dict:
        """Combine results from multiple sub-queries"""
        combined_metrics = {}
        all_entities = []
        all_sql_queries = []
        
        for result in results:
            if result.get("success", False):
                combined_metrics.update(result.get("metrics", {}))
                all_entities.extend(result.get("entities", []))
                all_sql_queries.append(result.get("sql_query", ""))
        
        # Generate combined analysis
        analysis_text = self._analyze_results(original_query, combined_metrics)
        
        return {
            "success": True,
            "original_query": original_query,
            "metrics": combined_metrics,
            "entities": all_entities,
            "sql_queries": all_sql_queries,
            "analysis": analysis_text
        }

    def get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query"""
        # Create a unique key based on the query
        return hashlib.md5(query.encode()).hexdigest()

    def clear_cache(self):
        """Clear the entire cache"""
        self.cache = {}

    def redo_analysis(self, query: str, clear_cache: bool = True) -> Dict:
        """Redo analysis with option to clear cache"""
        try:
            if clear_cache:
                self.clear_cache()
            
            print(f"\nRedoing analysis for query: {query}")
            return self.process_query(query)
            
        except Exception as e:
            error_msg = f"Reanalysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def _cache_result(self, query: str, result: Dict):
        """Cache a query result"""
        cache_key = self.get_cache_key(query)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

    def _get_cached_result(self, query: str) -> Optional[Dict]:
        """Get a cached result if available"""
        cache_key = self.get_cache_key(query)
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            # Check if cache is still valid (e.g., less than 1 hour old)
            if time.time() - cached_data['timestamp'] < 3600:  # 1 hour
                return cached_data['result']
        
        return None

    def _cache_prompt(self, prompt: str, response: str):
        """Cache a prompt and its response"""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        self.prompt_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

    def _get_cached_prompt(self, prompt: str) -> Optional[str]:
        """Get cached prompt response if available"""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cached_data = self.prompt_cache.get(cache_key)
        
        if cached_data:
            # Check if cache is still valid (24 hours)
            if time.time() - cached_data['timestamp'] < 86400:
                return cached_data['response']
        return None

    def process_query(self, query: str) -> Dict:
        """Process query with enhanced caching"""
        try:
            print("\n" + "="*50)
            print(f"Processing Query: {query}")
            print("="*50)
            
            # Check query cache first
            cache_key = hashlib.md5(query.encode()).hexdigest()
            cached_result = self.query_cache.get(cache_key)
            
            if cached_result:
                print("Using cached query result")
                # Update conversation memory
                self.memory.save_context(
                    {"input": query},
                    {"output": cached_result.get('analysis', '')}
                )
                return cached_result
            
            # Get conversation history
            chat_history = self.memory.load_memory_variables({})
            
            # Add chat history to decomposition context
            sub_queries = self.query_decomposer.decompose_query(
                query, 
                chat_history.get("chat_history", [])
            )
            
            # Process each sub-query
            results = []
            for sub_query in sub_queries:
                result = self._process_single_query(sub_query)
                results.append(result)
            
            # Combine results if needed
            if len(results) > 1:
                final_result = self._combine_results(results, query)
            else:
                final_result = results[0] if results else {"success": False, "error": "No results generated"}
            
            # Cache the final result
            self.query_cache[cache_key] = final_result
            
            # Update conversation memory
            self.memory.save_context(
                {"input": query},
                {"output": final_result.get('analysis', '')}
            )
            
            print("\n" + "="*50)
            print("Final Results:")
            print("="*50)
            print(self.format_output(final_result))
            
            return final_result
            
        except Exception as e:
            error_msg = f"Error during query processing: {str(e)}"
            print(f"\n❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "metrics": {},
                "entities": [],
                "sql_query": ""
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
                        # if col_def.hierarchy_level:
                        #     columns_metadata += f"  Hierarchy Level: {col_def.hierarchy_level}\n"
                        # if col_def.distinct_values:
                        #     columns_metadata += f"  Possible Values: {', '.join(str(v) for v in col_def.distinct_values)}\n"
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

    def _identify_column_for_value(self, value: str) -> Tuple[str, str, float]:
        """
        Identify the most appropriate column for a given value.
        Returns: (table_name, column_name, confidence_score)
        """
        best_column, score = self.vector_store.find_best_column_for_value(value)
        if best_column:
            table_name, column_name = best_column.split('.')
            return table_name, column_name, score
        return None, None, 0.0

    def _get_relevant_table_metadata(self, query: str) -> Tuple[str, Dict]:
        """
        Determine the most relevant table based on query and metadata.
        Returns: (table_name, table_metadata)
        """
        prompt = f"""Based on the following query and table metadata, determine the most relevant table to use:

Query: {query}

Available Tables:
{self.metadata.get_metadata_prompt()}

IMPORTANT: Return ONLY the exact table name, without any explanation. 
For example, just return: final_income_sheet_new_seq
"""
        response = self.llm.invoke([
            SystemMessage(content="You must return ONLY the table name, without any explanation or additional text."),
            HumanMessage(content=prompt)
        ])
        
        # Clean up the response to get just the table name
        selected_table = response.content.strip().split('\n')[0]  # Take first line only
        
        # Remove any common prefixes/suffixes that might appear
        selected_table = selected_table.replace('Table:', '').strip()
        
        # Validate that we got a known table name
        if selected_table not in self.metadata.tables:
            raise ValueError(f"Selected table '{selected_table}' not found in metadata")
        
        # Get the table metadata
        table_metadata = self.metadata.get_table_info(selected_table)
        if not table_metadata:
            raise ValueError(f"Invalid table selection: {selected_table}")
        
        print(f"\nSelected table '{selected_table}' based on query intent")
        return selected_table, table_metadata

    def _generate_sql_query(self, query_info: Dict) -> str:
        """Generate SQL with prompt caching"""
        query = query_info['query']
        entities = query_info['entities']
        
        # Get table metadata
        table_name, table_metadata = self._get_relevant_table_metadata(query)
        
        # Initialize sentence transformer for column matching
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create structured column metadata dynamically
        column_metadata = []
        hierarchy_levels = set()
        for col_name, col_info in table_metadata.columns.items():
            metadata = {
                "name": col_name,
                "description": col_info.description,
                "hierarchy_level": col_info.hierarchy_level,
                "distinct_values": col_info.distinct_values[:5] if col_info.distinct_values else None
            }
            column_metadata.append(metadata)
            if col_info.hierarchy_level:
                hierarchy_levels.add(col_info.hierarchy_level)
        
        # Match entities to columns
        column_matches = []
        for entity in entities:
            entity_embedding = embedding_model.encode(entity, convert_to_tensor=True)
            best_match = None
            best_score = -1
            
            for col_info in column_metadata:
                # Create column context for matching
                col_context = f"{col_info['name']} {col_info['description']}"
                col_embedding = embedding_model.encode(col_context, convert_to_tensor=True)
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(
                    entity_embedding.unsqueeze(0),
                    col_embedding.unsqueeze(0)
                ).item()
                
                # Check for exact matches in distinct values
                if col_info['distinct_values'] and entity.lower() in [str(v).lower() for v in col_info['distinct_values']]:
                    similarity = 1.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = col_info
            
            if best_match:
                column_matches.append({
                    "entity": entity,
                    "column": best_match['name'],
                    "confidence": best_score,
                    "hierarchy_level": best_match['hierarchy_level'],
                    "description": best_match['description']
                })
        
        # Create hierarchy information dynamically
        hierarchy_info = []
        for level in sorted(hierarchy_levels):
            columns_at_level = [col['name'] for col in column_metadata if col['hierarchy_level'] == level]
            if columns_at_level:
                hierarchy_info.append(f"Level {level}: {', '.join(columns_at_level)}")
        
        # Create the SQL generation prompt
        sql_prompt = f"""Generate a SQL query for: {query_info['query']}

Table: {table_name}

Available Columns and Their Meanings:
{json.dumps(column_metadata, indent=2)}

Entity to Column Mappings:
{json.dumps(column_matches, indent=2)}

Column Hierarchy Information:
{chr(10).join(hierarchy_info)}

Important Rules:
1. ONLY use columns from the provided metadata
2. Use exact column names as shown in the metadata
3. For dates, use 'Month' column with YYYY-MM-DD format
4. Consider hierarchy levels when grouping data
5. Use column descriptions to ensure semantic correctness
6. Include appropriate aggregations for numeric data
7. Consider entity-column mapping confidence scores

Generate a SQL query that accurately answers the question while following these rules.
"""

        # Check prompt cache
        cached_response = self._get_cached_prompt(sql_prompt)
        if cached_response:
            print("\nUsing cached SQL generation")
            return self._extract_sql(cached_response)

        # Add debugging information
        print("\nTable Selected:", table_name)
        print("\nEntity-Column Matches:")
        for match in column_matches:
            print(f"• Entity '{match['entity']}' → Column '{match['column']}' (confidence: {match['confidence']:.2f})")
        
        # Generate SQL with strong constraints
        response = self.llm.invoke([
            SystemMessage(content="""You are a precise SQL query generator that:
1. Only uses columns from the provided metadata
2. Considers column descriptions and hierarchy levels
3. Uses exact column names as specified
4. Generates valid SQL for financial analysis"""),
            HumanMessage(content=sql_prompt)
        ])
        
        # Cache the response
        self._cache_prompt(sql_prompt, response.content)
        
        sql = self._extract_sql(response.content)
        
        # Validate the generated SQL using dynamic column list
        valid_columns = {col['name'] for col in column_metadata}
        sql_tokens = set(word.strip('(),; ').lower() for word in sql.split())
        sql_tokens = {token for token in sql_tokens if not token.replace('.', '').isdigit()}
        
        # Get common SQL keywords dynamically from a configuration or generate from common patterns
        common_sql_keywords = {'select', 'from', 'where', 'group', 'by', 'having', 'order', 'limit', 
                             'and', 'or', 'as', 'sum', 'avg', 'count', 'min', 'max', 'distinct', 
                             'case', 'when', 'then', 'else', 'end', 'is', 'not', 'null', 'like'}
        
        invalid_columns = sql_tokens - {col.lower() for col in valid_columns} - common_sql_keywords
        
        if invalid_columns:
            print(f"\nWarning: Found potentially invalid columns: {invalid_columns}")
        
        print(f"\nGenerated SQL: {sql}")
        return sql

    def format_output(self, results: Dict) -> str:
        """Format the output for display"""
        try:
            output = []
            
            if not results.get("success", False):
                return f"❌ Error: {results.get('error', 'Unknown error occurred')}"
            
            # Add query for context
            output.append(f"🔍 **Query**: {results.get('query', '')}\n")
            
            # Format results
            if results.get("formatted_results"):
                output.append("📊 **Results**:")
                output.append("-" * 30)
                for row in results["formatted_results"]:
                    for key, value in row.items():
                        formatted_value = f"{value:,.2f}" if isinstance(value, (int, float)) else value
                        output.append(f"• **{key}**: {formatted_value}")
                output.append("")  # Add spacing
            
            # Format analysis
            if results.get("analysis"):
                output.append("📝 **Analysis**:")
                output.append("-" * 30)
                # Split analysis into paragraphs and format each properly
                analysis_parts = results["analysis"].split('\n')
                for part in analysis_parts:
                    if part.strip():
                        if part.strip().startswith(('1.', '2.', '3.')):
                            output.append(f"\n**{part.strip()}**")
                        elif part.strip().startswith('-'):
                            output.append(part.strip())
                        else:
                            output.append(part.strip())
                output.append("")
            
            # Format SQL query
            if results.get("sql_query"):
                output.append("💻 **SQL Query**:")
                output.append("-" * 30)
                output.append(f"```sql\n{results['sql_query']}\n```")
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Output formatting error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return f"Error formatting output: {str(e)}"

    def _analyze_results(self, query: str, results: List[Dict]) -> str:
        """Generate detailed analysis of the results"""
        try:
            analysis_prompt = f"""Analyze these results for the query: "{query}"

Results:
{json.dumps(results, indent=2)}

Please provide:
1. A clear interpretation of the numbers
2. Any notable insights or trends
3. Context about what these metrics mean

Keep the response clear and concise but informative."""

            print("\n🤔 Generating Analysis...")
            print(f"Analysis Input: {json.dumps(results, indent=2)}")
            
            response = self.llm.invoke([
                SystemMessage(content="You are a financial analyst providing clear, concise analysis of database query results."),
                HumanMessage(content=analysis_prompt)
            ])
            
            analysis = response.content if response else "No analysis generated."
            print(f"\nGenerated Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            error_msg = f"Analysis generation error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return f"Unable to generate analysis: {str(e)}"

class EntityStore:
    def __init__(self):
        self.entities = {}  # Simple storage for reference

    def add_entities_from_column(self, table_name: str, column_name: str, values: List[str]):
        """Store column values for reference"""
        key = f"{table_name}.{column_name}"
        self.entities[key] = [str(v).strip() for v in values if v and str(v).strip() != '-']

class QueryDecomposer:
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm

    def decompose_query(self, query: str, chat_history: List[Dict] = None) -> List[Dict]:
        """Decompose complex query with chat history context."""
        # Format chat history for context
        history_context = ""
        if chat_history:
            # Handle chat history as a list of messages
            if isinstance(chat_history, list):
                history_context = "\nPrevious conversation:\n"
                for msg in chat_history:
                    if isinstance(msg, dict):
                        # Handle dictionary format
                        history_context += f"User: {msg.get('input', '')}\n"
                        history_context += f"Assistant: {msg.get('output', '')}\n"
                    else:
                        # Handle message object format
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        role = msg.type if hasattr(msg, 'type') else 'message'
                        history_context += f"{role}: {content}\n"

        decomposition_prompt = f"""Analyze this query and break it down into simpler sub-queries if needed.
        If the query is already simple, return it as is.
        
        Query: {query}
        {history_context}
        
        Format your response as a JSON array of sub-queries. Example:
        [
            {{"sub_query": "What is X?", "explanation": "Analyzing X component"}},
            {{"sub_query": "What is Y?", "explanation": "Analyzing Y component"}}
        ]
        
        For simple queries, return a single-element array."""

        response = self.llm.invoke([HumanMessage(content=decomposition_prompt)])
        
        try:
            sub_queries_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            sub_queries_data = [{"sub_query": query, "explanation": "Direct query"}]
        
        result = []
        for sub_query_info in sub_queries_data:
            entities = self._extract_entities(sub_query_info['sub_query'])
            result.append({
                'type': 'direct',
                'query': sub_query_info['sub_query'],
                'entities': entities,
                'explanation': sub_query_info.get('explanation', 'Direct query')
            })
        
        return result

    def _extract_entities(self, query: str) -> List[str]:
        """Simply identify entities in the query."""
        entity_prompt = f"""Extract the key entities (important phrases, terms, or values) from this query.

        Query: {query}

        Format your response as a JSON array of entity strings. Example:
        ["total sales", "2023", "New York"]"""

        response = self.llm.invoke([HumanMessage(content=entity_prompt)])
        return json.loads(response.content)

def format_output(results: Dict) -> str:
    """Format the output for display"""
    output = []
    
    if not results.get("success", False):
        return f"❌ Error: {results.get('error', 'Unknown error occurred')}"
    
    output.append("🎯 Results:")
    output.append("-" * 30)
    
    if results.get("entities"):
        output.append("\n📋 Recognized Entities:")
        for entity in results["entities"]:
            output.append(f"• {entity['text']} ({entity['table']}.{entity['column']})")
            if 'match_type' in entity:
                output.append(f"  Match Type: {entity['match_type']}")
    
    if results.get("sql_query"):
        output.append("\n💻 Generated SQL:")
        output.append(results["sql_query"])
    
    if results.get("metrics"):
        output.append("\n📊 Metrics:")
        for key, value in results["metrics"].items():
            output.append(f"• {key}: {value}")
    
    if results.get("analysis"):
        output.append("\n📝 Analysis:")
        output.append(results["analysis"])
    
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

    # Chat interface
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # For assistant messages, use the formatted output
                if isinstance(message.get("content"), dict):
                    # If it's a result dictionary, format it
                    formatted_output = analyst.format_output(message["content"])
                    st.markdown(formatted_output)
                else:
                    # For regular messages
                    st.markdown(message["content"])

    # Query input
    if query := st.chat_input("Ask a question about your data"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query and get results
        with st.spinner("Analyzing..."):
            try:
                results = analyst.process_query(query)
                
                # Format the results using the format_output method
                formatted_output = analyst.format_output(results)
                
                # Add assistant message with both raw results and formatted output
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_output  # Store the formatted string directly
                })
                
                # Save chat
                chat_manager.save_chat(
                    st.session_state.current_chat_id,
                    st.session_state.messages
                )
                
                # Display the formatted output
                with st.chat_message("assistant"):
                    st.markdown(formatted_output)
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}"
                })

if __name__ == "__main__":
    main()