# Comprehensive Guide to Financial Sheets Structure and Analysis

## 1. Balance Sheet Structure

### 1.1 Column Structure
```sql
- id                               // Unique identifier
- account_type                     // Type of account classification
- SQL_Account_Name_Code            // Unique account code
- SQL_Account_Name                 // Account name
- SQL_Account_Category_Order_Code  // Category code for ordering
- SQL_Account_Category_Order       // Category order sequence
- Sub_Account_Category_Order_Code  // Sub-category code
- Sub_Account_Category_Order       // Sub-category order sequence
- SQL_Account_Group_Name_Code      // Group code
- SQL_Account_Group_Name           // Group name
- SQL_Sub_Account_Group_Name_Code  // Sub-group code
- SQL_Sub_Account_Group_Name       // Sub-group name
- DC_BS_Account_Name              // Display name for balance sheet account
- Amount                          // Current period amount
- Total_Amount                    // Cumulative total
- Prior                          // Previous period amount
- Operator                       // System operator identifier
- SQL_BS_Account_ID              // Balance sheet account ID
- SQL_Property                   // Property identifier
- Month                         // Reporting month
- updated_at                    // Last update timestamp
```

### 1.2 Purpose and Usage
- **Primary Function**: Provides point-in-time financial position
- **Key Categories**:
  * Assets (Current and Fixed)
  * Liabilities (Current and Long-term)
  * Owner's Equity

### 1.3 Analysis Capabilities
1. Financial Position Analysis
   - Asset composition
   - Liability structure
   - Equity position

2. Financial Ratios
   - Liquidity ratios
   - Solvency metrics
   - Asset utilization

## 2. Income Statement Structure

### 2.1 Column Structure
```sql
- id                               // Unique identifier
- account_type                     // Type of account
- heading_sequence                 // Header ordering
- sequence                        // Item sequence
- SQL_Account_Name_Code           // Account code
- SQL_Account_Name                // Account name
- SQL_Account_Category_Order_Code // Category code
- SQL_Account_Category_Order      // Category sequence
- Sub_Account_Category_Order_Code // Sub-category code
- Sub_Account_Category_Order      // Sub-category sequence
- SQL_Account_Group_Name_Code     // Group code
- SQL_Account_Group_Name          // Group name
- DC_IS_Account_Name             // Display name
- DC_IS_Account_Category         // Account category
- Current_Actual_Month           // Current month actual
- Budget                         // Budgeted amount
- Variance_to_Budget             // Budget variance
- Last_Year_Actual_Month        // Prior year month
- YoY_Change                    // Year-over-year change
- YTD_Actual                    // Year-to-date actual
- YTD_Budget                    // Year-to-date budget
- Last_Year_Actual_YTD          // Prior year YTD
- Operator                      // System operator
- SQL_IS_Account_ID             // Income statement ID
- SQL_Property                  // Property identifier
- Month                         // Reporting month
- updated_at                    // Last update timestamp
```

### 2.2 Purpose and Usage
- **Primary Function**: Tracks revenue, expenses, and profitability
- **Key Categories**:
  * Operating Revenue
  * Department Expenses
  * Undistributed Expenses
  * Operating Profit/Loss

### 2.3 Analysis Capabilities
1. Performance Analysis
   - Revenue trends
   - Expense management
   - Profit margins

2. Comparative Analysis
   - Budget variances
   - Year-over-year comparison
   - YTD performance

## 3. Final Forecast Sheet Structure

### 3.1 Column Structure
```sql
- id                               // Unique identifier
- DC_FC_Assets_Type               // Asset type classification
- DC_FC_Assets_Name              // Asset name
- SQL_FC_Account_ID              // Forecast account ID
- SQL_Heading_Sequence           // Header sequence
- SQL_Sequence                   // Item sequence
- SQL_Account_Name_Code          // Account code
- SQL_Account_Name               // Account name
- SQL_Account_Category_Order_Code // Category code
- SQL_Account_Category_Order     // Category sequence
- SUB_Account_Category_Order_Code // Sub-category code
- SUB_Account_Category_Order     // Sub-category sequence
- SQL_Account_Group_Name_Code    // Group code
- SQL_Account_Group_Name         // Group name
- Accountnumber_ID              // Account number
- January through December      // Monthly forecast values
- Total                        // Annual total
- Account_Year                 // Fiscal year
- SQL_Property                 // Property identifier
- updated_at                   // Last update timestamp
```

### 3.2 Purpose and Usage
- **Primary Function**: Projects future financial performance
- **Key Components**:
  * Operational Metrics (Occupancy, ADR, RevPAR)
  * Revenue Projections
  * Expense Forecasts
  * Profit Forecasts

### 3.3 Analysis Capabilities
1. Forward-Looking Analysis
   - Trend projections
   - Seasonal patterns
   - Growth forecasting

2. Operational Planning
   - Resource allocation
   - Revenue management
   - Expense planning

## 4. Cross-Sheet Integration

### 4.1 Common Elements
1. Account Coding Structure
   - SQL_Account_Name_Code
   - SQL_Account_Category_Order
   - SQL_Account_Group_Name

2. Property Tracking
   - SQL_Property
   - Month/Period tracking
   - Update history

### 4.2 Key Relationships
1. Balance Sheet to Income Statement
   - Revenue impact on assets
   - Expense impact on liabilities
   - Profit impact on equity

2. Forecast to Actuals
   - Budget variance tracking
   - Performance monitoring
   - Forecast adjustments



TABLE_METADATA = {
    "balance_sheet": {
        "description": "Contains point-in-time financial position data showing assets, liabilities, and equity",
        "key_purposes": [
            "Track financial position",
            "Monitor asset and liability balances",
            "Calculate financial ratios"
        ],
        "common_queries": [
            "Current assets total",
            "Liability breakdown",
            "Equity position"
        ],
        "relationships": {
            "income_statement": "Affects through profit/loss",
            "forecast_sheet": "Provides actual vs projected positions"
        }
    },
    "income_statement": {
        "description": "Tracks revenue, expenses, and profitability over a period",
        "key_purposes": [
            "Monitor revenue and expenses",
            "Track profitability",
            "Compare actual vs budget"
        ],
        "common_queries": [
            "Revenue by type",
            "Expense analysis",
            "Profit margins"
        ],
        "relationships": {
            "balance_sheet": "Impacts through P&L",
            "forecast_sheet": "Actual vs forecast comparison"
        }
    },
    "forecast_sheet": {
        "description": "Contains future projections of financial performance",
        "key_purposes": [
            "Project future performance",
            "Plan resources",
            "Set targets"
        ],
        "common_queries": [
            "Revenue projections",
            "Expense forecasts",
            "Occupancy forecasts"
        ],
        "relationships": {
            "balance_sheet": "Projects future positions",
            "income_statement": "Projects future P&L"
        }
    }
}

COLUMN_CONTEXT = {
    "balance_sheet": {
        "Amount": {
            "description": "Current period amount",
            "common_uses": ["Current position", "Ratio calculations"],
            "related_columns": ["Total_Amount", "Prior"]
        },
        "SQL_Account_Category_Order": {
            "description": "Hierarchical category organization",
            "common_uses": ["Grouping", "Reporting structure"],
            "related_columns": ["SQL_Account_Group_Name"]
        }
        # ... other columns
    },
    "income_statement": {
        "Current_Actual_Month": {
            "description": "Current month's actual figures",
            "common_uses": ["Performance tracking", "Variance analysis"],
            "related_columns": ["Budget", "Last_Year_Actual_Month"]
        }
        # ... other columns
    }
    # ... other tables
}