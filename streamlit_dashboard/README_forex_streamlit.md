# Forex Deals Analysis Streamlit Dashboard

A comprehensive Streamlit dashboard for analyzing forex deals data between March 6-7, 2012.

## Features

- **BaseMV Timeline**: Interactive line chart showing BaseMV progression across all dates in the dataset
- **Visual Analysis**: Interactive charts showing BaseMV comparisons and deal breakdowns
- **New Deals Analysis**: Identification and analysis of new deals with BaseMV impact
- **Existing Deals Analysis**: Analysis of changes in existing deals
- **Rate Statistics**: Forward and spot rate analysis by currency
- **Top Deals**: Top 3 deals by impact for both new and existing deals
- **Export Functionality**: Download analysis results as JSON or summary reports
- **Raw Data Exploration**: View sample data from both analysis dates
- **Interactive Controls**: Sidebar options to customize timeline display

## Installation

1. Navigate to the streamlit_dashboard directory:
```bash
cd /home/deepak/ml/market_analysis/streamlit_dashboard
```

2. Install requirements:
```bash
pip install -r requirements_forex_streamlit.txt
```

## Usage

### Option 1: Using the run script
```bash
./run_forex_streamlit.sh
```

### Option 2: Direct Streamlit command
```bash
streamlit run forex_deals_streamlit_app.py --server.port=8501
```

## Data Requirements

The application expects the forex data file `Cashflows_FX_V3.csv` to be available in one of these locations:
- `../data/Cashflows_FX_V3.csv`
- `./data/Cashflows_FX_V3.csv` 
- `data/Cashflows_FX_V3.csv`
- `/home/deepak/ml/market_analysis/data/Cashflows_FX_V3.csv`

## Dashboard Sections

### 1. BaseMV Timeline (All Dates)
- **Complete History**: Line chart showing BaseMV progression across all dates in the dataset
- **Analysis Highlights**: Red star markers indicating March 6-7, 2012 analysis dates
- **Trend Analysis**: Optional trend line with R² correlation coefficient
- **Interactive Controls**: Sidebar toggles for trend line and analysis markers
- **Statistics**: Key metrics including date range, highest/lowest BaseMV values

### 2. March 6-7, 2012 Analysis
- **BaseMV Comparison**: Bar chart comparing total BaseMV between March 6 and 7, 2012
- **Deals Breakdown**: Side-by-side comparison of deal counts and BaseMV impacts

### 3. Top Deal Analysis
- **New Deals Tab**: Analysis of newly introduced deals and their BaseMV contributions
- **Existing Deals Tab**: Analysis of changes in existing deals

### 4. Rate Statistics
- **Forward Rates Tab**: Analysis of forward exchange rates by currency
- **Spot Rates Tab**: Analysis of spot exchange rates by currency

### 5. Raw Data Exploration
- Expandable section showing sample data from both analysis dates

### 6. Export Results
- Download complete analysis results as JSON
- Download summary report as text file

## Technical Details

- **Framework**: Streamlit 1.28.0+
- **Visualization**: Plotly 5.15.0+
- **Data Processing**: Pandas 1.5.0+
- **Caching**: Uses Streamlit's `@st.cache_data` for efficient data loading

## Error Handling

The application includes robust error handling for:
- Missing data files
- Data loading errors
- Empty datasets
- Chart rendering issues

## Notes

- The Summary Metrics section has been removed as requested
- Fixed AttributeError with Plotly Figure annotations
- Supports multiple data file locations for flexibility
- Responsive layout that works on different screen sizes
