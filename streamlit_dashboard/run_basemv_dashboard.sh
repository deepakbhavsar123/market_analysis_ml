#!/bin/bash

# BaseMV Streamlit Dashboard Launcher
echo "🚀 Starting BaseMV Streamlit Dashboard..."
echo "=================================="

# Check if data file exists
if [ ! -f "../data/Cashflows_FX_V3.csv" ]; then
    echo "❌ Error: Data file '../data/Cashflows_FX_V3.csv' not found!"
    echo "   Please ensure the data file exists in the parent directory's data folder."
    exit 1
fi

# Install requirements if needed
echo "📦 Installing dashboard requirements..."
pip install -r requirements_dashboard.txt

# Launch the Streamlit dashboard
echo "🌐 Starting Streamlit dashboard..."
echo "🔗 Dashboard will be available at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the dashboard"
echo ""

streamlit run basemv_streamlit_dashboard.py --server.port 8501 --server.address localhost
