#!/usr/bin/env python3
"""
Test script to verify the forex analysis functions work correctly
"""

import sys
import os
sys.path.append('/home/deepak/ml/market_analysis/streamlit_dashboard')

import pandas as pd
from datetime import datetime

def test_data_loading():
    """Test if we can load the data successfully"""
    print("Testing data loading...")
    
    # Try different possible paths for the data file
    possible_paths = [
        '../data/Cashflows_FX_V3.csv',
        './data/Cashflows_FX_V3.csv',
        'data/Cashflows_FX_V3.csv',
        '/home/deepak/ml/market_analysis/data/Cashflows_FX_V3.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                data['PositionDate'] = pd.to_datetime(data['PositionDate'])
                print(f"✅ Data loaded successfully from: {path}")
                print(f"   Records: {len(data)}")
                print(f"   Date range: {data['PositionDate'].min()} to {data['PositionDate'].max()}")
                return data
            except Exception as e:
                print(f"❌ Error reading {path}: {str(e)}")
                continue
    
    print("❌ Error: Cashflows_FX_V3.csv not found in any expected location")
    return None

def test_timeline_chart_data(data):
    """Test the timeline chart data preparation"""
    print("\nTesting timeline chart data preparation...")
    
    try:
        # Group by date and sum BaseMV for each date
        daily_basemv = data.groupby('PositionDate')['BaseMV'].sum().reset_index()
        daily_basemv = daily_basemv.sort_values('PositionDate')
        
        print(f"✅ Timeline data prepared successfully")
        print(f"   Unique dates: {len(daily_basemv)}")
        print(f"   Sample dates: {daily_basemv['PositionDate'].head(3).tolist()}")
        
        # Test ordinal conversion
        daily_basemv_copy = daily_basemv.copy()
        daily_basemv_copy['date_ordinal'] = daily_basemv_copy['PositionDate'].map(lambda x: x.toordinal())
        print(f"✅ Ordinal conversion successful")
        print(f"   Sample ordinals: {daily_basemv_copy['date_ordinal'].head(3).tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Error in timeline data preparation: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Forex Analysis Functions")
    print("=" * 50)
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("❌ Cannot proceed without data")
        return
    
    # Test timeline chart data preparation
    if test_timeline_chart_data(data):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

if __name__ == "__main__":
    main()
