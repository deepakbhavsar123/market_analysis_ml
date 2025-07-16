#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
print("Loading data...")
data = pd.read_excel('data/Cashflows_FX_V3.xlsx')
print(f"✓ Loaded {len(data)} records")

# Filter for the dates we're analyzing
date1 = datetime(2012, 3, 6)
date2 = datetime(2012, 3, 7)

data_date1 = data[data['PositionDate'] == date1]
data_date2 = data[data['PositionDate'] == date2]

print(f"\nDate analysis:")
print(f"Date 1: {len(data_date1)} records")
print(f"Date 2: {len(data_date2)} records")

# Check forward deals specifically
existing_deals = set(data_date1['DealId'])
existing_deals_date2 = data_date2[data_date2['DealId'].isin(existing_deals)]
forward_deals = existing_deals_date2[
    existing_deals_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV'])
]

print(f"\nForward deals analysis:")
print(f"Forward deals on date2: {len(forward_deals)} records")

if len(forward_deals) > 0:
    print(f"FwdRate column range: {forward_deals['FwdRate'].min():.6f} to {forward_deals['FwdRate'].max():.6f}")
    print(f"BaseMV column range: ${forward_deals['BaseMV'].min():,.2f} to ${forward_deals['BaseMV'].max():,.2f}")
    print(f"Sample FwdRate values: {forward_deals['FwdRate'].head(3).values}")
    print(f"Sample BaseMV values: {forward_deals['BaseMV'].head(3).values}")
    
    # Check the actual rate changes between dates
    print(f"\nActual rate changes analysis:")
    sample_deal_id = forward_deals['DealId'].iloc[0]
    
    deal_date1 = data_date1[data_date1['DealId'] == sample_deal_id]
    deal_date2 = data_date2[data_date2['DealId'] == sample_deal_id]
    
    if len(deal_date1) > 0 and len(deal_date2) > 0:
        rate1 = deal_date1['FwdRate'].iloc[0]
        rate2 = deal_date2['FwdRate'].iloc[0]
        actual_rate_change = rate2 - rate1
        
        basemv1 = deal_date1['BaseMV'].sum()  # Net for pay/receive
        basemv2 = deal_date2['BaseMV'].sum()  # Net for pay/receive
        actual_mv_change = basemv2 - basemv1
        
        print(f"Sample Deal {sample_deal_id}:")
        print(f"  FwdRate change: {rate1:.6f} -> {rate2:.6f} (Δ {actual_rate_change:.6f})")
        print(f"  BaseMV change: ${basemv1:,.2f} -> ${basemv2:,.2f} (Δ ${actual_mv_change:,.2f})")
        
        if actual_rate_change != 0:
            implied_sensitivity = actual_mv_change / actual_rate_change
            print(f"  Implied rate sensitivity: ${implied_sensitivity:,.2f} per unit rate change")

    # Check all currencies
    print(f"\nCurrencies in forward deals: {forward_deals['Currency'].unique()}")
    
    # Check if we're using the right normalization
    print(f"\nNormalization check:")
    print(f"BaseMV / 100000 range: {(forward_deals['BaseMV'] / 100000).min():.2f} to {(forward_deals['BaseMV'] / 100000).max():.2f}")
    print(f"Rate - 1.30 range: {(forward_deals['FwdRate'] - 1.30).min():.6f} to {(forward_deals['FwdRate'] - 1.30).max():.6f}")

print("\nDebugging complete!")
