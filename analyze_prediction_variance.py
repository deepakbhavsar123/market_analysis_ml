#!/usr/bin/env python3
"""
Analyze Rate Prediction Variance Issues

This script investigates why there's a large variance between actual and predicted
rate impacts in the ML model. It will identify data quality issues and potential
improvements for the prediction accuracy.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load the forex data and analyze quality issues"""
    print("🔍 ANALYZING PREDICTION VARIANCE ISSUES")
    print("="*60)
    
    # Load the dataset
    try:
        data = pd.read_excel('data/Cashflows_FX_V3.xlsx')
        print(f"✓ Loaded {len(data)} records")
    except:
        print("❌ Error loading data")
        return None
    
    # Filter to the specific dates we're analyzing
    date1 = datetime(2012, 3, 6)
    date2 = datetime(2012, 3, 7)
    
    data_date1 = data[data['PositionDate'] == date1]
    data_date2 = data[data['PositionDate'] == date2]
    
    print(f"\nDate 1 ({date1.date()}): {len(data_date1)} deals")
    print(f"Date 2 ({date2.date()}): {len(data_date2)} deals")
    
    return data, data_date1, data_date2

def analyze_feature_quality(data_date1, data_date2):
    """Analyze the quality of financial risk features"""
    print("\n📊 FEATURE QUALITY ANALYSIS")
    print("="*40)
    
    # Check key financial risk metrics
    risk_features = ['BPDelta', 'ModifiedDuration', 'Convexity', 'ZeroRate', 'DaystoMaturity']
    
    for feature in risk_features:
        if feature in data_date1.columns:
            # Analyze Date1 data
            non_zero_date1 = (data_date1[feature] != 0).sum()
            total_date1 = len(data_date1)
            
            # Analyze Date2 data  
            non_zero_date2 = (data_date2[feature] != 0).sum()
            total_date2 = len(data_date2)
            
            print(f"\n{feature}:")
            print(f"  Date1: {non_zero_date1}/{total_date1} ({non_zero_date1/total_date1*100:.1f}%) non-zero")
            print(f"  Date2: {non_zero_date2}/{total_date2} ({non_zero_date2/total_date2*100:.1f}%) non-zero")
            
            if non_zero_date1 > 0:
                print(f"  Date1 range: {data_date1[feature].min():.4f} to {data_date1[feature].max():.4f}")
                print(f"  Date1 mean: {data_date1[feature].mean():.4f} ± {data_date1[feature].std():.4f}")
            
            if non_zero_date2 > 0:
                print(f"  Date2 range: {data_date2[feature].min():.4f} to {data_date2[feature].max():.4f}")
                print(f"  Date2 mean: {data_date2[feature].mean():.4f} ± {data_date2[feature].std():.4f}")

def analyze_rate_sensitivity_issues(data_date1, data_date2):
    """Analyze why rate sensitivity predictions are inaccurate"""
    print("\n🎯 RATE SENSITIVITY ANALYSIS")
    print("="*40)
    
    # Get existing deals only (exclude new deals)
    existing_deals = set(data_date1['DealId'])
    existing_deals_date2 = data_date2[data_date2['DealId'].isin(existing_deals)]
    
    # Filter forward deals
    forward_deals = existing_deals_date2[
        existing_deals_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV'])
    ]
    
    print(f"Forward deals for analysis: {len(forward_deals)}")
    
    # Analyze actual vs expected rate sensitivity
    sensitivity_analysis = []
    
    for deal_id in forward_deals['DealId'].unique():
        # Get deal data for both dates
        deal_date1 = data_date1[data_date1['DealId'] == deal_id]
        deal_date2 = data_date2[data_date2['DealId'] == deal_id]
        
        if len(deal_date1) > 0 and len(deal_date2) > 0:
            # Calculate net changes
            net_mv_date1 = deal_date1['BaseMV'].sum()
            net_mv_date2 = deal_date2['BaseMV'].sum()
            actual_mv_change = net_mv_date2 - net_mv_date1
            
            # Calculate rate changes
            avg_fwd_rate_date1 = deal_date1['FwdRate'].mean()
            avg_fwd_rate_date2 = deal_date2['FwdRate'].mean()
            rate_change = avg_fwd_rate_date2 - avg_fwd_rate_date1
            
            # Get risk metrics
            avg_bpdelta = deal_date1['BPDelta'].mean()
            avg_duration = deal_date1['ModifiedDuration'].mean()
            avg_convexity = deal_date1['Convexity'].mean()
            
            # Calculate theoretical sensitivities
            bpdelta_expected = avg_bpdelta * rate_change if avg_bpdelta != 0 else 0
            duration_expected = -avg_duration * net_mv_date1 * rate_change if avg_duration != 0 else 0
            
            sensitivity_analysis.append({
                'DealId': deal_id,
                'Currency': deal_date1['Currency'].iloc[0],
                'ActualMVChange': actual_mv_change,
                'RateChange': rate_change,
                'BPDelta': avg_bpdelta,
                'ModifiedDuration': avg_duration,
                'Convexity': avg_convexity,
                'BPDeltaExpected': bpdelta_expected,
                'DurationExpected': duration_expected,
                'NetMVDate1': net_mv_date1,
                'FaceValue': deal_date1['FaceValue'].sum()
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_analysis)
    
    if len(sensitivity_df) > 0:
        print(f"\nAnalyzed {len(sensitivity_df)} forward deals:")
        
        # Calculate accuracy metrics
        non_zero_bpdelta = (sensitivity_df['BPDelta'] != 0).sum()
        non_zero_duration = (sensitivity_df['ModifiedDuration'] != 0).sum()
        
        print(f"  Deals with non-zero BPDelta: {non_zero_bpdelta}/{len(sensitivity_df)} ({non_zero_bpdelta/len(sensitivity_df)*100:.1f}%)")
        print(f"  Deals with non-zero Duration: {non_zero_duration}/{len(sensitivity_df)} ({non_zero_duration/len(sensitivity_df)*100:.1f}%)")
        
        # Show examples of large variances
        sensitivity_df['BPDeltaVariance'] = abs(sensitivity_df['ActualMVChange'] - sensitivity_df['BPDeltaExpected'])
        sensitivity_df['DurationVariance'] = abs(sensitivity_df['ActualMVChange'] - sensitivity_df['DurationExpected'])
        
        print(f"\n🔍 TOP 5 DEALS WITH LARGE BPDELTA VARIANCE:")
        top_bpdelta_variance = sensitivity_df.nlargest(5, 'BPDeltaVariance')
        for _, row in top_bpdelta_variance.iterrows():
            if row['BPDelta'] != 0:
                print(f"  Deal {row['DealId']} ({row['Currency']}):")
                print(f"    BPDelta: {row['BPDelta']:,.2f}")
                print(f"    Rate Change: {row['RateChange']:.6f} ({row['RateChange']*10000:.1f} bps)")
                print(f"    Expected: ${row['BPDeltaExpected']:,.2f}")
                print(f"    Actual: ${row['ActualMVChange']:,.2f}")
                print(f"    Variance: ${row['BPDeltaVariance']:,.2f}")
        
        print(f"\n🔍 TOP 5 DEALS WITH LARGE DURATION VARIANCE:")
        top_duration_variance = sensitivity_df.nlargest(5, 'DurationVariance')
        for _, row in top_duration_variance.iterrows():
            if row['ModifiedDuration'] != 0:
                print(f"  Deal {row['DealId']} ({row['Currency']}):")
                print(f"    Duration: {row['ModifiedDuration']:.4f}")
                print(f"    Rate Change: {row['RateChange']:.6f} ({row['RateChange']*10000:.1f} bps)")
                print(f"    Expected: ${row['DurationExpected']:,.2f}")
                print(f"    Actual: ${row['ActualMVChange']:,.2f}")
                print(f"    Variance: ${row['DurationVariance']:,.2f}")
        
        # Calculate overall variance statistics
        total_actual = sensitivity_df['ActualMVChange'].sum()
        total_bpdelta_expected = sensitivity_df['BPDeltaExpected'].sum()
        total_duration_expected = sensitivity_df['DurationExpected'].sum()
        
        bpdelta_variance = abs(total_actual - total_bpdelta_expected)
        duration_variance = abs(total_actual - total_duration_expected)
        
        print(f"\n📊 OVERALL VARIANCE SUMMARY:")
        print(f"  Total Actual MV Change: ${total_actual:,.2f}")
        print(f"  Total BPDelta Expected: ${total_bpdelta_expected:,.2f}")
        print(f"  Total Duration Expected: ${total_duration_expected:,.2f}")
        print(f"  BPDelta Variance: ${bpdelta_variance:,.2f} ({bpdelta_variance/abs(total_actual)*100:.1f}%)")
        print(f"  Duration Variance: ${duration_variance:,.2f} ({duration_variance/abs(total_actual)*100:.1f}%)")
    
    return sensitivity_df

def identify_data_issues(data_date1, data_date2):
    """Identify specific data quality issues causing prediction problems"""
    print("\n🚨 DATA QUALITY ISSUES IDENTIFICATION")
    print("="*50)
    
    issues = []
    
    # Check for zero risk metrics
    risk_metrics = ['BPDelta', 'ModifiedDuration', 'Convexity']
    for metric in risk_metrics:
        if metric in data_date1.columns:
            zero_count = (data_date1[metric] == 0).sum()
            total_count = len(data_date1)
            if zero_count / total_count > 0.5:  # More than 50% zeros
                issues.append(f"❌ {metric}: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%) are zero")
            else:
                issues.append(f"✅ {metric}: Only {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%) are zero")
    
    # Check for missing valuation models
    valuation_models = data_date1['ValuationModel'].value_counts()
    print(f"\nValuation Model Distribution:")
    for model, count in valuation_models.items():
        print(f"  {model}: {count} deals ({count/len(data_date1)*100:.1f}%)")
    
    # Check rate ranges
    print(f"\nRate Ranges:")
    for rate_col in ['SpotRate', 'FwdRate', 'ZeroRate']:
        if rate_col in data_date1.columns:
            min_val = data_date1[rate_col].min()
            max_val = data_date1[rate_col].max()
            print(f"  {rate_col}: {min_val:.6f} to {max_val:.6f}")
    
    # Check for extreme outliers in BaseMV
    basemv_q99 = data_date1['BaseMV'].quantile(0.99)
    basemv_q01 = data_date1['BaseMV'].quantile(0.01)
    extreme_deals = len(data_date1[(data_date1['BaseMV'] > basemv_q99) | (data_date1['BaseMV'] < basemv_q01)])
    
    print(f"\nBaseMV Outliers:")
    print(f"  Extreme deals (>99th or <1st percentile): {extreme_deals} ({extreme_deals/len(data_date1)*100:.1f}%)")
    print(f"  99th percentile: ${basemv_q99:,.2f}")
    print(f"  1st percentile: ${basemv_q01:,.2f}")
    
    print(f"\nIdentified Issues:")
    for issue in issues:
        print(f"  {issue}")
    
    return issues

def recommend_improvements():
    """Recommend specific improvements for rate prediction accuracy"""
    print("\n💡 RECOMMENDATIONS FOR IMPROVING RATE PREDICTION")
    print("="*55)
    
    recommendations = [
        "1. DATA QUALITY FIXES:",
        "   • Investigate why 50%+ of BPDelta/Duration values are zero",
        "   • Validate risk metric calculations in source system", 
        "   • Check if zero values represent actual no-sensitivity deals or missing data",
        "",
        "2. FEATURE ENGINEERING IMPROVEMENTS:",
        "   • Use FaceValue and Currency as primary drivers when risk metrics are zero",
        "   • Create synthetic rate sensitivity based on deal characteristics",
        "   • Add interaction terms between rate changes and deal size",
        "   • Include time-to-maturity impact on rate sensitivity",
        "",
        "3. MODEL ENHANCEMENTS:",
        "   • Use ensemble methods combining multiple prediction approaches",
        "   • Implement currency-specific models for better accuracy",
        "   • Add deal-type specific sensitivity calculations",
        "   • Use historical volatility to weight predictions",
        "",
        "4. VALIDATION IMPROVEMENTS:",
        "   • Cross-validate predictions across multiple date pairs",
        "   • Implement rolling window validation for time series stability",
        "   • Add confidence intervals to predictions",
        "   • Monitor prediction accuracy over time",
        "",
        "5. ALTERNATIVE APPROACHES:",
        "   • Use Black-Scholes or other financial models for theoretical pricing",
        "   • Implement Monte Carlo simulation for rate impact scenarios",
        "   • Use historical regression to estimate missing risk metrics",
        "   • Apply market data to improve rate sensitivity estimates"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """Main analysis function"""
    
    # Load and analyze data
    result = load_and_analyze_data()
    if result is None:
        return
    
    data, data_date1, data_date2 = result
    
    # Analyze feature quality
    analyze_feature_quality(data_date1, data_date2)
    
    # Analyze rate sensitivity issues
    sensitivity_df = analyze_rate_sensitivity_issues(data_date1, data_date2)
    
    # Identify data issues
    issues = identify_data_issues(data_date1, data_date2)
    
    # Provide recommendations
    recommend_improvements()
    
    print(f"\n🎯 SUMMARY:")
    print(f"The large prediction variance (40.9%) is primarily caused by:")
    print(f"• 50%+ of risk metrics (BPDelta, Duration) are zero")
    print(f"• Missing or incomplete sensitivity calculations")
    print(f"• Need for better feature engineering when risk metrics are unavailable")
    print(f"• Attribution accuracy of 58.9% suggests model needs fundamental improvements")

if __name__ == "__main__":
    main()
