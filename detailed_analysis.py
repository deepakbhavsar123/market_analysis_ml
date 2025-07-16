#!/usr/bin/env python3

"""
Test script to run detailed market value change explanation
"""

from enhanced_ml_prediction import EnhancedForexMLPredictor
from datetime import datetime

def main():
    """
    Run the detailed market value change explanation
    """
    print("🔍 DETAILED MARKET VALUE CHANGE ANALYSIS")
    print("="*70)
    
    # Initialize enhanced predictor
    predictor = EnhancedForexMLPredictor()
    
    # Load data with enhanced features
    data = predictor.load_and_prepare_data()
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Run detailed market value explanation
    date1 = datetime(2012, 3, 6)
    date2 = datetime(2012, 3, 7)
    
    # Run the comprehensive analysis
    results = predictor.explain_market_value_changes(date1, date2)
    
    if results:
        print(f"\n" + "="*70)
        print(f"🎯 EXECUTIVE SUMMARY")
        print(f"="*70)
        print(f"Total Portfolio Impact: ${results['net_change']:,.2f}")
        print(f"Total Gains: ${results['total_gains']:,.2f}")
        print(f"Total Losses: ${results['total_losses']:,.2f}")
        
        if results['top_gains']:
            biggest_gain = max([x[1] for x in results['top_gains']])
            print(f"Biggest Single Gain: ${biggest_gain:,.2f}")
        
        if results['top_losses']:
            biggest_loss = min([x[1] for x in results['top_losses']])
            print(f"Biggest Single Loss: ${biggest_loss:,.2f}")
        
        print(f"\n💡 This analysis provides detailed explanations of:")
        print(f"  • Why specific deals gained or lost value")
        print(f"  • Which features (BPDelta, Duration, etc.) drove changes")
        print(f"  • Currency-specific performance patterns")
        print(f"  • Market conditions and risk factors")
        print(f"  • Deal size and maturity impact analysis")

if __name__ == "__main__":
    main()
