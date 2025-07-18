"""
Forex Deals Analysis for March 6-7, 2012
Analyzes forex deals data to extract new deals and rate statistics
"""

import pandas as pd
import json
from datetime import datetime
import requests


def load_and_analyze_forex_data():
    """
    Load forex data and perform comprehensive analysis for March 6-7, 2012
    """
    print("📊 Loading Forex Deals Data...")
    
    # Load the CSV data
    data = pd.read_csv('data/Cashflows_FX_V3.csv')
    
    # Convert PositionDate to datetime
    data['PositionDate'] = pd.to_datetime(data['PositionDate'])
    
    # Define target dates
    date1 = datetime(2012, 3, 6)
    date2 = datetime(2012, 3, 7)
    
    print(f"✓ Loaded {len(data)} total records")
    print(f"✓ Analyzing dates: {date1.date()} and {date2.date()}")
    
    # Filter data for the two dates
    data_date1 = data[data['PositionDate'] == date1]
    data_date2 = data[data['PositionDate'] == date2]
    
    print(f"✓ March 6, 2012: {len(data_date1)} records")
    print(f"✓ March 7, 2012: {len(data_date2)} records")
    
    # Initialize results dictionary
    results = {
        'target_dates': {
            'date1': date1.date().isoformat(),
            'date2': date2.date().isoformat()
        },
        'summary': {},
        'new_deals_analysis': {},
        'rate_statistics': {}
    }
    
    # === NEW DEALS ANALYSIS ===
    print("\n🔍 Analyzing New Deals...")
    
    # Get deal IDs for each date
    deals_date1 = set(data_date1['DealId'].unique())
    deals_date2 = set(data_date2['DealId'].unique())
    # Identify new deals (present on date2 but not on date1)
    new_deals = deals_date2 - deals_date1
    # Identify existing deals (present on both dates)
    existing_deals = deals_date1.intersection(deals_date2)

    
    # Get new deals data
    new_deals_data = data_date2[data_date2['DealId'].isin(new_deals)]
    
    # Calculate BaseMV sum for new deals
    new_deals_basemv_sum = new_deals_data['BaseMV'].sum()
    
    print(f"✓ New deals identified: {len(new_deals)} deals")
    print(f"✓ New deals total records (pay/receive pairs): {len(new_deals_data)}")
    print(f"✓ New deals BaseMV sum: ${new_deals_basemv_sum:,.2f}")
    
    # Get top 3 new deals by BaseMV contribution
    deal_basemv_contributions = []
    for deal_id in new_deals:
        deal_records = new_deals_data[new_deals_data['DealId'] == deal_id]
        deal_basemv_sum = deal_records['BaseMV'].sum()
        deal_basemv_contributions.append({
            'deal_id': deal_id,
            'basemv_contribution': float(deal_basemv_sum),
            'record_count': len(deal_records),
            'currencies': list(deal_records['Currency'].unique())
        })
    
    # Sort by absolute BaseMV contribution (highest impact) and get top 3
    top_3_deals = sorted(deal_basemv_contributions, key=lambda x: abs(x['basemv_contribution']), reverse=True)[:3]
    
    print(f"\n🔝 Top 3 New Deals by BaseMV Impact:")
    for i, deal in enumerate(top_3_deals, 1):
        print(f"  {i}. Deal {deal['deal_id']}: ${deal['basemv_contribution']:,.2f} ({deal['currencies']})")
    
    # Store new deals analysis
    results['new_deals_analysis'] = {
        'total_new_deals': len(new_deals),
        'total_new_records': len(new_deals_data),
        'basemv_sum': float(new_deals_basemv_sum),
        'new_deal_ids': list(new_deals),
        'top_3_deals_by_basemv': top_3_deals
    }
    
    # === EXISTING DEALS ANALYSIS ===
    print(f"\n🔄 Analyzing Existing Deals Changes...")
    print(f"✓ Existing deals (present on both dates): {len(existing_deals)} deals")
    
    # Get top 3 existing deals by BaseMV change
    existing_deal_changes = []
    for deal_id in existing_deals:
        deal_records_date1 = data_date1[data_date1['DealId'] == deal_id]
        deal_records_date2 = data_date2[data_date2['DealId'] == deal_id]
        
        basemv_date1 = deal_records_date1['BaseMV'].sum()
        basemv_date2 = deal_records_date2['BaseMV'].sum()
        basemv_change = basemv_date2 - basemv_date1
        
        existing_deal_changes.append({
            'deal_id': deal_id,
            'basemv_date1': float(basemv_date1),
            'basemv_date2': float(basemv_date2),
            'basemv_change': float(basemv_change),
            'record_count_date1': len(deal_records_date1),
            'record_count_date2': len(deal_records_date2),
            'currencies': list(set(deal_records_date1['Currency'].unique()) | set(deal_records_date2['Currency'].unique()))
        })
    
    # Sort by absolute BaseMV change (highest impact) and get top 3
    top_3_existing_deals = sorted(existing_deal_changes, key=lambda x: abs(x['basemv_change']), reverse=True)[:3]
    
    # Calculate total BaseMV change for existing deals
    total_existing_basemv_change = sum(deal['basemv_change'] for deal in existing_deal_changes)
    
    print(f"✓ Total existing deals BaseMV change: ${total_existing_basemv_change:,.2f}")
    print(f"\n🔝 Top 3 Existing Deals by BaseMV Change:")
    for i, deal in enumerate(top_3_existing_deals, 1):
        print(f"  {i}. Deal {deal['deal_id']}: ${deal['basemv_change']:,.2f} (${deal['basemv_date1']:,.2f} → ${deal['basemv_date2']:,.2f}) ({deal['currencies']})")
    
    # Store existing deals analysis
    results['existing_deals_analysis'] = {
        'total_existing_deals': len(existing_deals),
        'total_basemv_change': float(total_existing_basemv_change),
        'top_3_deals_by_basemv_change': top_3_existing_deals
    }
    
    # === RATE STATISTICS ANALYSIS ===
    print("\n📈 Analyzing Rate Statistics...")
    
    # Define valuation models to analyze
    forward_valuation_models = ['FORWARD', 'FORWARD NPV']
    spot_valuation_models = ['NPV SPOT']
    
    # Analyze both dates
    for date_key, date_value, date_data in [
        ('date1', date1, data_date1),
        ('date2', date2, data_date2)
    ]:
        print(f"\n📊 Analyzing {date_value.date()}...")
        
        # Filter for forward deals
        forward_deals = date_data[date_data['ValuationModel'].isin(forward_valuation_models)]
        # Filter for spot deals
        spot_deals = date_data[date_data['ValuationModel'].isin(spot_valuation_models)]
        
        print(f"  Forward deals: {len(forward_deals)} records")
        print(f"  Spot NPV deals: {len(spot_deals)} records")
        
        # Calculate currency stats for forward deals
        forward_currency_stats = {}
        for currency in forward_deals['Currency'].unique():
            currency_data = forward_deals[forward_deals['Currency'] == currency]
            
            # Calculate rate statistics - only forward rates for forward deals
            fwd_rates = currency_data['FwdRate'].dropna()
            
            forward_currency_stats[currency] = {
                'record_count': len(currency_data),
                'fwd_rate': {
                    'min': float(fwd_rates.min()) if len(fwd_rates) > 0 else None,
                    'max': float(fwd_rates.max()) if len(fwd_rates) > 0 else None,
                    'count': len(fwd_rates)
                }
            }
            
            print(f"    Forward {currency}: {len(currency_data)} records")
            if len(fwd_rates) > 0:
                print(f"      FwdRate: {fwd_rates.min():.6f} - {fwd_rates.max():.6f}")
        
        # Calculate currency stats for spot NPV deals
        spot_currency_stats = {}
        for currency in spot_deals['Currency'].unique():
            currency_data = spot_deals[spot_deals['Currency'] == currency]
            
            # Calculate rate statistics - only spot rates for spot NPV deals
            spot_rates = currency_data['SpotRate'].dropna()
            
            spot_currency_stats[currency] = {
                'record_count': len(currency_data),
                'spot_rate': {
                    'min': float(spot_rates.min()) if len(spot_rates) > 0 else None,
                    'max': float(spot_rates.max()) if len(spot_rates) > 0 else None,
                    'count': len(spot_rates)
                }
            }
            
            print(f"    Spot NPV {currency}: {len(currency_data)} records")
            if len(spot_rates) > 0:
                print(f"      SpotRate: {spot_rates.min():.6f} - {spot_rates.max():.6f}")
        
        # Store results
        results['rate_statistics'][date_key] = {
            'date': date_value.date().isoformat(),
            'total_forward_records': len(forward_deals),
            'total_spot_npv_records': len(spot_deals),
            'currencies_analyzed_forward': len(forward_currency_stats),
            'currencies_analyzed_spot_npv': len(spot_currency_stats),
            'forward_currency_stats': forward_currency_stats,
            'spot_npv_currency_stats': spot_currency_stats
        }
    
    # === SUMMARY STATISTICS ===
    # Calculate BaseMV totals for each date
    basemv_date1 = data_date1['BaseMV'].sum()
    basemv_date2 = data_date2['BaseMV'].sum()
    basemv_difference = basemv_date2 - basemv_date1
    
    results['summary'] = {
        'total_records_date1': len(data_date1),
        'total_records_date2': len(data_date2),
        'total_unique_deals_date1': len(deals_date1),
        'total_unique_deals_date2': len(deals_date2),
        'existing_deals_count': len(existing_deals),
        'new_deals_count': len(new_deals),
        'new_deals_basemv_sum': float(new_deals_basemv_sum),
        'existing_deals_basemv_change': float(total_existing_basemv_change),
        'basemv_totals': {
            'date1_total': float(basemv_date1),
            'date2_total': float(basemv_date2),
            'difference': float(basemv_difference)
        },
        'currencies_analyzed': {
            'date1_forward': len(results['rate_statistics']['date1']['forward_currency_stats']),
            'date2_forward': len(results['rate_statistics']['date2']['forward_currency_stats']),
            'date1_spot_npv': len(results['rate_statistics']['date1']['spot_npv_currency_stats']),
            'date2_spot_npv': len(results['rate_statistics']['date2']['spot_npv_currency_stats'])
        }
    }
    
    return results

def save_results_to_json(results, filename='forex_deals_analysis_results.json'):
    """
    Save analysis results to JSON file
    """
    print(f"\n💾 Saving results to {filename}...")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved successfully!")
    
    # Print summary
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"="*50)
    print(f"• Existing deals count: {results['summary']['existing_deals_count']}")
    print(f"• Existing deals BaseMV change: ${results['summary']['existing_deals_basemv_change']:,.2f}")
    print(f"• New deals identified: {results['summary']['new_deals_count']}")
    print(f"• New deals BaseMV sum: ${results['summary']['new_deals_basemv_sum']:,.2f}")
    print(f"• BaseMV Date 1 total: ${results['summary']['basemv_totals']['date1_total']:,.2f}")
    print(f"• BaseMV Date 2 total: ${results['summary']['basemv_totals']['date2_total']:,.2f}")
    print(f"• BaseMV Difference (Date2 - Date1): ${results['summary']['basemv_totals']['difference']:,.2f}")
    print(f"• Forward currencies analyzed (Date 1): {results['summary']['currencies_analyzed']['date1_forward']}")
    print(f"• Forward currencies analyzed (Date 2): {results['summary']['currencies_analyzed']['date2_forward']}")
    print(f"• Spot NPV currencies analyzed (Date 1): {results['summary']['currencies_analyzed']['date1_spot_npv']}")
    print(f"• Spot NPV currencies analyzed (Date 2): {results['summary']['currencies_analyzed']['date2_spot_npv']}")
    print(f"• New deal IDs: {', '.join(map(str, results['new_deals_analysis']['new_deal_ids'][:5]))}{'...' if len(results['new_deals_analysis']['new_deal_ids']) > 5 else ''}")
    
    # Show top 3 existing deals by BaseMV change
    print(f"\n🔄 Top 3 Existing Deals by BaseMV Change:")
    for i, deal in enumerate(results['existing_deals_analysis']['top_3_deals_by_basemv_change'], 1):
        print(f"  {i}. Deal {deal['deal_id']}: ${deal['basemv_change']:,.2f} (${deal['basemv_date1']:,.2f} → ${deal['basemv_date2']:,.2f}) ({', '.join(deal['currencies'])})")
    
    # Show top 3 deals by BaseMV impact
    print(f"\n🔝 Top 3 New Deals by BaseMV Impact:")
    for i, deal in enumerate(results['new_deals_analysis']['top_3_deals_by_basemv'], 1):
        print(f"  {i}. Deal {deal['deal_id']}: ${deal['basemv_contribution']:,.2f} ({', '.join(deal['currencies'])})")
    
    # Show sample rate ranges for forward deals
    print(f"\n📈 Sample Forward Rate Ranges (March 7, 2012):")
    date2_forward_stats = results['rate_statistics']['date2']['forward_currency_stats']
    for i, (currency, stats) in enumerate(list(date2_forward_stats.items())[:3]):
        fwd_min = stats['fwd_rate']['min']
        fwd_max = stats['fwd_rate']['max']
        if fwd_min is not None and fwd_max is not None:
            print(f"  {currency}: FwdRate {fwd_min:.6f} - {fwd_max:.6f}")
    
    # Show sample rate ranges for spot NPV deals
    print(f"\n📈 Sample Spot NPV Rate Ranges (March 7, 2012):")
    date2_spot_stats = results['rate_statistics']['date2']['spot_npv_currency_stats']
    for i, (currency, stats) in enumerate(list(date2_spot_stats.items())[:3]):
        spot_min = stats['spot_rate']['min']
        spot_max = stats['spot_rate']['max']
        if spot_min is not None and spot_max is not None:
            print(f"  {currency}: SpotRate {spot_min:.6f} - {spot_max:.6f}")

def call_chat_openai_api(results, api_key, model="gpt-3.5-turbo"):
            """
            Call the OpenAI Chat API with the analysis results as context.
            """

            # Prepare the chat message payload
            messages = [
                {
                    "role": "system",
                    "content": """
                    You are provided with metadata from a user's forex deal portfolio, comparing two reporting dates. Your task is to generate a business-level summary that explains the key drivers behind the change in Base Market Value (BaseMV). Focus on identifying whether the movement was primarily due to:

                    Introduction of new deals
                    Fluctuations in spot exchange rates
                    Changes in forward rates
                    Important Instruction:
                    Do not include currencies in the summary if their spot or forward rates remained stable between the two dates.

                    Use the following examples as guidance for the style and depth of analysis expected:

                    Example 1 - BaseMV Dip Due to New Deals:
                    On March 7, the user's BaseMV declined by $500M. Analysis revealed that 45 new deals were added, contributing a net negative BaseMV of $620M. Spot and forward rates remained relatively stable, indicating that the dip was primarily driven by the new deal activity.

                    Example 2 - BaseMV Rise Due to Rate Fluctuations:
                    Between March 6 and March 7, the BaseMV increased by $800M. Although 53 new deals were added with a net negative impact of $560M, favorable movements in EUR and AUD forward rates, along with a rise in JPY spot rates, contributed positively to the portfolio valuation.

                    Now, using the metadata provided, generate a similar business-level summary highlighting the financial impact and strategic implications of the BaseMV change.
                    """
                },
                {
                    "role": "user",
                    "content": f"Here are the JSON analysis results:\n{json.dumps(results, indent=2)}"
                }
            ]

            # Call the OpenAI Chat API
            api_url = "http://localhost:8000/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.7
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            response = response.json()

            # Extract and return the assistant's reply
            return response['choices'][0]['message']['content']

def main():
    """
    Main execution function
    """
    print("🚀 FOREX DEALS ANALYSIS - March 6-7, 2012")
    print("="*60)
    
    try:
        # Perform analysis
        results = load_and_analyze_forex_data()
        
        # Save to JSON
        save_results_to_json(results)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"\nJSON file contains:")
        print(f"• New deals analysis with BaseMV calculations and top 3 contributors")
        print(f"• Existing deals analysis with BaseMV changes and top 3 changes")
        print(f"• Forward deals rate statistics (min/max) by currency and date")
        print(f"• Spot NPV deals rate statistics (min/max) by currency and date")
        print(f"• Deal IDs and summary statistics")
        print(f"• Summary statistics and metadata")
        
    except FileNotFoundError:
        print("❌ Error: Cashflows_FX_V3.csv not found in data/ directory")
        print("   Please ensure the data file exists.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        
if __name__ == "__main__":
    main()
