"""
Forex Deals Analysis Streamlit Dashboard
Analyzes forex deals data to extract new deals and rate statistics for March 6-7, 2012
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
import traceback
from datetime import datetime
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
load_dotenv()


# Configure Streamlit page
st.set_page_config(
    page_title="FX Deals Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_forex_data():
    """Load and cache the forex data"""
    # Try different possible paths for the data file
    possible_paths = [
        '../data/Cashflows_FX_V3.csv',
        './data/Cashflows_FX_V3.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                data['PositionDate'] = pd.to_datetime(data['PositionDate'])
                st.success(f"✅ Data loaded successfully from: {path}")
                return data
            except Exception as e:
                st.error(f"❌ Error reading {path}: {str(e)}")
                continue
    
    st.error("❌ Error: Cashflows_FX_V3.csv not found in any expected location")
    st.info("Expected locations: " + ", ".join(possible_paths))
    return None

def analyze_forex_data(data):
    """
    Perform comprehensive analysis for March 6-7, 2012
    """
    # Define target dates
    date1 = datetime(2012, 3, 6)
    date2 = datetime(2012, 3, 7)
    
    # Filter data for the two dates
    data_date1 = data[data['PositionDate'] == date1]
    data_date2 = data[data['PositionDate'] == date2]
    
    # Initialize results dictionary
    results = {
        'target_dates': {
            'date1': date1.date().isoformat(),
            'date2': date2.date().isoformat()
        },
        'summary': {},
        'new_deals_analysis': {},
        'existing_deals_analysis': {},
        'rate_statistics': {}
    }
    
    # === NEW DEALS ANALYSIS ===
    # Get deal IDs for each date
    deals_date1 = set(data_date1['DealId'].unique())
    deals_date2 = set(data_date2['DealId'].unique())
    
    # Identify new deals and existing deals
    new_deals = deals_date2 - deals_date1
    existing_deals = deals_date1.intersection(deals_date2)
    
    # Get new deals data
    new_deals_data = data_date2[data_date2['DealId'].isin(new_deals)]
    new_deals_basemv_sum = new_deals_data['BaseMV'].sum()
    
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
    
    top_3_deals = sorted(deal_basemv_contributions, key=lambda x: abs(x['basemv_contribution']), reverse=True)[:3]
    
    results['new_deals_analysis'] = {
        'total_new_deals': len(new_deals),
        'total_new_records': len(new_deals_data),
        'basemv_sum': float(new_deals_basemv_sum),
        'new_deal_ids': list(new_deals),
        'top_3_deals_by_basemv': top_3_deals,
        'all_deal_contributions': deal_basemv_contributions
    }
    
    # === EXISTING DEALS ANALYSIS ===
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
    
    top_3_existing_deals = sorted(existing_deal_changes, key=lambda x: abs(x['basemv_change']), reverse=True)[:3]
    total_existing_basemv_change = sum(deal['basemv_change'] for deal in existing_deal_changes)
    
    results['existing_deals_analysis'] = {
        'total_existing_deals': len(existing_deals),
        'total_basemv_change': float(total_existing_basemv_change),
        'top_3_deals_by_basemv_change': top_3_existing_deals,
        'all_deal_changes': existing_deal_changes
    }
    
    # === RATE STATISTICS ANALYSIS ===
    forward_valuation_models = ['FORWARD', 'FORWARD NPV']
    spot_valuation_models = ['NPV SPOT']
    
    for date_key, date_value, date_data in [
        ('date1', date1, data_date1),
        ('date2', date2, data_date2)
    ]:
        forward_deals = date_data[date_data['ValuationModel'].isin(forward_valuation_models)]
        spot_deals = date_data[date_data['ValuationModel'].isin(spot_valuation_models)]
        
        # Calculate currency stats for forward deals
        forward_currency_stats = {}
        for currency in forward_deals['Currency'].unique():
            currency_data = forward_deals[forward_deals['Currency'] == currency]
            fwd_rates = currency_data['FwdRate'].dropna()
            
            forward_currency_stats[currency] = {
                'record_count': len(currency_data),
                'fwd_rate': {
                    'min': float(fwd_rates.min()) if len(fwd_rates) > 0 else None,
                    'max': float(fwd_rates.max()) if len(fwd_rates) > 0 else None,
                    'mean': float(fwd_rates.mean()) if len(fwd_rates) > 0 else None,
                    'count': len(fwd_rates)
                }
            }
        
        # Calculate currency stats for spot NPV deals
        
        spot_currency_stats = {}
        for currency in spot_deals['Currency'].unique():
            currency_data = spot_deals[spot_deals['Currency'] == currency]
            spot_rates = currency_data['SpotRate'].dropna()
            
            spot_currency_stats[currency] = {
                'record_count': len(currency_data),
                'spot_rate': {
                    'min': float(spot_rates.min()) if len(spot_rates) > 0 else None,
                    'max': float(spot_rates.max()) if len(spot_rates) > 0 else None,
                    'mean': float(spot_rates.mean()) if len(spot_rates) > 0 else None,
                    'count': len(spot_rates)
                }
            }
        
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
    
    return results, data_date1, data_date2, data

# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_basemv_comparison_chart(results, date1_str=None, date2_str=None):
    """Create BaseMV comparison chart with dynamic date labels"""
    basemv_data = results['summary']['basemv_totals']
    
    # Use provided date strings or fall back to defaults
    if date1_str is None:
        date1_str = 'Date 1'
    if date2_str is None:
        date2_str = 'Date 2'
    
    fig = go.Figure()
    
    # Add bars for each date
    fig.add_trace(go.Bar(
        x=[date1_str, date2_str],
        y=[basemv_data['date1_total'], basemv_data['date2_total']],
        name='BaseMV Total',
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f"${basemv_data['date1_total']:,.0f}", f"${basemv_data['date2_total']:,.0f}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="BaseMV Comparison Between Selected Dates",
        xaxis_title="Date",
        yaxis_title="BaseMV ($)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_deals_breakdown_chart(results, date2):
    """Create deals breakdown chart"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Deal Counts '+date2, 'BaseMV Impact '+date2),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Deal counts
    fig.add_trace(
        go.Bar(
            x=['Existing Deals', 'New Deals', 'Matured Deals'],
            y=[results['summary']['existing_deals_count'], 
               results['summary']['new_deals_count'],
               results['summary']['matured_deals_count']],
            name='Deal Count',
            marker_color=['#2ca02c', '#d62728', '#ff7f0e'],
            text=[results['summary']['existing_deals_count'], 
                  results['summary']['new_deals_count'],
                  results['summary']['matured_deals_count']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # BaseMV impact
    fig.add_trace(
        go.Bar(
            x=['Existing Deals Change', 'New Deals Total', 'Matured Deals Total'],
            y=[results['summary']['existing_deals_basemv_change'], 
               results['summary']['new_deals_basemv_sum'],
               results['summary']['matured_deals_basemv_sum']],
            name='BaseMV Impact',
            marker_color=['#ff7f0e', '#1f77b4', '#9467bd'],
            text=[f"${results['summary']['existing_deals_basemv_change']:,.0f}", 
                  f"${results['summary']['new_deals_basemv_sum']:,.0f}",
                  f"${results['summary']['matured_deals_basemv_sum']:,.0f}"],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_top_deals_chart(results, deal_type='new'):
    """Create top deals chart"""
    if deal_type == 'new':
        deals_data = results['new_deals_analysis']['top_3_deals_by_basemv']
        title = "Top 3 New Deals by BaseMV Impact"
        y_label = "BaseMV Contribution ($)"
    elif deal_type == 'matured':
        deals_data = results['matured_deals_analysis']['top_3_deals_by_basemv']
        title = "Top 3 Matured Deals by BaseMV Impact"
        y_label = "BaseMV Contribution ($)"
    else:
        deals_data = results['existing_deals_analysis']['top_3_deals_by_basemv_change']
        title = "Top 3 Existing Deals by BaseMV Change"
        y_label = "BaseMV Change ($)"
    
    if not deals_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    
    deal_ids = [f"Deal {deal['deal_id']}" for deal in deals_data]
    if deal_type == 'new' or deal_type == 'matured':
        values = [deal['basemv_contribution'] for deal in deals_data]
    else:
        values = [deal['basemv_change'] for deal in deals_data]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=deal_ids,
        y=values,
        text=[f"${val:,.0f}" for val in values],
        textposition='auto',
        marker_color=['#ff7f0e', '#2ca02c', '#d62728']
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Deal ID",
        yaxis_title=y_label,
        template="plotly_white",
        height=400
    )
    
    return fig

def create_rate_statistics_chart(results, rate_type='forward'):
    """Create rate statistics comparison chart"""
    date1_stats = results['rate_statistics']['date1']
    date2_stats = results['rate_statistics']['date2']
    
    if rate_type == 'forward':
        stats1 = date1_stats['forward_currency_stats']
        stats2 = date2_stats['forward_currency_stats']
        rate_key = 'fwd_rate'
        title = "Forward Rate Ranges by Currency"
    else:
        stats1 = date1_stats['spot_npv_currency_stats']
        stats2 = date2_stats['spot_npv_currency_stats']
        rate_key = 'spot_rate'
        title = "Spot Rate Ranges by Currency"
    
    # Get common currencies
    common_currencies = set(stats1.keys()).intersection(set(stats2.keys()))
    if not common_currencies:
        fig = go.Figure()
        fig.add_annotation(text="No common currencies found", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    
    fig = go.Figure()
    
    for currency in list(common_currencies)[:5]:  # Limit to 5 currencies for readability
        if (stats1[currency][rate_key]['min'] is not None and 
            stats2[currency][rate_key]['min'] is not None):
            
            fig.add_trace(go.Scatter(
                x=[f"{currency} (Mar 6)", f"{currency} (Mar 7)"],
                y=[stats1[currency][rate_key]['min'], stats2[currency][rate_key]['min']],
                mode='lines+markers',
                name=f"{currency} Min",
                line=dict(dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=[f"{currency} (Mar 6)", f"{currency} (Mar 7)"],
                y=[stats1[currency][rate_key]['max'], stats2[currency][rate_key]['max']],
                mode='lines+markers',
                name=f"{currency} Max"
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Currency and Date",
        yaxis_title="Rate",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_unified_currency_minmax_chart(data_date1, data_date2, currency, date1, date2, rate_type='forward'):
    """Create unified min/max rate chart for selected currency (forward or spot)"""
    if rate_type == 'forward':
        valuation_models = ['FORWARD', 'FORWARD NPV']
        rate_column = 'FwdRate'
        rate_name = 'Forward Rate'
        min_color = 'lightcoral'
        max_color = 'lightblue'
        mean_color = 'darkgreen'
    else:
        valuation_models = ['NPV SPOT']
        rate_column = 'SpotRate'
        rate_name = 'Spot Rate'
        min_color = 'salmon'
        max_color = 'skyblue'
        mean_color = 'darkorange'
    
    # Filter for the appropriate deals
    data_date1_filtered = data_date1[data_date1['ValuationModel'].isin(valuation_models)]
    data_date2_filtered = data_date2[data_date2['ValuationModel'].isin(valuation_models)]
    
    currency_date1 = data_date1_filtered[data_date1_filtered['Currency'] == currency]
    currency_date2 = data_date2_filtered[data_date2_filtered['Currency'] == currency]
    
    fig = go.Figure()
    
    if not currency_date1.empty and not currency_date2.empty:
        # Calculate min, max, and mean rates for each date
        rates_date1 = currency_date1[rate_column].dropna()
        rates_date2 = currency_date2[rate_column].dropna()
        
        if len(rates_date1) > 0 and len(rates_date2) > 0:
            date1_min = rates_date1.min()
            date1_max = rates_date1.max()
            date1_mean = rates_date1.mean()
            date2_min = rates_date2.min()
            date2_max = rates_date2.max()
            date2_mean = rates_date2.mean()
            
            # Add min line
            fig.add_trace(go.Scatter(
                x=[date1.strftime('%b %d, %Y'), date2.strftime('%b %d, %Y')],
                y=[date1_min, date2_min],
                mode='lines+markers',
                name=f'Minimum {rate_name}',
                line=dict(color=min_color, width=3, dash='dash'),
                marker=dict(size=8, color=min_color),
                hovertemplate=f'<b>%{{x}}</b><br>Min Rate: %{{y:.6f}}<extra></extra>'
            ))
            
            # Add max line
            fig.add_trace(go.Scatter(
                x=[date1.strftime('%b %d, %Y'), date2.strftime('%b %d, %Y')],
                y=[date1_max, date2_max],
                mode='lines+markers',
                name=f'Maximum {rate_name}',
                line=dict(color=max_color, width=3, dash='dash'),
                marker=dict(size=8, color=max_color),
                hovertemplate=f'<b>%{{x}}</b><br>Max Rate: %{{y:.6f}}<extra></extra>'
            ))
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=[date1.strftime('%b %d, %Y'), date2.strftime('%b %d, %Y')],
                y=[date1_mean, date2_mean],
                mode='lines+markers',
                name=f'Mean {rate_name}',
                line=dict(color=mean_color, width=4),
                marker=dict(size=10, color=mean_color),
                hovertemplate=f'<b>%{{x}}</b><br>Mean Rate: %{{y:.6f}}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{currency} - {rate_name} Trends (Min/Max/Mean)",
                xaxis_title="Date",
                yaxis_title=rate_name,
                template="plotly_white",
                height=400,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
        else:
            fig.add_annotation(
                text=f"No {rate_name.lower()} data available for {currency} on selected dates",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper"
            )
            fig.update_layout(
                title=f"{currency} - Min/Max {rate_name} Comparison",
                template="plotly_white",
                height=400
            )
    else:
        fig.add_annotation(
            text=f"No {rate_name.lower()} deals available for {currency} on selected dates",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper"
        )
        fig.update_layout(
            title=f"{currency} - Min/Max {rate_name} Comparison",
            template="plotly_white",
            height=400
        )
    
    return fig

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def display_currency_rate_statistics(data_date1, data_date2, selected_currency, date1, date2, rate_type='forward'):
    """Display currency rate statistics in a consistent format"""
    if rate_type == 'forward':
        valuation_models = ['FORWARD', 'FORWARD NPV']
        rate_column = 'FwdRate'
        rate_name = 'Forward'
    else:
        valuation_models = ['NPV SPOT']
        rate_column = 'SpotRate'
        rate_name = 'Spot'
    
    # Filter data
    data_date1_filtered = data_date1[data_date1['ValuationModel'].isin(valuation_models)]
    data_date2_filtered = data_date2[data_date2['ValuationModel'].isin(valuation_models)]
    currency_date1 = data_date1_filtered[data_date1_filtered['Currency'] == selected_currency]
    currency_date2 = data_date2_filtered[data_date2_filtered['Currency'] == selected_currency]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not currency_date1.empty:
            st.metric(
                f"{selected_currency} {rate_name} Records ({date1.strftime('%b %d')})",
                len(currency_date1)
            )
        else:
            st.metric(f"{selected_currency} {rate_name} Records ({date1.strftime('%b %d')})", 0)
    
    with col2:
        if not currency_date2.empty:
            st.metric(
                f"{selected_currency} {rate_name} Records ({date2.strftime('%b %d')})",
                len(currency_date2)
            )
        else:
            st.metric(f"{selected_currency} {rate_name} Records ({date2.strftime('%b %d')})", 0)
    
    with col3:
        if not currency_date1.empty:
            rates_date1 = currency_date1[rate_column].dropna()
            if len(rates_date1) > 0:
                mean_rate_date1 = rates_date1.mean()
                st.metric(
                    f"Mean {rate_name} Rate ({date1.strftime('%b %d')})",
                    f"{mean_rate_date1:.6f}"
                )
            else:
                st.metric(f"Mean {rate_name} Rate ({date1.strftime('%b %d')})", "N/A")
        else:
            st.metric(f"Mean {rate_name} Rate ({date1.strftime('%b %d')})", "N/A")
    
    with col4:
        if not currency_date2.empty:
            rates_date2 = currency_date2[rate_column].dropna()
            if len(rates_date2) > 0:
                mean_rate_date2 = rates_date2.mean()
                st.metric(
                    f"Mean {rate_name} Rate ({date2.strftime('%b %d')})",
                    f"{mean_rate_date2:.6f}"
                )
            else:
                st.metric(f"Mean {rate_name} Rate ({date2.strftime('%b %d')})", "N/A")
        else:
            st.metric(f"Mean {rate_name} Rate ({date2.strftime('%b %d')})", "N/A")

def display_analysis_summary_metrics(results, date1, date2):
    """Display deal counts and changes in a consistent format"""
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric(
            f"Total Deals ({date1.strftime('%b %d')})",
            results['summary']['total_unique_deals_date1']
        )
    
    with col_metrics2:
        st.metric(
            f"Total Deals ({date2.strftime('%b %d')})",
            results['summary']['total_unique_deals_date2']
        )
    
    with col_metrics3:
        deals_change = results['summary']['total_unique_deals_date2'] - results['summary']['total_unique_deals_date1']
        st.metric(
            "Deal Count Change",
            deals_change
        )

def display_key_financial_metrics(results, date1, date2):
    """Display key financial metrics in a consistent format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"📊 BaseMV ({date1.strftime('%b %d')})",
            f"${results['summary']['basemv_totals']['date1_total']:,.0f}"
        )
    
    with col2:
        st.metric(
            f"📊 BaseMV ({date2.strftime('%b %d')})",
            f"${results['summary']['basemv_totals']['date2_total']:,.0f}"
        )
    
    with col3:
        st.metric(
            f"📊 BaseMV Change",
            f"${results['summary']['basemv_totals']['difference']:,.0f}"
        )
    
    # Second row: Deal impacts
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            "🆕 New Deals Impact",
            f"${results['summary']['new_deals_basemv_sum']:,.0f}"
        )
    
    with col5:
        st.metric(
            "🔄 Existing Deals Change",
            f"${results['summary']['existing_deals_basemv_change']:,.0f}"
        )
    
    with col6:
        st.metric(
            "📤 Matured Deals Impact",
            f"${results['summary']['matured_deals_basemv_sum']:,.0f}"
        )

# =============================================================================
# API INTEGRATION FUNCTIONS
# =============================================================================

def call_chat_openai_api(results, api_key, user_context=None, model="gpt-4", api_version="2023-05-15"):
    """
    Call the OpenAI Chat API with the analysis results as context.
    """
    print("Calling OpenAI API with model:", model)
    
    # Build user context string
    context_info = ""
    if user_context:
        context_parts = []
        if user_context.get('entity') and user_context['entity'] != 'All':
            context_parts.append(f"Entity: {user_context['entity']}")
        if user_context.get('counterparty') and user_context['counterparty'] != 'All':
            context_parts.append(f"Counterparty: {user_context['counterparty']}")
        
        if context_parts:
            context_info = f"\n\nThis analysis is filtered for specific users: {' | '.join(context_parts)}. Please include this context in your summary."
    
    # Prepare the chat message payload
    messages = [
        {
            "role": "system",
            "content": f"""
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

            Now, using the metadata provided, generate a similar business-level summary highlighting the financial impact and strategic implications of the BaseMV change.{context_info}
            """
        },
        {
            "role": "user",
            "content": f"Here are the JSON analysis results:\n{json.dumps(results, indent=2, cls=NumpyEncoder)}\n\n"
        }
    ]

    # Call the OpenAI Chat API
    api_url = os.getenv("api_url")
    
    if api_url and api_key:
        api_url = api_url.format(model=model, api_version=api_version)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache"
        }
        try:
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            response = response.json()
            return response['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as err:
            raise Exception(f"HTTP error occurred: {err}")
        except Exception as e:
            raise Exception(f"An error occurred while calling the OpenAI API: {e}")
    else:
        # Fallback mock response for demo purposes
        return (
            f"MR:\nBetween {results['target_dates']['date1']} and {results['target_dates']['date2']}, "
            f"the BaseMV changed by ${results['summary']['basemv_totals']['difference']:,.0f}. "
            f"A total of {results['summary']['new_deals_count']} new deals were added, "
            f"contributing ${results['summary']['new_deals_basemv_sum']:,.0f} to the portfolio. "
            f"Existing deals showed a change of ${results['summary']['existing_deals_basemv_change']:,.0f}. "
            "The overall movement reflects market dynamics and portfolio adjustments during this period."
        )

def calculate_daily_basemv_sum(data):
    """
    Calculate the sum of BaseMV for each date in the dataset
    
    Args:
        data (pd.DataFrame): The forex data with PositionDate and BaseMV columns
    
    Returns:
        pd.DataFrame: DataFrame with PositionDate and daily BaseMV sums
    """
    # Group by PositionDate and sum BaseMV for each date
    daily_basemv = data.groupby('PositionDate')['BaseMV'].sum().reset_index()
    daily_basemv = daily_basemv.sort_values('PositionDate')
    daily_basemv['BaseMV_Sum_Formatted'] = daily_basemv['BaseMV'].apply(lambda x: f"${x:,.0f}")
    
    return daily_basemv

def create_daily_basemv_line_chart(daily_basemv_data, selected_date1=None, selected_date2=None):
    """
    Create a line chart showing BaseMV sum for each date
    
    Args:
        daily_basemv_data (pd.DataFrame): DataFrame with PositionDate and BaseMV columns
        selected_date1 (date): First selected analysis date
        selected_date2 (date): Second selected analysis date
    
    Returns:
        plotly.graph_objects.Figure: Line chart figure
    """
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=daily_basemv_data['PositionDate'],
        y=daily_basemv_data['BaseMV'],
        mode='lines+markers',
        name='Daily BaseMV Sum',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6, color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>BaseMV Sum:</b> $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Highlight selected analysis dates if provided
    if selected_date1 and selected_date2:
        analysis_dates = daily_basemv_data[
            daily_basemv_data['PositionDate'].dt.date.isin([selected_date1, selected_date2])
        ]
        
        if not analysis_dates.empty:
            fig.add_trace(go.Scatter(
                x=analysis_dates['PositionDate'],
                y=analysis_dates['BaseMV'],
                mode='markers',
                name='Selected Analysis Dates',
                marker=dict(size=12, color='red', symbol='star'),
                hovertemplate='<b>Selected Date:</b> %{x}<br>' +
                              '<b>BaseMV Sum:</b> $%{y:,.0f}<br>' +
                              '<extra></extra>'
            ))
    else:
        # Fallback: Highlight March 6-7, 2012 if they exist in the data
        march_dates = daily_basemv_data[
            daily_basemv_data['PositionDate'].dt.date.isin([
                datetime(2012, 3, 6).date(), 
                datetime(2012, 3, 7).date()
            ])
        ]
        
        if not march_dates.empty:
            fig.add_trace(go.Scatter(
                x=march_dates['PositionDate'],
                y=march_dates['BaseMV'],
                mode='markers',
                name='Default Analysis Dates',
                marker=dict(size=12, color='red', symbol='star'),
                hovertemplate='<b>Analysis Date:</b> %{x}<br>' +
                              '<b>BaseMV Sum:</b> $%{y:,.0f}<br>' +
                              '<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title="Daily BaseMV Sum Over Time",
        xaxis_title="Date",
        yaxis_title="BaseMV Sum ($)",
        template="plotly_white",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Format y-axis to show currency
    # fig.update_yaxis(tickformat='$,.0f')
    
    return fig

def main():
    """Main Streamlit application"""
    
    # =============================================================================
    # DATA LOADING AND PROCESSING
    # =============================================================================
    
    # Load data
    with st.spinner("Loading forex data..."):
        data = load_forex_data()
    
    if data is None:
        st.stop()
    
    # =============================================================================
    # USER FILTERING (MOVED TO SIDEBAR)
    # =============================================================================
    # Get unique entities and counterparties
    available_entities = sorted(list(data['Entity'].dropna().unique()))
    available_counterparties = sorted(list(data['Counterparty'].dropna().unique()))
    
    # Create single user selection interface in sidebar
    with st.sidebar:
        st.markdown("### 👥 User Selection")
        st.markdown("Select ONE user for analysis:")
        
        user_type = st.radio(
            "Select user type:",
            options=["All Users", "Entity", "Counterparty"],
            index=0,
            help="Choose to analyze all users, or filter by a specific Entity or Counterparty"
        )
        
        selected_entity = 'All'
        selected_counterparty = 'All'
        selected_user = None
        
        if user_type == "Entity":
            selected_user = st.selectbox(
                "Select Entity:",
                options=available_entities,
                help="Choose specific entity for analysis"
            )
            selected_entity = selected_user
        elif user_type == "Counterparty":
            selected_user = st.selectbox(
                "Select Counterparty:",
                options=available_counterparties,
                help="Choose specific counterparty for analysis"
            )
            selected_counterparty = selected_user
    
    # Filter data based on selection
    filtered_data = data.copy()
    
    if user_type == "Entity":
        filtered_data = filtered_data[filtered_data['Entity'] == selected_entity]
    elif user_type == "Counterparty":
        filtered_data = filtered_data[filtered_data['Counterparty'] == selected_counterparty]
    
    # Display filtering information in main area (compact)
    if user_type != "All Users":
        st.success(f"🔍 **Filtered Analysis:** {user_type}: {selected_user} | **Records:** {len(filtered_data):,} (from {len(data):,} total)")
    else:
        st.info("📊 **Analyzing all users** (no filters applied)")
    
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.error(f"❌ No data available for the selected {user_type}: {selected_user}. Please choose a different user.")
        st.stop()
    
    # =============================================================================
    # HEADER AND INITIAL SETUP
    # =============================================================================
    
    # Header
    st.title("🚀 FX Deals Analysis Dashboard")
    
    # Display current user context if filtered
    if user_type != "All Users":
        st.caption(f"🎯 Filtered for: {user_type}: {selected_user}")
    
    st.markdown("---")
    
    # Sidebar
    
    # Display current filters in sidebar (updated after user selection)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📈 Data Overview")
        
        if user_type != "All Users":
            st.markdown(f"**Current Filter:** {user_type}")
            st.markdown(f"**Selected:** {selected_user}")
            st.markdown(f"**Filtered Records:** {len(filtered_data):,}")
            st.markdown(f"**Total Records:** {len(data):,}")
            
            # Calculate filter percentage
            filter_percentage = (len(filtered_data) / len(data)) * 100
            st.markdown(f"**Coverage:** {filter_percentage:.1f}%")
        else:
            st.markdown(f"**Total Records:** {len(filtered_data):,}")
            st.markdown("**Filter:** None (All Users)")
    
    st.sidebar.markdown("---")
    
    # st.markdown("---")
    
    # Perform analysis
    with st.spinner("Performing analysis..."):
        results, data_date1, data_date2, full_data = analyze_forex_data(filtered_data)
    
    # =============================================================================
    # BASEMV TIMELINE VISUALIZATION
    # =============================================================================
    st.subheader("📈 Daily BaseMV Trend Over Time")
    daily_basemv_detailed = calculate_daily_basemv_sum(full_data)
    
    # Initialize session state for tracking selected dates
    if 'chart_placeholder' not in st.session_state:
        st.session_state.chart_placeholder = st.empty()
    
    # Show the initial chart
    st.session_state.chart_placeholder.plotly_chart(create_daily_basemv_line_chart(daily_basemv_detailed), use_container_width=True, key="daily_basemv_initial")
    
    # Display daily statistics table in expandable section
    with st.expander("📋 View Daily BaseMV Data Table"):
        st.dataframe(
            daily_basemv_detailed[['PositionDate', 'BaseMV_Sum_Formatted']].rename(columns={
                'PositionDate': 'Date',
                'BaseMV_Sum_Formatted': 'BaseMV Sum Formatted'
            }),
            use_container_width=True,
            height=300
        )
    
    st.markdown("---")
    
    # --- User selects two analysis dates from the line chart's available dates ---
    st.subheader("📊 Select Analysis Dates")
    available_dates = sorted(daily_basemv_detailed['PositionDate'].dt.date.unique())
    
    # Default dates
    default_date1 = datetime(2012, 3, 6).date() if datetime(2012, 3, 6).date() in available_dates else available_dates[0]
    default_date2 = datetime(2012, 3, 7).date() if datetime(2012, 3, 7).date() in available_dates else (available_dates[1] if len(available_dates) > 1 else available_dates[0])
    
    # Create two columns for the dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date1 = st.selectbox(
            "First Analysis Date:",
            options=available_dates,
            index=available_dates.index(default_date1),
            help="Select the first date for comparison analysis"
        )
    
    with col2:
        selected_date2 = st.selectbox(
            "Second Analysis Date:",
            options=available_dates,
            index=available_dates.index(default_date2),
            help="Select the second date for comparison analysis"
        )
    
    # Validation
    if selected_date1 == selected_date2:
        st.warning("⚠️ Please select two different dates for comparison analysis.")
        st.stop()
    
    # Show selected dates info
    st.info(f"📊 Analyzing: **{selected_date1.strftime('%B %d, %Y')}** vs **{selected_date2.strftime('%B %d, %Y')}**")
    
    # Update the chart to highlight selected dates
    st.session_state.chart_placeholder.plotly_chart(
        create_daily_basemv_line_chart(daily_basemv_detailed, selected_date1, selected_date2), 
        use_container_width=True,
        key="daily_basemv_updated"
    )

    # Filter data for selected dates
    date1, date2 = sorted([selected_date1, selected_date2])
    data_date1 = full_data[full_data['PositionDate'].dt.date == date1]
    data_date2 = full_data[full_data['PositionDate'].dt.date == date2]

    # Re-run analysis for selected dates
    def analyze_forex_data_for_dates(data, date1, date2):
        # Copy of analyze_forex_data, but with date1 and date2 as arguments
        # Only the date filtering and results['target_dates'] assignment change
        # All other logic is the same as in analyze_forex_data
        # (Copy-paste the function and replace date1/date2 assignments)
        # --- BEGIN COPY ---
        # Initialize results dictionary
        results = {
            'target_dates': {
                'date1': date1.isoformat(),
                'date2': date2.isoformat()
            },
            'summary': {},
            'new_deals_analysis': {},
            'existing_deals_analysis': {},
            'rate_statistics': {}
        }

        # Filter data for the two dates
        data_date1 = data[data['PositionDate'].dt.date == date1]
        data_date2 = data[data['PositionDate'].dt.date == date2]

        # === NEW DEALS ANALYSIS ===
        deals_date1 = set(data_date1['DealId'].unique())
        deals_date2 = set(data_date2['DealId'].unique())
        new_deals = deals_date2 - deals_date1
        existing_deals = deals_date1.intersection(deals_date2)
        matured_deals = deals_date1 - deals_date2
        new_deals_data = data_date2[data_date2['DealId'].isin(new_deals)]
        new_deals_basemv_sum = new_deals_data['BaseMV'].sum()
        
        # Get matured deals data (deals that existed on date1 but not on date2)
        matured_deals_data = data_date1[data_date1['DealId'].isin(matured_deals)]
        matured_deals_basemv_sum = matured_deals_data['BaseMV'].sum()
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
        top_3_deals = sorted(deal_basemv_contributions, key=lambda x: abs(x['basemv_contribution']), reverse=True)[:3]
        
        # Get top 3 matured deals by BaseMV contribution
        matured_deal_contributions = []
        for deal_id in matured_deals:
            deal_records = matured_deals_data[matured_deals_data['DealId'] == deal_id]
            deal_basemv_sum = deal_records['BaseMV'].sum()
            matured_deal_contributions.append({
                'deal_id': deal_id,
                'basemv_contribution': float(deal_basemv_sum),
                'record_count': len(deal_records),
                'currencies': list(deal_records['Currency'].unique())
            })
        
        top_3_matured_deals = sorted(matured_deal_contributions, key=lambda x: abs(x['basemv_contribution']), reverse=True)[:3]
        results['new_deals_analysis'] = {
            'total_new_deals': len(new_deals),
            'total_new_records': len(new_deals_data),
            'basemv_sum': float(new_deals_basemv_sum),
            'new_deal_ids': list(new_deals),
            'top_3_deals_by_basemv': top_3_deals,
            'all_deal_contributions': deal_basemv_contributions
        }
        
        results['matured_deals_analysis'] = {
            'total_matured_deals': len(matured_deals),
            'total_matured_records': len(matured_deals_data),
            'basemv_sum': float(matured_deals_basemv_sum),
            'matured_deal_ids': list(matured_deals),
            'top_3_deals_by_basemv': top_3_matured_deals,
            'all_deal_contributions': matured_deal_contributions
        }

        # === EXISTING DEALS ANALYSIS ===
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
        top_3_existing_deals = sorted(existing_deal_changes, key=lambda x: abs(x['basemv_change']), reverse=True)[:3]
        total_existing_basemv_change = sum(deal['basemv_change'] for deal in existing_deal_changes)
        results['existing_deals_analysis'] = {
            'total_existing_deals': len(existing_deals),
            'total_basemv_change': float(total_existing_basemv_change),
            'top_3_deals_by_basemv_change': top_3_existing_deals,
            'all_deal_changes': existing_deal_changes
        }

        # === RATE STATISTICS ANALYSIS ===
        forward_valuation_models = ['FORWARD', 'FORWARD NPV']
        spot_valuation_models = ['NPV SPOT']
        for date_key, date_value, date_data in [
            ('date1', date1, data_date1),
            ('date2', date2, data_date2)
        ]:
            forward_deals = date_data[date_data['ValuationModel'].isin(forward_valuation_models)]
            spot_deals = date_data[date_data['ValuationModel'].isin(spot_valuation_models)]
            forward_currency_stats = {}
            for currency in forward_deals['Currency'].unique():
                currency_data = forward_deals[forward_deals['Currency'] == currency]
                fwd_rates = currency_data['FwdRate'].dropna()
                forward_currency_stats[currency] = {
                    'record_count': len(currency_data),
                    'fwd_rate': {
                        'min': float(fwd_rates.min()) if len(fwd_rates) > 0 else None,
                        'max': float(fwd_rates.max()) if len(fwd_rates) > 0 else None,
                        'mean': float(fwd_rates.mean()) if len(fwd_rates) > 0 else None,
                        'count': len(fwd_rates)
                    }
                }
            spot_currency_stats = {}
            for currency in spot_deals['Currency'].unique():
                currency_data = spot_deals[spot_deals['Currency'] == currency]
                spot_rates = currency_data['SpotRate'].dropna()
                spot_currency_stats[currency] = {
                    'record_count': len(currency_data),
                    'spot_rate': {
                        'min': float(spot_rates.min()) if len(spot_rates) > 0 else None,
                        'max': float(spot_rates.max()) if len(spot_rates) > 0 else None,
                        'mean': float(spot_rates.mean()) if len(spot_rates) > 0 else None,
                        'count': len(spot_rates)
                    }
                }
            results['rate_statistics'][date_key] = {
                'date': date_value.isoformat(),
                'total_forward_records': len(forward_deals),
                'total_spot_npv_records': len(spot_deals),
                'currencies_analyzed_forward': len(forward_currency_stats),
                'currencies_analyzed_spot_npv': len(spot_currency_stats),
                'forward_currency_stats': forward_currency_stats,
                'spot_npv_currency_stats': spot_currency_stats
            }

        # === SUMMARY STATISTICS ===
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
            'matured_deals_count': len(matured_deals),
            'new_deals_basemv_sum': float(new_deals_basemv_sum),
            'matured_deals_basemv_sum': float(matured_deals_basemv_sum),
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
        return results, data_date1, data_date2

    # Run analysis for selected dates
    results, data_date1, data_date2 = analyze_forex_data_for_dates(full_data, date1, date2)

    # Currency Min/Max Analysis Section
    st.markdown("---")
    st.subheader("💱 Currency Forward Rate Min/Max Analysis")
    
    # Get available currencies from selected dates with forward valuation
    forward_valuation_models = ['FORWARD', 'FORWARD NPV']
    forward_data_date1 = data_date1[data_date1['ValuationModel'].isin(forward_valuation_models)]
    forward_data_date2 = data_date2[data_date2['ValuationModel'].isin(forward_valuation_models)]
    available_currencies = sorted(list(set(forward_data_date1['Currency'].unique()) | set(forward_data_date2['Currency'].unique())))
    
    if available_currencies:
        selected_currency = st.selectbox(
            "Select Currency for Forward Rate Min/Max Analysis:",
            options=available_currencies,
            index=0,
            help="Select a currency to view min/max forward rate analysis for the selected dates"
        )
        
        # Display the currency min/max chart
        currency_chart = create_unified_currency_minmax_chart(data_date1, data_date2, selected_currency, date1, date2, 'forward')
        st.plotly_chart(currency_chart, use_container_width=True, key="forward_rate_chart")
        
        # Display currency forward rate statistics
        display_currency_rate_statistics(data_date1, data_date2, selected_currency, date1, date2, 'forward')
    
    else:
        st.warning("No forward rate data available for the selected dates.")

    st.markdown("---")

    # Currency Spot Rate Min/Max Analysis Section
    st.subheader("💰 Currency Spot Rate Min/Max Analysis")
    
    # Get available currencies from selected dates with spot valuation
    spot_valuation_models = ['NPV SPOT']
    spot_data_date1 = data_date1[data_date1['ValuationModel'].isin(spot_valuation_models)]
    spot_data_date2 = data_date2[data_date2['ValuationModel'].isin(spot_valuation_models)]
    available_spot_currencies = sorted(list(set(spot_data_date1['Currency'].unique()) | set(spot_data_date2['Currency'].unique())))
    
    if available_spot_currencies:
        selected_spot_currency = st.selectbox(
            "Select Currency for Spot Rate Min/Max Analysis:",
            options=available_spot_currencies,
            index=0,
            help="Select a currency to view min/max spot rate analysis for the selected dates"
        )
        
        # Display the currency min/max chart
        spot_currency_chart = create_unified_currency_minmax_chart(data_date1, data_date2, selected_spot_currency, date1, date2, 'spot')
        st.plotly_chart(spot_currency_chart, use_container_width=True, key="spot_rate_chart")
        
        # Display currency spot rate statistics
        display_currency_rate_statistics(data_date1, data_date2, selected_spot_currency, date1, date2, 'spot')
    
    else:
        st.warning("No spot rate data available for the selected dates.")

    st.markdown("---")

    # --- All visualizations below use the selected dates/results ---

    # BaseMV comparison for analysis dates
    st.subheader(f"📊 Analysis for {date1.strftime('%b %d, %Y')} and {date2.strftime('%b %d, %Y')}")
    
    # Display deal counts for both dates
    display_analysis_summary_metrics(results, date1, date2)
    
    # Key Financial Metrics for selected dates
    st.subheader("📊 Key Financial Metrics")
    display_key_financial_metrics(results, date1, date2)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        date1_str = date1.strftime('%b %d, %Y')
        date2_str = date2.strftime('%b %d, %Y')
        st.plotly_chart(create_basemv_comparison_chart(results, date1_str, date2_str), use_container_width=True, key="basemv_comparison")

    with col2:
        st.plotly_chart(create_deals_breakdown_chart(results, date2.strftime('%b %d, %Y')), use_container_width=True, key="deals_breakdown")

    # Top deals analysis
    st.subheader("🔝 Top Deal Analysis")

    tab1, tab2, tab3 = st.tabs(["New Deals", "Existing Deals", "Matured Deals"])

    with tab1:
        st.plotly_chart(create_top_deals_chart(results, 'new'), use_container_width=True, key="top_new_deals")

        # Display top new deals table
        if results['new_deals_analysis']['top_3_deals_by_basemv']:
            st.subheader("Top 3 New Deals Details")
            top_deals_df = pd.DataFrame(results['new_deals_analysis']['top_3_deals_by_basemv'])
            top_deals_df['currencies'] = top_deals_df['currencies'].apply(lambda x: ', '.join(x))
            st.dataframe(top_deals_df, use_container_width=True)

    with tab2:
        st.plotly_chart(create_top_deals_chart(results, 'existing'), use_container_width=True, key="top_existing_deals")

        # Display top existing deals table
        if results['existing_deals_analysis']['top_3_deals_by_basemv_change']:
            st.subheader("Top 3 Existing Deals Changes")
            top_existing_df = pd.DataFrame(results['existing_deals_analysis']['top_3_deals_by_basemv_change'])
            top_existing_df['currencies'] = top_existing_df['currencies'].apply(lambda x: ', '.join(x))
            st.dataframe(top_existing_df, use_container_width=True)

    with tab3:
        st.plotly_chart(create_top_deals_chart(results, 'matured'), use_container_width=True, key="top_matured_deals")

        # Display top matured deals table
        if results['matured_deals_analysis']['top_3_deals_by_basemv']:
            st.subheader("Top 3 Matured Deals Details")
            top_matured_df = pd.DataFrame(results['matured_deals_analysis']['top_3_deals_by_basemv'])
            top_matured_df['currencies'] = top_matured_df['currencies'].apply(lambda x: ', '.join(x))
            st.dataframe(top_matured_df, use_container_width=True)
        
        # Display matured deals summary
        st.info(f"""
        **Matured Deals Summary:**
        - Total Matured Deals: {results['matured_deals_analysis']['total_matured_deals']}
        - Total Records: {results['matured_deals_analysis']['total_matured_records']}
        - Total BaseMV: ${results['matured_deals_analysis']['basemv_sum']:,.0f}
        """)

    # LLM-Generated Business Summary Section
    st.markdown("---")
    st.header("🤖 AI-Generated Business Summary")
    
    # Add toggle for generating LLM summary
    generate_summary = st.button("🔮 Generate Business Insights", type="primary")
    
    if generate_summary:
        api_key = os.getenv("api_key")
        api_url = os.getenv("api_url")
        
        # Debug information - this will show in the Streamlit UI
        with st.expander("🔧 Debug Information"):
            st.write(f"**API Key:** `{repr(api_key)}`")
            st.write(f"**API URL:** `{repr(api_url)}`")
            st.write(f"**API Key is not None:** `{api_key is not None}`")
            st.write(f"**Both API URL and Key truthy:** `{bool(api_url and api_key)}`")
            if not bool(api_url and api_key):
                st.info("🎭 Will use mock response since API credentials are empty")
        
        if api_key is not None:
            with st.spinner("🧠 Analyzing data and generating business insights..."):
                try:
                    st.info("📊 Calling OpenAI API function...")
                    # Prepare user context for API
                    user_context = {}
                    if user_type == "Entity":
                        user_context['entity'] = selected_entity
                    elif user_type == "Counterparty":
                        user_context['counterparty'] = selected_counterparty
                    
                    summary = call_chat_openai_api(results, api_key, user_context)
                    st.info(f"📝 Received response length: {len(summary)} characters")
                    
                    # Display the summary in an attractive format
                    st.success("✅ Business insights generated successfully!")
                    
                    # Create an attractive summary box
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        color: white;
                        margin: 10px 0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="color: white; margin-bottom: 15px;">📊 Executive Summary</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the summary text in a nice info box
                    st.info(f"💡 **AI Analysis:** {summary}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating business summary: {str(e)}")
                    st.error(f"🔍 Error type: {type(e).__name__}")
                    st.error(f"📋 Full error details: {repr(e)}")
                    
                    # Show traceback for debugging
                    st.code(traceback.format_exc())
                    
                    st.warning("💡 Make sure your OpenAI API key is configured in the environment variables.")
        else:
            st.warning("⚠️ OpenAI API key not found. Please configure the 'api_key' environment variable to generate AI insights.")
            st.info("💡 The AI summary provides business-level insights about BaseMV changes, rate fluctuations, and deal impacts.")
    
    # Raw data exploration
    with st.expander("🔍 Raw Data Exploration"):
        st.subheader("Data Sample")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{date1.strftime('%b %d, %Y')} Data Sample**")
            st.dataframe(data_date1.head(), use_container_width=True)

        with col2:
            st.write(f"**{date2.strftime('%b %d, %Y')} Data Sample**")
            st.dataframe(data_date2.head(), use_container_width=True)
    # Export functionality
    st.markdown("---")
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download Analysis Results (JSON)"):
            # Create user-specific filename
            user_suffix = ""
            if user_type != "All Users":
                user_suffix = f"_{user_type.lower()}_{selected_user}"
            
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"forex_analysis{user_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download Summary Report"):
            # Create summary report
            summary_lines = [
                "FOREX DEALS ANALYSIS SUMMARY",
                "=" * 50,
                f"Analysis Period: {results['target_dates']['date1']} to {results['target_dates']['date2']}",
                "",
                "KEY METRICS:",
                f"• Total BaseMV Change: ${results['summary']['basemv_totals']['difference']:,.2f}",
                f"• New Deals Count: {results['summary']['new_deals_count']}",
                f"• New Deals BaseMV Sum: ${results['summary']['new_deals_basemv_sum']:,.2f}",
                f"• Existing Deals Count: {results['summary']['existing_deals_count']}",
                f"• Existing Deals BaseMV Change: ${results['summary']['existing_deals_basemv_change']:,.2f}",
                f"• Matured Deals Count: {results['summary']['matured_deals_count']}",
                f"• Matured Deals BaseMV Sum: ${results['summary']['matured_deals_basemv_sum']:,.2f}",
                "",
                "TOP NEW DEALS:",
            ]
            
            for i, deal in enumerate(results['new_deals_analysis']['top_3_deals_by_basemv'], 1):
                summary_lines.append(f"{i}. Deal {deal['deal_id']}: ${deal['basemv_contribution']:,.2f}")
            
            summary_lines.extend([
                "",
                "TOP EXISTING DEALS CHANGES:",
            ])
            
            for i, deal in enumerate(results['existing_deals_analysis']['top_3_deals_by_basemv_change'], 1):
                summary_lines.append(f"{i}. Deal {deal['deal_id']}: ${deal['basemv_change']:,.2f}")
            
            summary_lines.extend([
                "",
                "TOP MATURED DEALS:",
            ])
            
            for i, deal in enumerate(results['matured_deals_analysis']['top_3_deals_by_basemv'], 1):
                summary_lines.append(f"{i}. Deal {deal['deal_id']}: ${deal['basemv_contribution']:,.2f}")
            
            summary_text = "\n".join(summary_lines)
            
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"forex_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
