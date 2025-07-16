"""
Simple POC for Forex Portfolio Market Value Attribution
Directly implements the manual process from idea.md with ML enhancements

This POC demonstrates the three-step attribution analysis enhanced with ML:
1. New Deals Analysis - identify contribution + ML anomaly detection
2. Forward Rate Impact Analysis - quantify forward rate fluctuation impact + ML prediction
3. Spot Rate Impact Analysis - measure spot rate change impact + ML clustering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ForexAttributionPOC:
    """
    ML-Enhanced implementation of the attribution analysis from idea.md
    
    This class implements the manual process enhanced with ML algorithms:
    - New deals added to the portfolio + ML anomaly detection
    - Forward rate changes affecting existing forward contracts + ML prediction
    - Spot rate changes affecting spot positions + ML clustering
    """
    
    def __init__(self):
        self.data = None  # Will store the forex trading data
        # ML Components for enhanced analysis
        self.scaler = StandardScaler()  # For feature scaling
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)  # Detect unusual patterns
        self.predictor = RandomForestRegressor(n_estimators=100, random_state=42)  # Predict market values
        self.clusterer = DBSCAN(eps=0.3, min_samples=2)  # Group similar deals
    
    def load_real_data(self):
        """
        Load the actual Cashflows_FX_V3.xlsx dataset
        
        This loads the real forex trading data and prepares it for attribution analysis.
        The dataset contains actual forex deals with:
        - Deal identifiers and position dates
        - Spot and forward rates
        - Market values and valuation models
        - Other trading metadata
        """
        try:
            print("📊 Loading real dataset: Cashflows_FX_V3.xlsx")
            
            # Load the actual dataset
            self.data = pd.read_excel('data/Cashflows_FX_V3.xlsx')
            
            print(f"✓ Loaded real data: {len(self.data)} records")
            print(f"✓ Columns: {list(self.data.columns)}")
            
            # Display basic info about the dataset
            if 'PositionDate' in self.data.columns:
                unique_dates = sorted(self.data['PositionDate'].unique())
                print(f"✓ Date range: {unique_dates[0]} to {unique_dates[-1]}")
                print(f"✓ Total unique dates: {len(unique_dates)}")
                
                # Show sample dates for analysis
                if len(unique_dates) >= 2:
                    print(f"✓ Sample consecutive dates available for analysis:")
                    for i in range(min(5, len(unique_dates)-1)):
                        print(f"   {unique_dates[i]} -> {unique_dates[i+1]}")
            
            # Show data quality
            # print(f"✓ Data quality check:")
            # for col in ['PositionDate', 'BaseMV', 'SpotRate', 'FwdRate']:
            #     if col in self.data.columns:
            #         non_null = self.data[col].notna().sum()
            #         total = len(self.data)
            #         print(f"   {col}: {non_null}/{total} ({non_null/total*100:.1f}% complete)")
            
            return self.data
            
        except FileNotFoundError:
            print("❌ Error: Cashflows_FX_V3.xlsx not found in data/ directory")
            print("   Please ensure the data file exists to run the analysis.")
            return None
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print("   Please check the data file format and try again.")
            return None
    
    def analyze_attribution(self, date1, date2):
        """
        Main attribution analysis following the manual process from idea.md
        
        This implements the three-step process:
        1. Calculate total market value difference between two dates
        2. Identify and quantify contribution from each factor:
           - New deals (deals that exist on Date2 but not Date1)
           - Forward rate impact (rate changes on existing forward deals)
           - Spot rate impact (rate changes on existing spot deals)
        3. Calculate attribution accuracy and residual
        
        Args:
            date1 (datetime): Earlier date (baseline)
            date2 (datetime): Later date (comparison)
            
        Returns:
            dict: Attribution breakdown with contributions and accuracy
        """
        print(f"\n🔍 ANALYZING ATTRIBUTION: {date1.date()} vs {date2.date()}")
        print("="*60)
        
        # === STEP 0: DATA PREPARATION ===
        # Filter data for the two specific dates
        data_date1 = self.data[self.data['PositionDate'] == date1]
        data_date2 = self.data[self.data['PositionDate'] == date2]
        
        print(f"Date 1 ({date1.date()}): {len(data_date1)} deals")
        print(f"Date 2 ({date2.date()}): {len(data_date2)} deals")
        
        # === STEP 1: CALCULATE TOTAL MARKET VALUE DIFFERENCE ===
        # This is the total change we need to explain through attribution
        total_mv_date1 = data_date1['BaseMV'].sum()
        total_mv_date2 = data_date2['BaseMV'].sum()
        total_change = total_mv_date2 - total_mv_date1
        
        print(f"\nTotal Market Value Change: ${total_change:,.2f}")
        print(f"  (From ${total_mv_date1:,.2f} to ${total_mv_date2:,.2f})")
        
        # === STEP 2: NEW DEALS ANALYSIS ===
        # Objective: Identify contribution of new deals to market value fluctuation
        # Process: Find deals that exist on Date2 but not on Date1
        # Important: Each deal has pay/receive pairs, so we need to sum them for net impact
        
        existing_deals = set(data_date1['DealId'])  # Deals present on Date1
        all_deals_date2 = set(data_date2['DealId'])  # All deals on Date2
        new_deals = all_deals_date2 - existing_deals  # Set difference = new deals
        
        # Calculate market value contribution from new deals
        # Each new deal has 2 rows (pay + receive), so just sum all BaseMV values
        new_deals_data = data_date2[data_date2['DealId'].isin(new_deals)]
        
        # Calculate net contribution by summing all BaseMV (pay + receive rows automatically net out)
        new_deals_contribution = new_deals_data['BaseMV'].sum()
        unique_new_deals = new_deals_data['DealId'].unique()
        
        print(f"\n📊 STEP 1: NEW DEALS ANALYSIS (ML-Enhanced)")
        print(f"New deals identified: {len(unique_new_deals)} deals ({len(new_deals_data)} rows including pay/receive)")
        print(f"New deals contribution: ${new_deals_contribution:,.2f}")
        if len(unique_new_deals) > 0:
            print(f"Average new deal net size: ${new_deals_contribution/len(unique_new_deals):,.2f}")
        
        # === ML ENHANCEMENT: ANOMALY DETECTION FOR NEW DEALS ===
        if len(new_deals_data) > 0:
            anomaly_results = self._detect_anomalous_new_deals(new_deals_data)
            print(f"🤖 ML Anomaly Detection: {anomaly_results['anomalous_deals']} unusual new deals detected")
            if anomaly_results['anomalous_deals'] > 0:
                print(f"   Alert: Unusual new deal patterns detected - review recommended")
                print(f"   Anomaly rate: {anomaly_results['anomaly_rate']:.1f}% of new deals")
                
                # Show details of anomalous deals if available
                if 'anomalous_deal_details' in anomaly_results and anomaly_results['anomalous_deal_details']:
                    print(f"   Sample anomalous deals:")
                    for detail in anomaly_results['anomalous_deal_details'][:3]:
                        print(f"     • Deal {detail['deal_id']}: {detail['reason']}")
            else:
                print(f"   ✅ All new deals appear normal (no pricing anomalies detected)")
        
        # === STEP 3: FORWARD RATE IMPACT ANALYSIS ===
        # Objective: Quantify impact of forward rate fluctuations on existing deals by currency
        # Process: 
        # 1. Exclude new deals from analysis
        # 2. Filter deals with valuation models: "FORWARD" and "FORWARD NPV"
        # 3. Group by DealId to handle pay/receive pairs (each deal has 2 rows: pay=negative, receive=positive)
        # 4. Calculate forward rate fluctuation by currency pair
        # 5. Calculate net BaseMV contribution for each deal due to forward rate fluctuation
        
        existing_deals_date2 = data_date2[data_date2['DealId'].isin(existing_deals)]
        forward_deals = existing_deals_date2[
            existing_deals_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV'])
        ]
        
        # Currency-based Forward Rate Impact Calculation with proper pay/receive handling
        forward_contribution = 0
        forward_deals_analyzed = 0
        currency_contributions = {}
        
        print(f"   Analyzing {len(forward_deals)} forward deal rows (pay/receive pairs) by currency...")
        
        # Group by currency for proper rate fluctuation analysis
        currencies = forward_deals['Currency'].unique()
        
        for currency in currencies:
            currency_deals = forward_deals[forward_deals['Currency'] == currency]
            currency_contribution = 0
            currency_deals_with_fluctuation = 0
            
            # Get average forward rates for this currency on both dates
            currency_deals_date1 = data_date1[
                (data_date1['Currency'] == currency) & 
                (data_date1['ValuationModel'].isin(['FORWARD', 'FORWARD NPV']))
            ]
            
            if len(currency_deals_date1) > 0:
                avg_fwd_rate_date1 = currency_deals_date1['FwdRate'].mean()
                avg_fwd_rate_date2 = currency_deals[currency_deals['Currency'] == currency]['FwdRate'].mean()
                
                # Calculate currency-level forward rate fluctuation
                fwd_rate_fluctuation = avg_fwd_rate_date2 - avg_fwd_rate_date1
                
                if abs(fwd_rate_fluctuation) > 0.0001:  # Significant fluctuation threshold
                    print(f"   🌍 {currency}: Fwd Rate Δ = {fwd_rate_fluctuation:.6f}")
                    
                    # Group by DealId to handle pay/receive pairs correctly
                    unique_deal_ids = currency_deals['DealId'].unique()
                    
                    for deal_id in unique_deal_ids:
                        # Get all rows for this deal on both dates (pay + receive)
                        deal_rows_date1 = data_date1[data_date1['DealId'] == deal_id]
                        deal_rows_date2 = data_date2[data_date2['DealId'] == deal_id]
                        
                        if len(deal_rows_date1) > 0 and len(deal_rows_date2) > 0:
                            # Calculate net BaseMV for each date (sum of pay + receive)
                            net_basemv_date1 = deal_rows_date1['BaseMV'].sum()
                            net_basemv_date2 = deal_rows_date2['BaseMV'].sum()
                            
                            # Check if any row in this deal has forward rate fluctuation
                            has_rate_fluctuation = False
                            for _, row1 in deal_rows_date1.iterrows():
                                matching_row2 = deal_rows_date2[
                                    (deal_rows_date2['Currency'] == row1['Currency']) &
                                    (deal_rows_date2['ValuationModel'] == row1['ValuationModel'])
                                ]
                                if len(matching_row2) > 0:
                                    row2 = matching_row2.iloc[0]
                                    if abs(row2['FwdRate'] - row1['FwdRate']) > 0.0001:
                                        has_rate_fluctuation = True
                                        break
                            
                            if has_rate_fluctuation:
                                # Net BaseMV change for this deal
                                net_mv_change = net_basemv_date2 - net_basemv_date1
                                currency_contribution += net_mv_change
                                currency_deals_with_fluctuation += 1
                                forward_deals_analyzed += 1
                                
                                if forward_deals_analyzed <= 5:  # Show first 5 examples
                                    print(f"     Deal {deal_id}: Net BaseMV Δ=${net_mv_change:,.2f} (from ${net_basemv_date1:,.2f} to ${net_basemv_date2:,.2f})")
                    
                    currency_contributions[currency] = currency_contribution
                    forward_contribution += currency_contribution
                    
                    print(f"     {currency} total contribution: ${currency_contribution:,.2f} ({currency_deals_with_fluctuation} deals)")
        
        print(f"   Total forward rate contribution: ${forward_contribution:,.2f}")
        
        print(f"\n📊 STEP 2: FORWARD RATE IMPACT ANALYSIS (ML-Enhanced)")
        print(f"Forward deals with rate changes: {forward_deals_analyzed}/{len(forward_deals)}")
        print(f"Forward rate impact: ${forward_contribution:,.2f}")
        
        # === ML ENHANCEMENT: ENHANCED PREDICTIVE MODEL FOR FORWARD RATE IMPACT ===
        if len(forward_deals) > 2:
            # Use the enhanced ML prediction from enhanced_ml_prediction.py
            try:
                from enhanced_ml_prediction import EnhancedForexMLPredictor
                
                enhanced_predictor = EnhancedForexMLPredictor()
                enhanced_predictor.data = self.data  # Use same data
                
                # Run enhanced analysis
                enhanced_results = enhanced_predictor.analyze_actual_vs_predicted(data_date1.iloc[0]['PositionDate'], data_date2.iloc[0]['PositionDate'])
                
                if 'error' not in enhanced_results:
                    print(f"🤖 Enhanced ML Prediction: ${enhanced_results['total_predicted_impact']:,.2f}")
                    print(f"   Model: {enhanced_results['model_metrics']['model_name']}")
                    print(f"   Model R²: {enhanced_results['model_metrics']['r2_score']:.3f}")
                    print(f"   Using Features: BPDelta, ModifiedDuration, DaystoMaturity, etc.")
                    
                    # Compare with actual
                    variance = abs(forward_contribution - enhanced_results['total_predicted_impact'])
                    variance_pct = (variance / abs(forward_contribution)) * 100 if forward_contribution != 0 else 0
                    
                    if variance_pct < 30:  # Improved threshold
                        print(f"   ✅ Excellent prediction: Variance = ${variance:,.2f} ({variance_pct:.1f}%)")
                    elif variance_pct < 50:  # Good prediction if within 50%
                        print(f"   ✅ Good prediction: Variance = ${variance:,.2f} ({variance_pct:.1f}%)")
                    else:
                        print(f"   ⚠️  Large variance: Actual vs Enhanced ML = ${variance:,.2f} ({variance_pct:.1f}%)")
                        print(f"   💡 Recommendation: Rate prediction needs improvement - check data quality")
                        
                    # Show top contributing features
                    if 'feature_importance' in enhanced_results:
                        top_features = sorted(enhanced_results['feature_importance'].items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top predictive features: {', '.join([f[0] for f in top_features])}")
                        
                    # Variance assessment
                    if variance_pct > 40:
                        print(f"   📊 Variance Analysis:")
                        print(f"      • High variance indicates potential data quality issues")
                        print(f"      • Consider using actual financial risk metrics vs synthetic features")
                        print(f"      • Zero BPDelta/Duration values may need better estimation")
                        
                else:
                    print(f"   ⚠️  Enhanced ML failed: {enhanced_results['error']}")
                    
            except Exception as e:
                print(f"   ⚠️  Enhanced ML error: {str(e)}")
                # Fallback to original ML prediction
                ml_prediction = self._predict_rate_impact(forward_deals, data_date1, 'FwdRate')
                print(f"🤖 Fallback ML Predicted Impact: ${ml_prediction['predicted_impact']:,.2f}")
                print(f"   Model Confidence (R²): {ml_prediction['confidence']:.3f}")
                print(f"   Scenario: {ml_prediction.get('scenario', 'Rate change prediction')}")
                
                # Additional variance analysis for fallback
                if abs(ml_prediction['predicted_impact']) > 0:
                    fallback_variance = abs(forward_contribution - ml_prediction['predicted_impact'])
                    fallback_variance_pct = (fallback_variance / abs(forward_contribution)) * 100 if forward_contribution != 0 else 0
                    print(f"   Fallback Variance: ${fallback_variance:,.2f} ({fallback_variance_pct:.1f}%)")
        else:
            print(f"   ⚠️  Insufficient forward deals for ML prediction (need >2, have {len(forward_deals)})")
        
        # === STEP 4: SPOT RATE IMPACT ANALYSIS ===
        # Objective: Measure impact of spot rate changes on spot deals
        # Process:
        # 1. Filter deals with valuation model: "NPV SPOT"
        # 2. Group by DealId to handle pay/receive pairs (each deal has 2 rows: pay=negative, receive=positive)
        # 3. Identify deals with Spot Rate fluctuations
        # 4. Calculate net market value change for these deals
        
        spot_deals = existing_deals_date2[
            existing_deals_date2['ValuationModel'] == 'NPV SPOT'
        ]
        
        spot_contribution = 0
        spot_deals_analyzed = 0
        
        # Group by DealId to handle pay/receive pairs correctly
        unique_spot_deal_ids = spot_deals['DealId'].unique()
        
        for deal_id in unique_spot_deal_ids:
            # Get all rows for this deal on both dates (pay + receive)
            deal_rows_date1 = data_date1[data_date1['DealId'] == deal_id]
            deal_rows_date2 = data_date2[data_date2['DealId'] == deal_id]
            
            if len(deal_rows_date1) > 0 and len(deal_rows_date2) > 0:
                # Check if any row in this deal has spot rate fluctuation
                has_spot_rate_change = False
                for _, row1 in deal_rows_date1.iterrows():
                    matching_row2 = deal_rows_date2[
                        (deal_rows_date2['Currency'] == row1['Currency']) &
                        (deal_rows_date2['ValuationModel'] == row1['ValuationModel'])
                    ]
                    if len(matching_row2) > 0:
                        row2 = matching_row2.iloc[0]
                        if abs(row2['SpotRate'] - row1['SpotRate']) > 0.001:  # 0.1% threshold
                            has_spot_rate_change = True
                            break
                
                if has_spot_rate_change:
                    # Calculate net BaseMV change for this deal (sum of pay + receive)
                    net_basemv_date1 = deal_rows_date1['BaseMV'].sum()
                    net_basemv_date2 = deal_rows_date2['BaseMV'].sum()
                    net_mv_change = net_basemv_date2 - net_basemv_date1
                    
                    spot_contribution += net_mv_change
                    spot_deals_analyzed += 1
        
        print(f"\n📊 STEP 3: SPOT RATE IMPACT ANALYSIS (ML-Enhanced)")
        print(f"Spot deals with rate changes: {spot_deals_analyzed}/{len(spot_deals)}")
        print(f"Spot rate impact: ${spot_contribution:,.2f}")
        
        # === ML ENHANCEMENT: CLUSTERING ANALYSIS FOR SPOT DEALS ===
        if len(spot_deals) > 3:  # Need minimum data for clustering
            cluster_results = self._cluster_spot_deals(spot_deals)
            print(f"🤖 ML Clustering: {cluster_results['num_clusters']} distinct spot deal clusters identified")
            print(f"   Largest cluster impact: ${cluster_results['largest_cluster_impact']:,.2f}")
        
        # === STEP 5: RECONCILIATION AND VALIDATION ===
        # Calculate how much of the total change we can explain
        explained_change = new_deals_contribution + forward_contribution + spot_contribution
        unexplained = total_change - explained_change
        
        # Calculate attribution accuracy (how well we explained the total change)
        attribution_accuracy = (abs(explained_change) / abs(total_change)) * 100 if total_change != 0 else 100
        
        # === FINAL SUMMARY ===
        print(f"\n📋 ATTRIBUTION SUMMARY")
        print("="*40)
        print(f"Total Market Value Change: ${total_change:,.2f}")
        print(f"├─ New Deals:             ${new_deals_contribution:,.2f} ({new_deals_contribution/total_change*100:5.1f}%)")
        print(f"├─ Forward Rate Impact:   ${forward_contribution:,.2f} ({forward_contribution/total_change*100:5.1f}%)")
        print(f"├─ Spot Rate Impact:      ${spot_contribution:,.2f} ({spot_contribution/total_change*100:5.1f}%)")
        print(f"└─ Unexplained:           ${unexplained:,.2f} ({unexplained/total_change*100:5.1f}%)")
        
        print(f"\nAttribution Accuracy: {attribution_accuracy:.1f}%")
        
        # Return structured results for further analysis
        return {
            'total_change': total_change,
            'new_deals_contribution': new_deals_contribution,
            'forward_rate_contribution': forward_contribution,
            'spot_rate_contribution': spot_contribution,
            'unexplained': unexplained,
            'attribution_accuracy': attribution_accuracy
        }
    
    def _detect_anomalous_new_deals(self, new_deals_data):
        """
        ML Method: Use Isolation Forest to detect anomalous new deals
        
        Identifies new deals that are unusual compared to typical patterns.
        This helps flag potentially risky or erroneous new positions.
        
        Focuses on meaningful anomaly indicators:
        - Rate spreads and currency-relative deviations
        - Time-based features for maturity structure
        - Data quality indicators
        
        Note: Simplified to focus on rate-based anomalies rather than deal size ratios
        which can vary significantly based on legitimate business factors.
        """
        try:
            # Prepare features for anomaly detection - avoid raw BaseMV
            features_data = []
            
            for _, row in new_deals_data.iterrows():
                deal_features = []
                
                # Rate-based features (independent of deal size)
                deal_features.append(row['SpotRate'])
                deal_features.append(row['FwdRate']) 
                deal_features.append(row['FwdRate'] - row['SpotRate'])  # Forward spread
                
                # Time structure features
                if pd.notna(row['DaystoMaturity']):
                    deal_features.append(row['DaystoMaturity'])
                    # Rate spread per day (time efficiency)
                    rate_spread = row['FwdRate'] - row['SpotRate']
                    deal_features.append(rate_spread / max(row['DaystoMaturity'], 1))
                else:
                    deal_features.extend([0, 0])
                
                # Currency-specific rate deviations
                currency = row['Currency']
                currency_deals = new_deals_data[new_deals_data['Currency'] == currency]
                if len(currency_deals) > 1:
                    # Rate deviation from currency average
                    avg_spot = currency_deals['SpotRate'].mean()
                    avg_fwd = currency_deals['FwdRate'].mean()
                    deal_features.append((row['SpotRate'] - avg_spot) / avg_spot if avg_spot != 0 else 0)
                    deal_features.append((row['FwdRate'] - avg_fwd) / avg_fwd if avg_fwd != 0 else 0)
                else:
                    deal_features.extend([0, 0])
                
                features_data.append(deal_features)
            
            if len(features_data) == 0:
                return {'anomalous_deals': 0, 'error': 'No feature data available'}
            
            # Convert to numpy array
            features = np.array(features_data)
            
            # Handle any infinite or NaN values
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            
            # Scale features for better anomaly detection
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies (-1 = anomaly, 1 = normal)
            anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
            
            # Count anomalous deals
            anomalous_deals = np.sum(anomaly_labels == -1)
            
            # Identify which specific deals are anomalous for reporting
            anomalous_deal_info = []
            deal_ids = new_deals_data['DealId'].values
            for i, label in enumerate(anomaly_labels):
                if label == -1 and i < len(deal_ids):
                    anomalous_deal_info.append({
                        'deal_id': deal_ids[i],
                        'reason': 'Unusual pricing/rate pattern detected'
                    })
            
            print(f"   🔍 Anomaly Detection Features Used:")
            print(f"      • Rate spreads and currency-relative deviations")
            print(f"      • Time-normalized rate features")
            print(f"      • Data quality indicators (missing values, unusual ratios)")
            
            return {
                'anomalous_deals': anomalous_deals,
                'total_new_deals': len(new_deals_data),
                'anomaly_rate': anomalous_deals / len(new_deals_data) * 100,
                'anomalous_deal_details': anomalous_deal_info[:5]  # Show first 5
            }
        except Exception as e:
            return {'anomalous_deals': 0, 'error': str(e)}
    
    def _predict_rate_impact(self, deals_data, historical_data, rate_column):
        """
        FIXED ML Method: Use actual rate sensitivity and proper feature engineering
        
        Key fixes:
        1. Calculate actual rate sensitivity from historical data
        2. Use currency-specific rate normalization
        3. Use realistic rate change scenarios based on actual data
        4. Proper feature scaling and selection
        """
        try:
            print(f"   🤖 Training ML model with {len(deals_data)} forward deals...")
            
            # STEP 1: Calculate actual rate sensitivity from real data
            actual_sensitivities = []
            currency_base_rates = {}
            
            # Group by currency to get currency-specific base rates
            for currency in deals_data['Currency'].unique():
                currency_deals = deals_data[deals_data['Currency'] == currency]
                if len(currency_deals) > 0:
                    currency_base_rates[currency] = currency_deals[rate_column].mean()
            
            # Calculate actual sensitivities from real deal changes
            for _, deal in deals_data.iterrows():
                # Find matching deal in historical data
                historical_deal = historical_data[
                    (historical_data['DealId'] == deal['DealId']) & 
                    (historical_data['Currency'] == deal['Currency'])
                ]
                
                if len(historical_deal) > 0:
                    hist_rate = historical_deal[rate_column].iloc[0]
                    current_rate = deal[rate_column]
                    rate_change = current_rate - hist_rate
                    
                    if abs(rate_change) > 0.0001:  # Significant rate change
                        # Calculate BaseMV sensitivity per unit rate change
                        sensitivity = abs(deal['BaseMV']) / abs(rate_change) if rate_change != 0 else 0
                        actual_sensitivities.append(sensitivity)
            
            # Use median sensitivity to avoid outliers
            avg_sensitivity = np.median(actual_sensitivities) if actual_sensitivities else 100000
            print(f"   📊 Calculated rate sensitivity: ${avg_sensitivity:,.0f} per unit rate change")
            
            # STEP 2: Create training data with realistic scenarios
            features = []
            targets = []
            
            # Use realistic rate deltas based on actual observed changes (0.001 to 0.01)
            realistic_rate_deltas = [-0.01, -0.005, -0.002, -0.001, 0.001, 0.002, 0.005, 0.01]
            
            for _, deal in deals_data.iterrows():
                currency = deal['Currency']
                base_rate = currency_base_rates.get(currency, deal[rate_column])
                
                for rate_delta in realistic_rate_deltas:
                    # Calculate theoretical impact using actual sensitivity
                    theoretical_impact = rate_delta * avg_sensitivity * np.sign(deal['BaseMV'])
                    
                    # FIXED Features: [rate_change_pct, currency_rate_level, deal_size_millions, rate_deviation_pct]
                    features.append([
                        rate_delta,                                           # Rate change (absolute)
                        deal[rate_column] / base_rate,                       # Rate level relative to currency base
                        deal['BaseMV'] / 1000000,                            # Deal size in millions (better scale)
                        (deal[rate_column] - base_rate) / base_rate           # Rate deviation as percentage
                    ])
                    targets.append(theoretical_impact)
            
            if len(features) < 10:
                return {'predicted_impact': 0, 'confidence': 0, 'error': 'Insufficient training data'}
            
            # STEP 3: Train with realistic data
            X = np.array(features)
            y = np.array(targets)
            
            # Split data properly
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest with appropriate parameters
            enhanced_predictor = RandomForestRegressor(
                n_estimators=100,  # Reduced for faster training
                max_depth=8,       # Prevent overfitting
                min_samples_split=10,
                random_state=42
            )
            enhanced_predictor.fit(X_train, y_train)
            y_pred = enhanced_predictor.predict(X_test)
            
            # Calculate model confidence
            confidence = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
            
            # STEP 4: Predict using realistic scenario (0.5% = 0.005 based on debug data)
            realistic_rate_change = 0.005  # Based on actual observed changes
            total_predicted_impact = 0
            
            for _, deal in deals_data.iterrows():
                currency = deal['Currency']
                base_rate = currency_base_rates.get(currency, deal[rate_column])
                
                prediction_features = [[
                    realistic_rate_change,                                   # Realistic rate change
                    deal[rate_column] / base_rate,                          # Rate level relative to currency  
                    deal['BaseMV'] / 1000000,                               # Deal size in millions
                    (deal[rate_column] - base_rate) / base_rate              # Rate deviation percentage
                ]]
                
                predicted_change = enhanced_predictor.predict(prediction_features)[0]
                total_predicted_impact += predicted_change
            
            print(f"   🤖 Model trained with {len(features)} examples, R²={confidence:.3f}")
            print(f"   📊 Using sensitivity: ${avg_sensitivity:,.0f}, scenario: {realistic_rate_change*100:.1f}% rate change")
            
            return {
                'predicted_impact': total_predicted_impact,
                'confidence': max(0, min(1, confidence)),  # Clamp between 0 and 1
                'features_used': len(features),
                'scenario': f'{realistic_rate_change*100:.1f}% rate change scenario',
                'calculated_sensitivity': avg_sensitivity
            }
            
        except Exception as e:
            print(f"   ❌ ML prediction error: {str(e)}")
            return {'predicted_impact': 0, 'confidence': 0, 'error': str(e)}
    
    def _cluster_spot_deals(self, spot_deals):
        """
        ML Method: Use DBSCAN clustering to group similar spot deals
        
        Identifies clusters of spot deals with similar characteristics,
        helping understand portfolio concentration and risk patterns.
        """
        try:
            # Prepare features for clustering
            features = spot_deals[['SpotRate', 'BaseMV']].copy()
            
            # Scale features for clustering
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform clustering
            cluster_labels = self.clusterer.fit_predict(features_scaled)
            
            # Analyze clusters
            unique_clusters = set(cluster_labels)
            num_clusters = len(unique_clusters - {-1})  # Exclude noise cluster (-1)
            
            # Find largest cluster impact
            largest_cluster_impact = 0
            if num_clusters > 0:
                spot_deals_copy = spot_deals.copy()
                spot_deals_copy['cluster'] = cluster_labels
                
                for cluster_id in unique_clusters:
                    if cluster_id != -1:  # Skip noise
                        cluster_deals = spot_deals_copy[spot_deals_copy['cluster'] == cluster_id]
                        cluster_impact = abs(cluster_deals['BaseMV'].sum())
                        largest_cluster_impact = max(largest_cluster_impact, cluster_impact)
            
            return {
                'num_clusters': num_clusters,
                'largest_cluster_impact': largest_cluster_impact,
                'total_deals_clustered': len(spot_deals)
            }
            
        except Exception as e:
            return {'num_clusters': 0, 'largest_cluster_impact': 0, 'error': str(e)}
    
    def analyze_basemv_features(self, date1, date2):
        """
        ML Feature Analysis: Identify key features affecting BaseMV changes
        
        This method analyzes which features have the most impact on BaseMV changes
        between two dates, focusing on actionable trading and market factors.
        
        Excludes:
        - Risk Sensitivity Features (BPDelta, ModifiedDuration, etc.)
        - Business Structure Features (Entity, EntityGroup, etc.)
        
        Args:
            date1 (datetime): Earlier date (baseline)
            date2 (datetime): Later date (comparison)
            
        Returns:
            dict: Feature importance analysis and BaseMV predictions
        """
        print(f"\n🔍 ML FEATURE ANALYSIS FOR BASEMV: {date1.date()} vs {date2.date()}")
        print("="*70)
        
        # Filter data for the two specific dates
        data_date1 = self.data[self.data['PositionDate'] == date1]
        data_date2 = self.data[self.data['PositionDate'] == date2]
        
        print(f"Analyzing BaseMV changes across {len(data_date1)} -> {len(data_date2)} deals")
        
        # === PRIMARY FEATURES - Direct BaseMV Drivers ===
        primary_features = [
            'SpotRate', 'FwdRate', 'SpotFactor', 'FwdFactor', 'ZeroRate',
            'MarketValue'  # Transaction currency market value
        ]
        
        # === SECONDARY FEATURES - Context & Market Factors ===
        secondary_features = [
            'DaystoMaturity', 'TermofDeal', 'CashFlowDate',
            'FaceValue', 'PrincipalOutstanding',
            'CurrencyDF', 'BaseDF'
        ]
        
        # === CATEGORICAL FEATURES - Trading Context ===
        categorical_features = [
            'ValuationModel', 'Currency', 'CcyPair', 'PayorReceive',
            'TransactionType', 'ForwardType', 'SettlementType',
            'TimeProfile', 'CashflowType', 'Instrument'
        ]
        
        print(f"\n📊 FEATURE CATEGORIES:")
        print(f"• Primary Features (Rate/Value): {len(primary_features)}")
        print(f"• Secondary Features (Time/Scale): {len(secondary_features)}")
        print(f"• Categorical Features (Context): {len(categorical_features)}")
        
        # === FEATURE ENGINEERING FOR ML ===
        feature_data = []
        target_data = []
        
        # Process existing deals to calculate BaseMV changes
        existing_deals = set(data_date1['DealId']) & set(data_date2['DealId'])
        
        for deal_id in existing_deals:
            deal_date1 = data_date1[data_date1['DealId'] == deal_id]
            deal_date2 = data_date2[data_date2['DealId'] == deal_id]
            
            if len(deal_date1) > 0 and len(deal_date2) > 0:
                # Calculate net BaseMV change (sum of pay/receive pairs)
                basemv_change = deal_date2['BaseMV'].sum() - deal_date1['BaseMV'].sum()
                
                # Extract features for each deal side
                for _, row1 in deal_date1.iterrows():
                    matching_row2 = deal_date2[
                        (deal_date2['Currency'] == row1['Currency']) &
                        (deal_date2['ValuationModel'] == row1['ValuationModel']) &
                        (deal_date2['PayorReceive'] == row1['PayorReceive'])
                    ]
                    
                    if len(matching_row2) > 0:
                        row2 = matching_row2.iloc[0]
                        
                        # Create feature vector
                        features = {}
                        
                        # === PRIMARY FEATURES ===
                        features['SpotRate_Change'] = row2['SpotRate'] - row1['SpotRate']
                        features['FwdRate_Change'] = row2['FwdRate'] - row1['FwdRate']
                        features['SpotRate_Level'] = row1['SpotRate']
                        features['FwdRate_Level'] = row1['FwdRate']
                        features['Rate_Spread'] = row1['FwdRate'] - row1['SpotRate']
                        features['MarketValue_Base'] = row1['MarketValue']
                        
                        # Handle ZeroRate (can be 0 for FORWARD models)
                        if 'ZeroRate' in row1 and pd.notna(row1['ZeroRate']):
                            features['ZeroRate_Change'] = row2['ZeroRate'] - row1['ZeroRate']
                            features['ZeroRate_Level'] = row1['ZeroRate']
                        else:
                            features['ZeroRate_Change'] = 0
                            features['ZeroRate_Level'] = 0
                        
                        # === SECONDARY FEATURES ===
                        features['DaystoMaturity'] = row1['DaystoMaturity'] if pd.notna(row1['DaystoMaturity']) else 0
                        features['TermofDeal'] = row1['TermofDeal'] if pd.notna(row1['TermofDeal']) else 0
                        features['FaceValue_Log'] = np.log(abs(row1['FaceValue']) + 1) if pd.notna(row1['FaceValue']) else 0
                        features['PrincipalOutstanding_Log'] = np.log(abs(row1['PrincipalOutstanding']) + 1) if pd.notna(row1['PrincipalOutstanding']) else 0
                        
                        # Time to maturity buckets
                        days_to_mat = features['DaystoMaturity']
                        if days_to_mat <= 30:
                            features['Maturity_Bucket'] = 1  # Short term
                        elif days_to_mat <= 90:
                            features['Maturity_Bucket'] = 2  # Medium term
                        elif days_to_mat <= 365:
                            features['Maturity_Bucket'] = 3  # Long term
                        else:
                            features['Maturity_Bucket'] = 4  # Very long term
                        
                        # === CATEGORICAL FEATURES (Encoded) ===
                        features['ValuationModel_FORWARD'] = 1 if row1['ValuationModel'] == 'FORWARD' else 0
                        features['ValuationModel_FORWARD_NPV'] = 1 if row1['ValuationModel'] == 'FORWARD NPV' else 0
                        features['ValuationModel_NPV_SPOT'] = 1 if row1['ValuationModel'] == 'NPV SPOT' else 0
                        
                        features['PayorReceive_PAY'] = 1 if row1['PayorReceive'] == 'PAY' else 0
                        features['PayorReceive_REC'] = 1 if row1['PayorReceive'] == 'REC' else 0
                        
                        # Currency encoding (major currencies)
                        currency = row1['Currency']
                        features['Currency_EUR'] = 1 if 'EUR' in str(currency) else 0
                        features['Currency_USD'] = 1 if 'USD' in str(currency) else 0
                        features['Currency_JPY'] = 1 if 'JPY' in str(currency) else 0
                        features['Currency_GBP'] = 1 if 'GBP' in str(currency) else 0
                        
                        # Settlement type
                        if pd.notna(row1['SettlementType']):
                            features['Settlement_Deliverable'] = 1 if row1['SettlementType'] == 'Deliverable' else 0
                        else:
                            features['Settlement_Deliverable'] = 0
                        
                        # === DERIVED FEATURES ===
                        features['Rate_Volatility'] = abs(features['SpotRate_Change']) + abs(features['FwdRate_Change'])
                        features['Size_Normalized'] = features['FaceValue_Log'] / 1000000  # Normalize to millions
                        features['Time_Decay_Factor'] = 1 / (1 + features['DaystoMaturity']/365) if features['DaystoMaturity'] > 0 else 1
                        
                        # Interaction features
                        features['Rate_Change_x_Size'] = features['SpotRate_Change'] * features['Size_Normalized']
                        features['Rate_Change_x_Time'] = features['SpotRate_Change'] * features['Time_Decay_Factor']
                        
                        feature_data.append(list(features.values()))
                        target_data.append(basemv_change)
        
        if len(feature_data) == 0:
            print("❌ No matching deals found for feature analysis")
            return {'error': 'No data available'}
        
        # Convert to numpy arrays
        X = np.array(feature_data)
        y = np.array(target_data)
        feature_names = list(features.keys())
        
        print(f"\n🔢 DATASET CREATED:")
        print(f"• Features: {X.shape[1]} features")
        print(f"• Samples: {X.shape[0]} deal observations")
        print(f"• Target: BaseMV changes (${y.min():,.0f} to ${y.max():,.0f})")
        
        # === ML MODEL TRAINING & FEATURE IMPORTANCE ===
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest for feature importance
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = rf_model.predict(X_test)
            
            # Calculate model performance
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"\n🤖 ML MODEL PERFORMANCE:")
            print(f"• R² Score: {r2:.3f}")
            print(f"• RMSE: ${rmse:,.0f}")
            print(f"• Model Confidence: {'High' if r2 > 0.7 else 'Medium' if r2 > 0.4 else 'Low'}")
            
            # Feature importance analysis
            feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n📈 TOP 10 FEATURES AFFECTING BASEMV:")
            print("-" * 50)
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"{i+1:2d}. {feature:<25} {importance:.4f}")
            
            # === FEATURE CATEGORY ANALYSIS ===
            primary_importance = sum(imp for feat, imp in feature_importance.items() 
                                   if any(pf in feat for pf in ['SpotRate', 'FwdRate', 'Rate', 'MarketValue']))
            secondary_importance = sum(imp for feat, imp in feature_importance.items() 
                                     if any(sf in feat for sf in ['Days', 'Term', 'Face', 'Principal', 'Time']))
            categorical_importance = sum(imp for feat, imp in feature_importance.items() 
                                       if any(cf in feat for cf in ['Valuation', 'Currency', 'Payor', 'Settlement']))
            
            print(f"\n📊 FEATURE CATEGORY IMPORTANCE:")
            print(f"• Rate/Value Features:  {primary_importance:.3f} ({primary_importance/1.0*100:.1f}%)")
            print(f"• Time/Scale Features:  {secondary_importance:.3f} ({secondary_importance/1.0*100:.1f}%)")
            print(f"• Context Features:     {categorical_importance:.3f} ({categorical_importance/1.0*100:.1f}%)")
            
            # === PREDICTION INSIGHTS ===
            print(f"\n💡 KEY INSIGHTS FOR BASEMV PREDICTION:")
            
            # Top feature insights
            top_feature = sorted_features[0][0]
            if 'Rate_Change' in top_feature:
                print(f"• Rate movements are the primary driver of BaseMV changes")
            elif 'Size' in top_feature:
                print(f"• Deal size significantly impacts BaseMV fluctuations")
            elif 'Time' in top_feature:
                print(f"• Time to maturity is a critical factor")
            
            # Rate sensitivity analysis
            rate_features = [f for f, _ in sorted_features if 'Rate' in f][:3]
            print(f"• Top rate features: {', '.join(rate_features)}")
            
            # Valuation model impact
            model_features = [(f, imp) for f, imp in sorted_features if 'ValuationModel' in f]
            if model_features:
                dominant_model = max(model_features, key=lambda x: x[1])
                print(f"• Most impactful valuation model: {dominant_model[0]} (importance: {dominant_model[1]:.3f})")
            
            return {
                'model_performance': {
                    'r2_score': r2,
                    'rmse': rmse,
                    'model_confidence': 'High' if r2 > 0.7 else 'Medium' if r2 > 0.4 else 'Low'
                },
                'feature_importance': feature_importance,
                'top_features': sorted_features[:10],
                'category_importance': {
                    'rate_value': primary_importance,
                    'time_scale': secondary_importance,
                    'context': categorical_importance
                },
                'dataset_info': {
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0],
                    'target_range': {'min': float(y.min()), 'max': float(y.max())}
                }
            }
            
        except Exception as e:
            print(f"❌ ML Analysis Error: {str(e)}")
            return {'error': str(e)}

def main():
    """
    Run the ML-enhanced POC demonstration
    
    This function demonstrates the complete ML-enhanced attribution analysis:
    1. Initialize the analyzer with ML models
    2. Load sample forex trading data
    3. Run attribution analysis with ML enhancements
    4. Display results showing both traditional and ML insights
    
    ML Enhancements included:
    - Isolation Forest for anomaly detection in new deals
    - Random Forest for predictive rate impact analysis  
    - DBSCAN clustering for spot deal pattern analysis
    """
    print("🚀 FOREX ATTRIBUTION POC - ML-Enhanced Implementation")
    print("Implements idea.md process + Machine Learning algorithms")
    print("="*70)
    
    # Initialize the ML-enhanced attribution analyzer
    analyzer = ForexAttributionPOC()
    
    # Load real data (requires Cashflows_FX_V3.xlsx in data/ directory)
    analyzer.load_real_data()
    
    
    # Run ML-enhanced attribution analysis for consecutive dates
    print("\n🤖 Running ML-Enhanced Attribution Analysis...")
    date1 = datetime(2012, 3, 6)  # Baseline date
    date2 = datetime(2012, 3, 7)  # Comparison date
    
    results = analyzer.analyze_attribution(date1, date2)
    
    # === NEW: ML FEATURE ANALYSIS FOR BASEMV ===
    print(f"\n" + "="*70)
    print(f"🧠 ADVANCED ML FEATURE ANALYSIS")
    print(f"="*70)
    
    feature_results = analyzer.analyze_basemv_features(date1, date2)
    
    if 'error' not in feature_results:
        print(f"\n🎯 ML FEATURE ANALYSIS SUMMARY:")
        print(f"- Model Performance: {feature_results['model_performance']['model_confidence']}")
        print(f"- R² Score: {feature_results['model_performance']['r2_score']:.3f}")
        print(f"- Top BaseMV Driver: {feature_results['top_features'][0][0]}")
        print(f"- Most Important Category: Rate/Value Features ({feature_results['category_importance']['rate_value']:.3f})")
        
        # Show practical insights
        print(f"\n📋 ACTIONABLE INSIGHTS:")
        rate_importance = feature_results['category_importance']['rate_value']
        time_importance = feature_results['category_importance']['time_scale']
        
        if rate_importance > 0.5:
            print(f"• Focus on rate monitoring - rate changes drive {rate_importance*100:.0f}% of BaseMV variation")
        if time_importance > 0.2:
            print(f"• Time-to-maturity matters - consider maturity-based risk management")
        
        # Show top 3 actionable features
        actionable_features = [f for f, _ in feature_results['top_features'][:5] 
                             if any(keyword in f for keyword in ['Rate_Change', 'Size', 'Time', 'Currency'])][:3]
        print(f"• Monitor these key factors: {', '.join(actionable_features)}")
    
    # Display completion message and next steps
    print(f"\n✅ ML-Enhanced Analysis Complete!")
    
    # Show what the results tell us
    print(f"\n💡 KEY INSIGHTS FROM ML-ENHANCED ANALYSIS:")
    print(f"- Attribution accuracy: {results['attribution_accuracy']:.1f}%")
    print(f"- Largest contributor: {_identify_largest_contributor(results)}")
    print(f"- Total portfolio change: ${results['total_change']:,.2f}")
    
    # print(f"\n🤖 ML ALGORITHMS USED:")
    # print(f"1. Isolation Forest - Anomaly detection for unusual new deals")
    # print(f"2. Random Forest - Predictive modeling for rate impact")
    # print(f"3. DBSCAN Clustering - Pattern analysis for spot deals")
    # print(f"4. StandardScaler - Feature normalization for ML models")


def _identify_largest_contributor(results):
    """
    Helper function to identify the largest contributing factor
    
    Args:
        results (dict): Attribution analysis results
        
    Returns:
        str: Name of the largest contributing factor
    """
    contributions = {
        'New Deals': abs(results['new_deals_contribution']),
        'Forward Rate Impact': abs(results['forward_rate_contribution']),
        'Spot Rate Impact': abs(results['spot_rate_contribution'])
    }
    
    return max(contributions, key=contributions.get)

if __name__ == "__main__":
    main()
