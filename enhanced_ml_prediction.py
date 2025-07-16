"""
Enhanced ML Forward Rate Impact Prediction
Using Actual Financial Risk Metrics from the Dataset

This module uses proper financial risk measures for ML prediction:
- ModifiedDuration: Standard rate sensitivity coefficient
- DaystoMaturity: Time factor affecting sensitivity
- IRR: Internal Rate of Return (yield/return measure)
- ZeroRate: Interest rate environment
- FwdRate: Current forward rate level

This should dramatically improve prediction accuracy compared to 
synthetic rate sensitivity calculations.
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

class EnhancedForexMLPredictor:
    """
    Enhanced ML predictor using actual financial risk metrics
    
    Uses the following features for forward rate impact prediction:
    1. ModifiedDuration - Rate sensitivity measure
    2. DaystoMaturity - Time factor
    3. IRR - Internal Rate of Return (yield/return measure)
    4. ZeroRate - Interest rate environment
    5. FwdRate - Current rate level
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.predictor = None
        self.feature_importance = None
        self.model_metrics = {}
        
        # Define the enhanced feature set (excluding redundant features)
        self.feature_columns = [
            'ModifiedDuration', # Standard rate sensitivity measure
            'DaystoMaturity',   # Time factor
            'IRR',              # Internal Rate of Return - yield/return measure
            'ZeroRate',         # Interest rate environment
            'FwdRate'           # Current forward rate level
        ]
    
    def load_and_prepare_data(self, file_path='data/Cashflows_FX_V3.xlsx'):
        """
        Load and prepare the forex data with enhanced features
        """
        print("📊 Loading enhanced dataset with financial risk metrics...")
        
        try:
            self.data = pd.read_excel(file_path)
            print(f"✓ Loaded {len(self.data)} records")
            
            # Check availability of enhanced features
            available_features = []
            missing_features = []
            
            for feature in self.feature_columns:
                if feature in self.data.columns:
                    available_features.append(feature)
                else:
                    missing_features.append(feature)
            
            print(f"✓ Available features: {len(available_features)}/{len(self.feature_columns)}")
            print(f"  Available: {available_features}")
            if missing_features:
                print(f"  Missing: {missing_features}")
            
            # Update feature columns to only include available ones
            self.feature_columns = available_features
            
            # Display data quality for key features
            print(f"\n📈 Data Quality Check:")
            for feature in self.feature_columns:
                non_null = self.data[feature].notna().sum()
                total = len(self.data)
                print(f"  {feature}: {non_null}/{total} ({non_null/total*100:.1f}% complete)")
                
                # Show value ranges for key features
                if feature in ['ModifiedDuration', 'DaystoMaturity']:
                    non_zero_data = self.data[self.data[feature] != 0][feature]
                    if len(non_zero_data) > 0:
                        print(f"    Range: {non_zero_data.min():.4f} to {non_zero_data.max():.4f}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_training_data(self, date1, date2):
        """
        Prepare training data using actual rate changes and financial metrics
        
        This method:
        1. Gets deals that exist on both dates
        2. Calculates actual rate changes
        3. Calculates actual market value changes
        4. Uses financial risk metrics as features
        """
        print(f"\n🔧 Preparing Enhanced Training Data: {date1.date()} -> {date2.date()}")
        
        # Filter data for the two dates
        data_date1 = self.data[self.data['PositionDate'] == date1]
        data_date2 = self.data[self.data['PositionDate'] == date2]
        
        # Get existing deals (present on both dates)
        existing_deals = set(data_date1['DealId'])
        existing_deals_date2 = data_date2[data_date2['DealId'].isin(existing_deals)]
        
        # Filter for forward deals only (since we're predicting forward rate impact)
        forward_deals_date2 = existing_deals_date2[
            existing_deals_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV'])
        ]
        
        print(f"Forward deals for training: {len(forward_deals_date2)}")
        
        # Prepare training examples
        features = []
        targets = []
        deal_info = []
        
        # For each forward deal, create training example
        for _, deal in forward_deals_date2.iterrows():
            # Find corresponding deal on date1
            historical_deal = data_date1[
                (data_date1['DealId'] == deal['DealId']) & 
                (data_date1['Currency'] == deal['Currency'])
            ]
            
            if len(historical_deal) > 0:
                hist_deal = historical_deal.iloc[0]
                
                # Calculate actual rate change
                rate_change = deal['FwdRate'] - hist_deal['FwdRate']
                
                # Calculate actual market value change (net for pay/receive pairs)
                # Group by DealId to handle pay/receive pairs
                deal_mv_date1 = data_date1[data_date1['DealId'] == deal['DealId']]['BaseMV'].sum()
                deal_mv_date2 = data_date2[data_date2['DealId'] == deal['DealId']]['BaseMV'].sum()
                mv_change = deal_mv_date2 - deal_mv_date1
                
                # Only include deals with significant rate changes
                if abs(rate_change) > 0.0001:
                    # Prepare features using available financial metrics
                    feature_vector = []
                    
                    for feature in self.feature_columns:
                        if feature in deal:
                            value = deal[feature]
                            # Handle missing values
                            if pd.isna(value):
                                value = 0
                            feature_vector.append(value)
                        else:
                            feature_vector.append(0)
                    
                    features.append(feature_vector)
                    targets.append(mv_change)
                    deal_info.append({
                        'DealId': deal['DealId'],
                        'Currency': deal['Currency'],
                        'RateChange': rate_change,
                        'MVChange': mv_change,
                        'ModifiedDuration': deal.get('ModifiedDuration', 0)
                    })
        
        # print(f"✓ Created {len(features)} training examples with actual rate/MV changes")
        
        # Display sample training data with feature contributions
        if len(deal_info) > 0:
            # print(f"\n📊 Sample Training Examples with Feature Analysis:")
            for i, info in enumerate(deal_info[:5]):
                deal_features = features[i]
                # print(f"  Deal {info['DealId']} ({info['Currency']}):")
                # print(f"    Rate Change: {info['RateChange']:.6f}")
                # print(f"    BaseMV Change: ${info['MVChange']:,.2f}")
                # print(f"    Feature Values:")
                # for j, feature in enumerate(self.feature_columns):
                    # print(f"      {feature}: {deal_features[j]:,.2f}")
                    
        
        # Analyze feature correlations with BaseMV changes
        if len(features) > 0:
            self._analyze_feature_contributions(features, targets, deal_info)
        
        return np.array(features), np.array(targets), deal_info
    
    def train_enhanced_model(self, features, targets, deal_info):
        """
        Train enhanced ML model using financial risk metrics
        """
        print(f"\n🤖 Training Enhanced ML Model...")
        
        if len(features) < 10:
            print("❌ Insufficient training data")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models and select the best one
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"  {name}:")
            print(f"    R² Score: {r2:.4f}")
            print(f"    MAE: ${mae:,.2f}")
            print(f"    RMSE: ${rmse:,.2f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
        
        self.predictor = best_model
        
        # Store feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_columns, best_model.feature_importances_))
            
            print(f"\n📊 Feature Importance ({best_name}):")
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.4f}")
        
        # Store model metrics
        y_pred_final = best_model.predict(X_test_scaled)
        self.model_metrics = {
            'model_name': best_name,
            'r2_score': r2_score(y_test, y_pred_final),
            'mae': mean_absolute_error(y_test, y_pred_final),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_final)),
            'training_samples': len(features),
            'test_samples': len(y_test)
        }
        
        print(f"\n✅ Best Model: {best_name} (R² = {best_score:.4f})")
        return self.predictor
    
    def predict_rate_impact(self, deals_data, rate_change_scenario=0.005):
        """
        Predict forward rate impact using enhanced financial metrics
        """
        if self.predictor is None:
            return {'error': 'Model not trained'}
        
        print(f"\n🔮 Predicting Rate Impact (Scenario: {rate_change_scenario*100:.1f}% rate change)")
        
        total_predicted_impact = 0
        predictions_detail = []
        
        # Group by DealId to handle pay/receive pairs
        unique_deals = deals_data['DealId'].unique()
        
        for deal_id in unique_deals:
            deal_rows = deals_data[deals_data['DealId'] == deal_id]
            
            # Use first row for features (both pay/receive have same deal characteristics)
            deal = deal_rows.iloc[0]
            
            # Prepare features
            feature_vector = []
            for feature in self.feature_columns:
                if feature in deal:
                    value = deal[feature]
                    if pd.isna(value):
                        value = 0
                    feature_vector.append(value)
                else:
                    feature_vector.append(0)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict
            predicted_change = self.predictor.predict(feature_vector_scaled)[0]
            total_predicted_impact += predicted_change
            
            predictions_detail.append({
                'DealId': deal_id,
                'Currency': deal['Currency'],
                'PredictedChange': predicted_change,
                'ModifiedDuration': deal.get('ModifiedDuration', 0),
                'DaystoMaturity': deal.get('DaystoMaturity', 0)
            })
        
        # Show sample predictions with feature breakdown
        # print(f"Sample Predictions with Feature Analysis:")
        # for i, pred in enumerate(predictions_detail[:3]):
        #     print(f"  Deal {pred['DealId']} ({pred['Currency']}):")
        #     print(f"    Predicted Change: ${pred['PredictedChange']:,.2f}")
        #     print(f"    Key Features:")
        #     print(f"      ModifiedDuration: {pred['ModifiedDuration']:.4f}")
        #     print(f"      DaystoMaturity: {pred['DaystoMaturity']:.0f}")
        
        # Calculate feature contribution to total prediction
        if self.feature_importance is not None:
            print(f"\n🎯 Feature Contribution to Total Prediction:")
            total_feature_contributions = {}
            
            for deal in predictions_detail:
                # Get feature values for this deal
                deal_data = deals_data[deals_data['DealId'] == deal['DealId']].iloc[0]
                
                for feature in self.feature_columns:
                    if feature not in total_feature_contributions:
                        total_feature_contributions[feature] = 0
                    
                    feature_value = deal_data.get(feature, 0)
                    if pd.isna(feature_value):
                        feature_value = 0
                    
                    # Approximate contribution = feature_value * importance * prediction_magnitude
                    if feature in self.feature_importance:
                        contribution = feature_value * self.feature_importance[feature] * 0.01  # Scaled approximation
                        total_feature_contributions[feature] += contribution
            
            # Display sorted contributions
            sorted_contributions = sorted(total_feature_contributions.items(), 
                                        key=lambda x: abs(x[1]), reverse=True)
            
            for feature, contribution in sorted_contributions:
                importance = self.feature_importance.get(feature, 0)
                print(f"  {feature:18}: ${contribution:12,.2f} (Importance: {importance:.4f})")
        
        
        return {
            'total_predicted_impact': total_predicted_impact,
            'model_metrics': self.model_metrics,
            'feature_importance': self.feature_importance,
            'predictions_detail': predictions_detail,
            'scenario': f'{rate_change_scenario*100:.1f}% rate change'
        }
    
    def analyze_actual_vs_predicted(self, date1, date2):
        """
        Complete analysis: train model and compare with actual results
        """
        print(f"\n🎯 ENHANCED ML ANALYSIS: {date1.date()} vs {date2.date()}")
        print("="*70)
        
        # Prepare training data
        features, targets, deal_info = self.prepare_training_data(date1, date2)

        if len(features) == 0:
            return {'error': 'No training data available'}
        
        # Train model
        model = self.train_enhanced_model(features, targets, deal_info)
        
        if model is None:
            return {'error': 'Model training failed'}
        
        # Get forward deals for prediction
        data_date2 = self.data[self.data['PositionDate'] == date2]
        existing_deals = set(self.data[self.data['PositionDate'] == date1]['DealId'])
        forward_deals = data_date2[
            (data_date2['DealId'].isin(existing_deals)) &
            (data_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV']))
        ]
        
        # Predict impact
        prediction_results = self.predict_rate_impact(forward_deals)
        
        # Calculate actual forward rate impact (from the main attribution analysis)
        # This would need to be passed in or calculated here
        # For now, we'll return the prediction results
        
        return prediction_results

    def _analyze_feature_contributions(self, features, targets, deal_info):
        """
        Analyze the contribution of each feature column to BaseMV changes
        """
        print(f"\n🔍 FEATURE CONTRIBUTION ANALYSIS")
        print("="*60)
        
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        print(f"Analyzing {len(features)} deals with BaseMV changes...")
        
        # Calculate correlations between each feature and BaseMV changes
        correlations = {}
        for i, feature in enumerate(self.feature_columns):
            feature_values = features_array[:, i]
            
            # Filter out zero values for better correlation
            non_zero_mask = feature_values != 0
            if np.sum(non_zero_mask) > 5:  # Need at least 5 non-zero values
                correlation = np.corrcoef(feature_values[non_zero_mask], 
                                        targets_array[non_zero_mask])[0, 1]
                correlations[feature] = correlation if not np.isnan(correlation) else 0
            else:
                correlations[feature] = 0
        
        # Display correlation analysis
        print(f"\n📈 Feature Correlations with BaseMV Changes:")
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, corr in sorted_correlations:
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            print(f"  {feature:18}: {corr:7.4f} ({strength} {direction})")
        
        # Analyze feature value ranges and their impact
        print(f"\n📊 Feature Value Ranges and Impact Analysis:")
        for i, feature in enumerate(self.feature_columns):
            feature_values = features_array[:, i]
            non_zero_values = feature_values[feature_values != 0]
            
            if len(non_zero_values) > 0:
                min_val = np.min(non_zero_values)
                max_val = np.max(non_zero_values)
                mean_val = np.mean(non_zero_values)
                std_val = np.std(non_zero_values)
                
                # print(f"  {feature:18}:")
                # print(f"    Range: {min_val:12,.2f} to {max_val:12,.2f}")
                # print(f"    Mean:  {mean_val:12,.2f} ± {std_val:8,.2f}")
                # print(f"    Non-zero deals: {len(non_zero_values)}/{len(feature_values)} ({len(non_zero_values)/len(feature_values)*100:.1f}%)")
        
        # Calculate theoretical vs actual BaseMV changes for key features
        print(f"\n🎯 KEY FEATURE IMPACT ANALYSIS:")
        
        # Analyze ModifiedDuration impact
        if 'ModifiedDuration' in self.feature_columns:
            mod_dur_idx = self.feature_columns.index('ModifiedDuration')
            mod_dur_values = features_array[:, mod_dur_idx]
            basemv_idx = self.feature_columns.index('BaseMV') if 'BaseMV' in self.feature_columns else None
            
            mod_dur_deals = [(i, val, targets_array[i]) for i, val in enumerate(mod_dur_values) if abs(val) > 0.01]
            
            if mod_dur_deals and basemv_idx is not None:
                print(f"\n  ModifiedDuration Analysis ({len(mod_dur_deals)} deals):")
                for i, (deal_idx, mod_dur, mv_change) in enumerate(mod_dur_deals[:5]):
                    deal_id = deal_info[deal_idx]['DealId']
                    rate_change = deal_info[deal_idx]['RateChange']
                    current_basemv = features_array[deal_idx, basemv_idx]
                    
                    # ModifiedDuration formula: % change in value = -ModifiedDuration × rate change
                    expected_pct_change = -mod_dur * rate_change
                    expected_mv_change = current_basemv * expected_pct_change
                    difference = mv_change - expected_mv_change
                    
                    print(f"    Deal {deal_id}:")
                    print(f"      ModifiedDuration: {mod_dur:8.4f}")
                    print(f"      Current BaseMV: ${current_basemv:12,.2f}")
                    print(f"      Rate Change: {rate_change:8.6f}")
                    print(f"      Expected MV Δ: ${expected_mv_change:12,.2f}")
                    print(f"      Actual MV Δ:   ${mv_change:12,.2f}")
                    print(f"      Difference:    ${difference:12,.2f}")
        
        return correlations

    def explain_market_value_changes(self, date1, date2):
        """
        Detailed explanation of why market values rose or fell
        
        This method provides comprehensive analysis of:
        1. Which deals had the largest MV changes (positive/negative)
        2. What features drove each change
        3. Root cause analysis of market value movements
        4. Quantified impact of each feature
        """
        print(f"\n📈 MARKET VALUE CHANGE EXPLANATION: {date1.date()} -> {date2.date()}")
        print("="*80)
        
        # Get training data with detailed feature analysis
        features, targets, deal_info = self.prepare_training_data(date1, date2)
        
        if len(features) == 0:
            print("❌ No data available for analysis")
            return
        
        # Convert to arrays for analysis
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        # Analyze positive and negative changes separately
        positive_changes = [(i, target) for i, target in enumerate(targets_array) if target > 0]
        negative_changes = [(i, target) for i, target in enumerate(targets_array) if target < 0]
        
        print(f"📊 OVERALL SUMMARY:")
        print(f"  Total deals analyzed: {len(targets_array)}")
        print(f"  Deals with gains: {len(positive_changes)} (Total: ${sum([x[1] for x in positive_changes]):,.2f})")
        print(f"  Deals with losses: {len(negative_changes)} (Total: ${sum([x[1] for x in negative_changes]):,.2f})")
        print(f"  Net change: ${targets_array.sum():,.2f}")
        
        # Analyze top gainers and losers
        print(f"\n🚀 TOP 5 BIGGEST GAINS:")
        top_gains = sorted(positive_changes, key=lambda x: x[1], reverse=True)[:5]
        self._explain_individual_changes(top_gains, features_array, deal_info, "GAIN")
        
        print(f"\n📉 TOP 5 BIGGEST LOSSES:")
        top_losses = sorted(negative_changes, key=lambda x: x[1])[:5]
        self._explain_individual_changes(top_losses, features_array, deal_info, "LOSS")
        
        # Feature-based analysis
        print(f"\n🔍 FEATURE-BASED IMPACT ANALYSIS:")
        self._analyze_feature_impact_on_changes(features_array, targets_array, deal_info)
        
        # Currency-based analysis
        print(f"\n🌍 CURRENCY-BASED ANALYSIS:")
        self._analyze_currency_impact(deal_info, targets_array)
        
        # Market conditions analysis
        print(f"\n📊 MARKET CONDITIONS ANALYSIS:")
        self._analyze_market_conditions(features_array, targets_array, deal_info)
        
        return {
            'total_gains': sum([x[1] for x in positive_changes]),
            'total_losses': sum([x[1] for x in negative_changes]),
            'net_change': targets_array.sum(),
            'top_gains': top_gains,
            'top_losses': top_losses
        }
    
    def _explain_individual_changes(self, changes, features_array, deal_info, change_type):
        """
        Explain individual deal changes with feature attribution
        """
        for i, (deal_idx, mv_change) in enumerate(changes):
            deal_features = features_array[deal_idx]
            deal = deal_info[deal_idx]
            
            print(f"\n  {i+1}. Deal {deal['DealId']} ({deal['Currency']}) - {change_type}: ${mv_change:,.2f}")
            print(f"     Rate Change: {deal['RateChange']:.6f} ({deal['RateChange']*10000:+.1f} bps)")
            
            # Analyze key contributing features
            feature_contributions = []
            
            for j, feature in enumerate(self.feature_columns):
                feature_value = deal_features[j]
                
                if feature == 'ModifiedDuration' and abs(feature_value) > 0.01:
                    # Modified Duration analysis
                    current_mv = deal_features[self.feature_columns.index('BaseMV')] if 'BaseMV' in self.feature_columns else 0
                    expected_change = -feature_value * deal['RateChange'] * current_mv
                    contribution_pct = (expected_change / mv_change) * 100 if mv_change != 0 else 0
                    feature_contributions.append((feature, feature_value, expected_change, contribution_pct))
                    
                elif feature == 'FaceValue':
                    # Face value impact (deal size effect)
                    size_factor = "Large" if abs(feature_value) > 10000000 else "Medium" if abs(feature_value) > 1000000 else "Small"
                    feature_contributions.append((feature, feature_value, None, size_factor))
            
            # Display feature analysis
            print(f"     Key Feature Analysis:")
            if feature_contributions:
                for feature, value, expected, contribution in feature_contributions:
                    if feature == 'ModifiedDuration':
                        print(f"       {feature}: {value:.4f} → Expected: ${expected:,.2f} ({contribution:+.1f}% of actual)")
                    elif feature == 'FaceValue':
                        print(f"       {feature}: ${value:,.0f} ({contribution} deal)")
                    else:
                        print(f"       {feature}: {value:,.2f}")
            else:
                print(f"       No significant traditional risk factors detected")
                print(f"       This suggests complex market dynamics or data quality issues")
    
    def _analyze_feature_impact_on_changes(self, features_array, targets_array, deal_info):
        """
        Analyze how each feature contributes to overall market value changes
        """
        # Calculate feature statistics for positive vs negative changes
        positive_mask = targets_array > 0
        negative_mask = targets_array < 0
        
        print(f"\n  Feature Statistics (Gains vs Losses):")
        print(f"  {'Feature':18} | {'Avg (Gains)':12} | {'Avg (Losses)':12} | {'Difference':12} | {'Impact'}")
        print(f"  {'-'*18} | {'-'*12} | {'-'*13} | {'-'*12} | {'-'*20}")
        
        for i, feature in enumerate(self.feature_columns):
            gains_avg = np.mean(features_array[positive_mask, i]) if np.sum(positive_mask) > 0 else 0
            losses_avg = np.mean(features_array[negative_mask, i]) if np.sum(negative_mask) > 0 else 0
            difference = gains_avg - losses_avg
            
            # Determine impact description
            if feature == 'ModifiedDuration':
                impact = "Higher duration → More sensitivity" 
            elif feature == 'DaystoMaturity':
                impact = "Longer maturity → Higher risk"
            elif feature == 'FaceValue':
                impact = "Larger deals → Bigger changes"
            else:
                impact = "Mixed impact"
            
            print(f"  {feature:18} | {gains_avg:12,.2f} | {losses_avg:13,.2f} | {difference:+12,.2f} | {impact}")
    
    def _analyze_currency_impact(self, deal_info, targets_array):
        """
        Analyze market value changes by currency
        """
        currency_changes = {}
        
        for i, deal in enumerate(deal_info):
            currency = deal['Currency']
            mv_change = targets_array[i]
            rate_change = deal['RateChange']
            
            if currency not in currency_changes:
                currency_changes[currency] = {
                    'total_change': 0,
                    'deal_count': 0,
                    'avg_rate_change': 0,
                    'gains': 0,
                    'losses': 0
                }
            
            currency_changes[currency]['total_change'] += mv_change
            currency_changes[currency]['deal_count'] += 1
            currency_changes[currency]['avg_rate_change'] += rate_change
            
            if mv_change > 0:
                currency_changes[currency]['gains'] += mv_change
            else:
                currency_changes[currency]['losses'] += mv_change
        
        # Calculate averages
        for currency in currency_changes:
            count = currency_changes[currency]['deal_count']
            currency_changes[currency]['avg_rate_change'] /= count
            currency_changes[currency]['avg_change'] = currency_changes[currency]['total_change'] / count
        
        # Display results
        print(f"\n  {'Currency':8} | {'Deals':5} | {'Total Change':13} | {'Avg Change':12} | {'Rate Δ':8} | {'Explanation'}")
        print(f"  {'-'*8} | {'-'*5} | {'-'*13} | {'-'*12} | {'-'*8} | {'-'*30}")
        
        for currency, data in sorted(currency_changes.items(), key=lambda x: x[1]['total_change'], reverse=True):
            explanation = self._explain_currency_performance(currency, data)
            print(f"  {currency:8} | {data['deal_count']:5} | ${data['total_change']:12,.0f} | ${data['avg_change']:11,.0f} | {data['avg_rate_change']*10000:+6.1f}bp | {explanation}")
    
    def _explain_currency_performance(self, currency, data):
        """
        Explain why a specific currency performed well or poorly
        """
        rate_change_bp = data['avg_rate_change'] * 10000
        
        if data['total_change'] > 1000000:  # Big gains
            if rate_change_bp > 0:
                return f"Strong gains despite rate increase"
            else:
                return f"Benefited from rate decrease"
        elif data['total_change'] < -1000000:  # Big losses
            if rate_change_bp > 0:
                return f"Hurt by rate increase"
            else:
                return f"Losses despite rate decrease"
        else:
            return f"Stable performance"
    
    def _analyze_market_conditions(self, features_array, targets_array, deal_info):
        """
        Analyze overall market conditions that affected changes
        """
        # Time to maturity analysis
        maturity_idx = self.feature_columns.index('DaystoMaturity') if 'DaystoMaturity' in self.feature_columns else None
        
        if maturity_idx is not None:
            short_term_mask = features_array[:, maturity_idx] <= 30  # 30 days or less
            medium_term_mask = (features_array[:, maturity_idx] > 30) & (features_array[:, maturity_idx] <= 365)
            long_term_mask = features_array[:, maturity_idx] > 365
            
            print(f"\n  Maturity Impact Analysis:")
            print(f"    Short-term (≤30 days): {np.sum(short_term_mask)} deals, Avg change: ${np.mean(targets_array[short_term_mask]):,.0f}")
            print(f"    Medium-term (31-365 days): {np.sum(medium_term_mask)} deals, Avg change: ${np.mean(targets_array[medium_term_mask]):,.0f}")
            print(f"    Long-term (>365 days): {np.sum(long_term_mask)} deals, Avg change: ${np.mean(targets_array[long_term_mask]):,.0f}")
        
        # Rate sensitivity analysis using ModifiedDuration
        mod_dur_idx = self.feature_columns.index('ModifiedDuration') if 'ModifiedDuration' in self.feature_columns else None
        
        if mod_dur_idx is not None:
            high_sensitivity_mask = np.abs(features_array[:, mod_dur_idx]) > 5  # High duration sensitivity
            low_sensitivity_mask = np.abs(features_array[:, mod_dur_idx]) <= 5
            
            print(f"\n  Rate Sensitivity Analysis (ModifiedDuration):")
            print(f"    High sensitivity deals: {np.sum(high_sensitivity_mask)} deals, Avg change: ${np.mean(targets_array[high_sensitivity_mask]):,.0f}")
            print(f"    Low sensitivity deals: {np.sum(low_sensitivity_mask)} deals, Avg change: ${np.mean(targets_array[low_sensitivity_mask]):,.0f}")
        
        # Deal size analysis
        face_value_idx = self.feature_columns.index('FaceValue') if 'FaceValue' in self.feature_columns else None
        
        if face_value_idx is not None:
            large_deals_mask = np.abs(features_array[:, face_value_idx]) > 10000000  # >10M
            medium_deals_mask = (np.abs(features_array[:, face_value_idx]) > 1000000) & (np.abs(features_array[:, face_value_idx]) <= 10000000)
            small_deals_mask = np.abs(features_array[:, face_value_idx]) <= 1000000
            
            print(f"\n  Deal Size Impact Analysis:")
            print(f"    Large deals (>$10M): {np.sum(large_deals_mask)} deals, Avg change: ${np.mean(targets_array[large_deals_mask]):,.0f}")
            print(f"    Medium deals ($1M-$10M): {np.sum(medium_deals_mask)} deals, Avg change: ${np.mean(targets_array[medium_deals_mask]):,.0f}")
            print(f"    Small deals (<$1M): {np.sum(small_deals_mask)} deals, Avg change: ${np.mean(targets_array[small_deals_mask]):,.0f}")
        
        print(f"\n  💡 MARKET INSIGHTS:")
        self._generate_market_insights(features_array, targets_array, deal_info)
    
    def _generate_market_insights(self, features_array, targets_array, deal_info):
        """
        Generate key insights about market movements
        """
        # Overall trend
        total_change = np.sum(targets_array)
        avg_change = np.mean(targets_array)
        
        if total_change > 0:
            trend = "POSITIVE (Portfolio gained value)"
        else:
            trend = "NEGATIVE (Portfolio lost value)"
        
        print(f"    • Overall market trend: {trend}")
        print(f"    • Average deal impact: ${avg_change:,.0f}")
        
        # Rate environment
        rate_changes = [deal['RateChange'] for deal in deal_info]
        avg_rate_change = np.mean(rate_changes) * 10000
        
        if avg_rate_change > 5:
            rate_env = "RISING RATES (Average +{:.1f} bps)".format(avg_rate_change)
        elif avg_rate_change < -5:
            rate_env = "FALLING RATES (Average {:.1f} bps)".format(avg_rate_change)
        else:
            rate_env = "STABLE RATES (Average {:.1f} bps)".format(avg_rate_change)
        
        print(f"    • Rate environment: {rate_env}")
        
        # Volatility
        volatility = np.std(targets_array)
        if volatility > 1000000:
            vol_desc = "HIGH volatility - Large swings in deal values"
        elif volatility > 100000:
            vol_desc = "MODERATE volatility - Some variation in outcomes"
        else:
            vol_desc = "LOW volatility - Consistent deal performance"
        
        print(f"    • Market volatility: {vol_desc} (σ = ${volatility:,.0f})")
        
        # Concentration risk
        top_5_changes = np.sort(np.abs(targets_array))[-5:]
        concentration = np.sum(top_5_changes) / np.sum(np.abs(targets_array)) * 100
        
        print(f"    • Concentration risk: Top 5 deals represent {concentration:.1f}% of total absolute change")
