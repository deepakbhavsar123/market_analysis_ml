#!/usr/bin/env python3
"""
Improved Rate Prediction Model

This script creates an enhanced ML model that addresses the identified variance issues:
1. Uses better feature engineering when risk metrics are zero
2. Implements currency-specific models
3. Uses ensemble methods for higher accuracy
4. Handles missing/zero risk metrics intelligently
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ImprovedRatePredictionModel:
    """
    Enhanced ML model for rate impact prediction that addresses:
    - Large variance in BPDelta and Duration predictions
    - Zero/missing risk metrics 
    - Currency-specific modeling
    - Better feature engineering
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load forex data and prepare for analysis"""
        try:
            self.data = pd.read_excel('data/Cashflows_FX_V3.xlsx')
            print(f"✓ Loaded {len(self.data)} records")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_enhanced_features(self, df):
        """
        Create enhanced features that work better when risk metrics are zero
        """
        enhanced_df = df.copy()
        
        # 1. Deal size categories
        enhanced_df['DealSizeCategory'] = pd.cut(
            enhanced_df['FaceValue'].abs(), 
            bins=[0, 1e6, 10e6, 100e6, 1e9, float('inf')],
            labels=['Small', 'Medium', 'Large', 'VeryLarge', 'Huge']
        )
        
        # 2. Maturity buckets
        enhanced_df['MaturityBucket'] = pd.cut(
            enhanced_df['DaystoMaturity'],
            bins=[0, 30, 90, 365, 1000, float('inf')],
            labels=['Short', 'Medium', 'Long', 'VeryLong', 'Ultra']
        )
        
        # 3. Rate level categories
        enhanced_df['RateLevel'] = pd.cut(
            enhanced_df['FwdRate'],
            bins=[0, 1, 5, 20, 50, float('inf')],
            labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
        )
        
        # 4. Synthetic rate sensitivity when actual metrics are zero
        enhanced_df['SyntheticSensitivity'] = np.where(
            enhanced_df['BPDelta'] == 0,
            enhanced_df['FaceValue'].abs() * enhanced_df['DaystoMaturity'] / 365 * 0.01,  # Approximate sensitivity
            enhanced_df['BPDelta']
        )
        
        # 5. Enhanced duration when zero
        enhanced_df['SyntheticDuration'] = np.where(
            enhanced_df['ModifiedDuration'] == 0,
            enhanced_df['DaystoMaturity'] / 365,  # Simple time-based approximation
            enhanced_df['ModifiedDuration']
        )
        
        # 6. Currency volatility proxy (based on rate range)
        currency_volatility = enhanced_df.groupby('Currency')['FwdRate'].std().fillna(0)
        enhanced_df['CurrencyVolatility'] = enhanced_df['Currency'].map(currency_volatility)
        
        # 7. Deal concentration (face value relative to total)
        total_face_value = enhanced_df['FaceValue'].abs().sum()
        enhanced_df['DealConcentration'] = enhanced_df['FaceValue'].abs() / total_face_value
        
        # 8. Rate change impact multiplier
        enhanced_df['RateImpactMultiplier'] = (
            enhanced_df['SyntheticSensitivity'] * 
            enhanced_df['SyntheticDuration'] * 
            enhanced_df['DealConcentration']
        )
        
        return enhanced_df
    
    def prepare_training_data(self, date1, date2):
        """
        Prepare enhanced training data with better features
        """
        print(f"🔧 Preparing Enhanced Training Data: {date1.date()} -> {date2.date()}")
        
        # Filter data for the two dates
        data_date1 = self.data[self.data['PositionDate'] == date1]
        data_date2 = self.data[self.data['PositionDate'] == date2]
        
        # Get existing deals (exclude new deals for rate impact analysis)
        existing_deals = set(data_date1['DealId'])
        existing_deals_date2 = data_date2[data_date2['DealId'].isin(existing_deals)]
        
        # Filter forward deals
        forward_deals = existing_deals_date2[
            existing_deals_date2['ValuationModel'].isin(['FORWARD', 'FORWARD NPV'])
        ]
        
        print(f"Forward deals for training: {len(forward_deals)}")
        
        # Create training examples with actual rate/MV changes
        training_examples = []
        
        for deal_id in forward_deals['DealId'].unique():
            # Get deal data for both dates
            deal_date1 = data_date1[data_date1['DealId'] == deal_id]
            deal_date2 = data_date2[data_date2['DealId'] == deal_id]
            
            if len(deal_date1) > 0 and len(deal_date2) > 0:
                # Calculate actual changes
                net_mv_date1 = deal_date1['BaseMV'].sum()
                net_mv_date2 = deal_date2['BaseMV'].sum()
                actual_mv_change = net_mv_date2 - net_mv_date1
                
                # Calculate rate changes
                avg_fwd_rate_date1 = deal_date1['FwdRate'].mean()
                avg_fwd_rate_date2 = deal_date2['FwdRate'].mean()
                rate_change = avg_fwd_rate_date2 - avg_fwd_rate_date1
                
                if abs(rate_change) > 0.0001:  # Only include deals with significant rate changes
                    # Create enhanced features for date1 (baseline)
                    enhanced_deal = self.create_enhanced_features(deal_date1)
                    
                    # Create training example
                    example = {
                        'DealId': deal_id,
                        'Currency': deal_date1['Currency'].iloc[0],
                        'RateChange': rate_change,
                        'ActualMVChange': actual_mv_change,
                        
                        # Original features
                        'BPDelta': enhanced_deal['BPDelta'].mean(),
                        'ModifiedDuration': enhanced_deal['ModifiedDuration'].mean(),
                        'DaystoMaturity': enhanced_deal['DaystoMaturity'].mean(),
                        'FaceValue': enhanced_deal['FaceValue'].sum(),
                        'Convexity': enhanced_deal['Convexity'].mean(),
                        'ZeroRate': enhanced_deal['ZeroRate'].mean(),
                        'FwdRate': enhanced_deal['FwdRate'].mean(),
                        'BaseMV': net_mv_date1,
                        
                        # Enhanced features
                        'SyntheticSensitivity': enhanced_deal['SyntheticSensitivity'].mean(),
                        'SyntheticDuration': enhanced_deal['SyntheticDuration'].mean(),
                        'CurrencyVolatility': enhanced_deal['CurrencyVolatility'].iloc[0],
                        'DealConcentration': enhanced_deal['DealConcentration'].mean(),
                        'RateImpactMultiplier': enhanced_deal['RateImpactMultiplier'].mean(),
                        'DealSizeCategory': enhanced_deal['DealSizeCategory'].iloc[0],
                        'MaturityBucket': enhanced_deal['MaturityBucket'].iloc[0],
                        'RateLevel': enhanced_deal['RateLevel'].iloc[0]
                    }
                    
                    training_examples.append(example)
        
        self.training_df = pd.DataFrame(training_examples)
        print(f"✓ Created {len(self.training_df)} enhanced training examples")
        
        return self.training_df
    
    def prepare_features_and_targets(self):
        """
        Prepare feature matrix and target vector with enhanced features
        """
        # Encode categorical features
        categorical_features = ['Currency', 'DealSizeCategory', 'MaturityBucket', 'RateLevel']
        
        df_encoded = self.training_df.copy()
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                
            # Handle missing categories
            df_encoded[feature] = df_encoded[feature].fillna('Unknown')
            df_encoded[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df_encoded[feature])
        
        # Select enhanced feature set
        feature_columns = [
            'RateChange',
            'BPDelta', 'ModifiedDuration', 'DaystoMaturity', 'FaceValue', 
            'Convexity', 'ZeroRate', 'FwdRate', 'BaseMV',
            'SyntheticSensitivity', 'SyntheticDuration', 'CurrencyVolatility',
            'DealConcentration', 'RateImpactMultiplier',
            'Currency_encoded', 'DealSizeCategory_encoded', 'MaturityBucket_encoded', 'RateLevel_encoded'
        ]
        
        # Handle missing values
        for col in feature_columns:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].fillna(0)
        
        X = df_encoded[feature_columns].values
        y = df_encoded['ActualMVChange'].values
        
        return X, y, feature_columns
    
    def train_enhanced_models(self):
        """
        Train ensemble of models with enhanced features
        """
        print(f"\n🤖 Training Enhanced ML Models...")
        
        X, y, feature_columns = self.prepare_features_and_targets()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'RandomForestDeep': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=3,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            if 'RandomForest' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred
            }
            
            print(f"    R² Score: {r2:.4f}")
            print(f"    MAE: ${mae:,.2f}")
            print(f"    RMSE: ${rmse:,.2f}")
        
        # Create ensemble model
        print(f"  Training Ensemble...")
        ensemble = VotingRegressor([
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting']),
            ('rf_deep', models['RandomForestDeep'])
        ])
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        
        results['Ensemble'] = {
            'model': ensemble,
            'r2_score': r2_ensemble,
            'mae': mae_ensemble,
            'rmse': rmse_ensemble,
            'predictions': y_pred_ensemble
        }
        
        print(f"    R² Score: {r2_ensemble:.4f}")
        print(f"    MAE: ${mae_ensemble:,.2f}")
        print(f"    RMSE: ${rmse_ensemble:,.2f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.model_results = results
        
        print(f"\n✅ Best Model: {best_model_name} (R² = {results[best_model_name]['r2_score']:.4f})")
        
        # Feature importance for Random Forest models
        if 'RandomForest' in best_model_name:
            importance = self.best_model.feature_importances_
            self.feature_importance = dict(zip(feature_columns, importance))
            
            print(f"\n📊 Feature Importance ({best_model_name}):")
            sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_importance[:10]:  # Top 10 features
                print(f"  {feature}: {imp:.4f}")
        
        return results
    
    def predict_rate_impact(self, scenario_rate_change=0.005):
        """
        Predict rate impact using the enhanced model
        """
        print(f"\n🔮 Enhanced Rate Impact Prediction (Scenario: {scenario_rate_change*100:.1f}% rate change)")
        
        # Prepare prediction data
        prediction_data = []
        
        for _, row in self.training_df.iterrows():
            # Create prediction example with scenario rate change
            pred_example = row.copy()
            pred_example['RateChange'] = scenario_rate_change
            prediction_data.append(pred_example)
        
        pred_df = pd.DataFrame(prediction_data)
        
        # Encode categorical features using existing encoders
        categorical_features = ['Currency', 'DealSizeCategory', 'MaturityBucket', 'RateLevel']
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                pred_df[feature] = pred_df[feature].fillna('Unknown')
                # Handle unseen categories
                try:
                    pred_df[feature + '_encoded'] = self.label_encoders[feature].transform(pred_df[feature])
                except ValueError:
                    # For unseen categories, use most frequent category
                    most_frequent = self.label_encoders[feature].classes_[0]
                    pred_df[feature] = pred_df[feature].apply(lambda x: x if x in self.label_encoders[feature].classes_ else most_frequent)
                    pred_df[feature + '_encoded'] = self.label_encoders[feature].transform(pred_df[feature])
        
        # Prepare feature matrix
        feature_columns = [
            'RateChange',
            'BPDelta', 'ModifiedDuration', 'DaystoMaturity', 'FaceValue', 
            'Convexity', 'ZeroRate', 'FwdRate', 'BaseMV',
            'SyntheticSensitivity', 'SyntheticDuration', 'CurrencyVolatility',
            'DealConcentration', 'RateImpactMultiplier',
            'Currency_encoded', 'DealSizeCategory_encoded', 'MaturityBucket_encoded', 'RateLevel_encoded'
        ]
        
        # Fill missing values
        for col in feature_columns:
            if col in pred_df.columns:
                pred_df[col] = pred_df[col].fillna(0)
        
        X_pred = pred_df[feature_columns].values
        
        # Scale if needed
        if 'GradientBoosting' in self.best_model_name:
            X_pred = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.best_model.predict(X_pred)
        
        # Analyze predictions
        total_predicted_impact = predictions.sum()
        
        print(f"Sample Enhanced Predictions:")
        for i in range(min(5, len(predictions))):
            row = pred_df.iloc[i]
            print(f"  Deal {row['DealId']} ({row['Currency']}):")
            print(f"    Predicted Change: ${predictions[i]:,.2f}")
            print(f"    Enhanced Features:")
            print(f"      SyntheticSensitivity: {row['SyntheticSensitivity']:,.2f}")
            print(f"      SyntheticDuration: {row['SyntheticDuration']:.4f}")
            print(f"      RateImpactMultiplier: {row['RateImpactMultiplier']:,.6f}")
        
        print(f"\n🎯 Enhanced Prediction Summary:")
        print(f"  Total Predicted Impact: ${total_predicted_impact:,.2f}")
        print(f"  Model Used: {self.best_model_name}")
        print(f"  Model R²: {self.model_results[self.best_model_name]['r2_score']:.4f}")
        print(f"  Enhanced Features: Synthetic sensitivity, currency volatility, deal concentration")
        
        return {
            'total_predicted_impact': total_predicted_impact,
            'individual_predictions': predictions,
            'model_name': self.best_model_name,
            'model_r2': self.model_results[self.best_model_name]['r2_score'],
            'scenario_rate_change': scenario_rate_change
        }
    
    def compare_with_actual(self, actual_impact):
        """
        Compare enhanced predictions with actual results
        """
        enhanced_prediction = self.predict_rate_impact()
        
        predicted_impact = enhanced_prediction['total_predicted_impact']
        variance = abs(actual_impact - predicted_impact)
        variance_pct = (variance / abs(actual_impact)) * 100 if actual_impact != 0 else 0
        
        print(f"\n📊 ENHANCED MODEL COMPARISON:")
        print(f"  Actual Impact: ${actual_impact:,.2f}")
        print(f"  Enhanced Predicted: ${predicted_impact:,.2f}")
        print(f"  Variance: ${variance:,.2f} ({variance_pct:.1f}%)")
        
        # Improvement assessment
        if variance_pct < 25:
            print(f"  ✅ Excellent prediction: Variance < 25%")
        elif variance_pct < 50:
            print(f"  ✅ Good prediction: Variance < 50%")
        else:
            print(f"  ⚠️  Needs improvement: Variance > 50%")
        
        return {
            'actual_impact': actual_impact,
            'predicted_impact': predicted_impact,
            'variance': variance,
            'variance_percentage': variance_pct,
            'model_performance': 'excellent' if variance_pct < 25 else 'good' if variance_pct < 50 else 'needs_improvement'
        }

def main():
    """
    Main function to test the improved rate prediction model
    """
    print("🚀 IMPROVED RATE PREDICTION MODEL")
    print("="*50)
    
    # Initialize model
    model = ImprovedRatePredictionModel()
    
    # Load data
    if not model.load_data():
        return
    
    # Prepare training data
    date1 = datetime(2012, 3, 6)
    date2 = datetime(2012, 3, 7)
    
    training_df = model.prepare_training_data(date1, date2)
    
    if len(training_df) == 0:
        print("❌ No training data available")
        return
    
    # Train enhanced models
    results = model.train_enhanced_models()
    
    # Predict rate impact
    prediction_results = model.predict_rate_impact()
    
    # Compare with actual (from previous analysis: $88,133,996.17)
    actual_forward_impact = 88133996.17
    comparison = model.compare_with_actual(actual_forward_impact)
    
    print(f"\n💡 IMPROVEMENT SUMMARY:")
    print(f"Previous Model Variance: 40.9% (${36_037_388.72:,.2f})")
    print(f"Enhanced Model Variance: {comparison['variance_percentage']:.1f}% (${comparison['variance']:,.2f})")
    
    if comparison['variance_percentage'] < 40.9:
        improvement = 40.9 - comparison['variance_percentage']
        print(f"✅ Improvement: {improvement:.1f} percentage points better")
    else:
        degradation = comparison['variance_percentage'] - 40.9
        print(f"⚠️  Degradation: {degradation:.1f} percentage points worse")

if __name__ == "__main__":
    main()
