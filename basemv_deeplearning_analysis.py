"""
BaseMV Analysis using Deep Learning
Analyzes feature contributions to BaseMV for each date using neural networks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# For feature importance and interpretability
import shap
from scipy import stats

class BaseMVDeepLearningAnalyzer:
    """
    Deep Learning analysis for BaseMV feature contributions by date
    """
    
    def __init__(self, data_path='data/Cashflows_FX_V3.csv'):
        """
        Initialize the deep learning analyzer
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.daily_contributions = {}
        self.attention_weights = {}
        
        # Create output directory
        self.output_dir = 'deeplearning_results'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Created output directory: {self.output_dir}")
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for deep learning analysis
        """
        print("🚀 Loading and preparing forex data for deep learning...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.data)} records")
        
        # Convert date columns
        self.data['PositionDate'] = pd.to_datetime(self.data['PositionDate'])
        self.data['CashFlowDate'] = pd.to_datetime(self.data['CashFlowDate'])
        self.data['DealDate'] = pd.to_datetime(self.data['DealDate'])
        self.data['MaturityDate'] = pd.to_datetime(self.data['MaturityDate'])
        
        # Create additional features
        self.create_derived_features()
        
        # Prepare features and target
        self.prepare_features_target()
        
        print(f"✓ Prepared {len(self.features.columns)} features for {len(self.data['PositionDate'].unique())} unique dates")
        
    def create_derived_features(self):
        """
        Create derived features that might influence BaseMV
        """
        print("🔧 Creating derived features...")
        
        # Time-based features
        self.data['DaysToMaturity_binned'] = pd.cut(self.data['DaystoMaturity'], 
                                                   bins=[0, 30, 90, 365, np.inf], 
                                                   labels=['Short', 'Medium', 'Long', 'VeryLong'])
        
        # Currency strength indicators
        self.data['USD_Involved'] = (self.data['Currency'] == 'USD').astype(int)
        self.data['EUR_Involved'] = (self.data['Currency'] == 'EUR').astype(int)
        self.data['JPY_Involved'] = (self.data['Currency'] == 'JPY').astype(int)
        
        # Market timing features
        self.data['Month'] = self.data['PositionDate'].dt.month
        self.data['DayOfWeek'] = self.data['PositionDate'].dt.dayofweek
        
        print("✓ Created derived features")
        
    def prepare_features_target(self):
        """
        Prepare feature matrix and target variable for deep learning
        """
        print("🎯 Preparing features and target for neural networks...")
        
        # Select numerical features
        numerical_features = [
            'SpotRate', 'FwdRate', 'DaystoMaturity',
            'USD_Involved', 'EUR_Involved', 'JPY_Involved', 'Month', 'DayOfWeek'
        ]
        
        # Select categorical features
        categorical_features = [
            'Currency', 'ValuationModel', 'CcyPair',
            'DaysToMaturity_binned'
        ]
        
        # Create feature matrix
        features_df = self.data[numerical_features + categorical_features].copy()
        
        # Handle categorical variables with label encoding
        for col in categorical_features:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[col] = features_df[col].astype(str)
                features_df[col] = le.fit_transform(features_df[col])
                self.label_encoders[col] = le
        
        # Remove rows with missing target
        mask = ~self.data['BaseMV'].isna()
        self.features = features_df[mask].fillna(0)
        self.target = self.data[mask]['BaseMV']
        self.dates = self.data[mask]['PositionDate']
        
        # Store feature names for later use
        self.feature_names = self.features.columns.tolist()

        
    def create_attention_model(self, input_dim):
        """
        Create a neural network with attention mechanism for feature importance
        """
        print("🧠 Building attention-based neural network...")
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='features')
        
        # Feature embedding layers
        x = layers.Dense(128, activation='relu', name='embedding_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu', name='embedding_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Attention mechanism
        attention_weights = layers.Dense(input_dim, activation='softmax', name='attention')(x)
        
        # Apply attention to original features
        attended_features = layers.Multiply(name='attended_features')([inputs, attention_weights])
        
        # Prediction layers
        x = layers.Dense(32, activation='relu', name='prediction_1')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(1, name='basemv_prediction')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[outputs, attention_weights], 
                           name='BaseMV_Attention_Model')
        
        return model
    
    def create_feature_importance_model(self, input_dim):
        """
        Create a simpler model for SHAP analysis
        """
        print("🔍 Building feature importance model...")
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ], name='BaseMV_Simple_Model')
        
        return model
    
    def train_models(self):
        """
        Train deep learning models
        """
        print("🚀 Training deep learning models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            self.features, self.target, self.dates, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['feature_scaler'] = scaler
        
        # Scale target
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        self.scalers['target_scaler'] = target_scaler
        
        input_dim = X_train_scaled.shape[1]
        
        # Train attention model
        print("\n📊 Training Attention Model...")
        attention_model = self.create_attention_model(input_dim)
        attention_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=['mse', 'mse'],  # Change to mse for both outputs
            loss_weights=[1.0, 0.1],
            metrics=[['mae'], ['mae']]  # Use mae for both outputs
        )
        
        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True  # Reduced patience
        )
        
        # Create dummy attention targets (uniform distribution)
        attention_targets = np.ones((len(X_train_scaled), input_dim)) / input_dim
        attention_targets_test = np.ones((len(X_test_scaled), input_dim)) / input_dim
        
        history = attention_model.fit(
            X_train_scaled, [y_train_scaled, attention_targets],
            validation_data=(X_test_scaled, [y_test_scaled, attention_targets_test]),
            epochs=50,  # Reduced epochs
            batch_size=64,  # Increased batch size
            callbacks=[early_stopping],
            verbose=1  # Show progress
        )
        
        # Train simple model for SHAP
        print("📊 Training Simple Model for SHAP...")
        simple_model = self.create_feature_importance_model(input_dim)
        simple_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        simple_history = simple_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_test_scaled, y_test_scaled),
            epochs=50,  # Reduced epochs
            batch_size=64,  # Increased batch size
            callbacks=[early_stopping],
            verbose=1  # Show progress
        )
        
        # Store models and data
        self.models['attention'] = attention_model
        self.models['simple'] = simple_model
        self.train_data = (X_train_scaled, y_train_scaled, dates_train)
        self.test_data = (X_test_scaled, y_test_scaled, dates_test)
        
        # Evaluate models
        self.evaluate_models()
        
        print("✅ Model training completed!")
        
    def evaluate_models(self):
        """
        Evaluate model performance
        """
        print("\n📈 Evaluating model performance...")
        
        X_test_scaled, y_test_scaled, dates_test = self.test_data
        target_scaler = self.scalers['target_scaler']
        
        # Attention model predictions
        attention_pred, attention_weights = self.models['attention'].predict(X_test_scaled, verbose=0)
        attention_pred_original = target_scaler.inverse_transform(attention_pred.reshape(-1, 1)).flatten()
        y_test_original = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        # Simple model predictions
        simple_pred = self.models['simple'].predict(X_test_scaled, verbose=0)
        simple_pred_original = target_scaler.inverse_transform(simple_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        attention_r2 = r2_score(y_test_original, attention_pred_original)
        attention_rmse = np.sqrt(mean_squared_error(y_test_original, attention_pred_original))
        attention_mae = mean_absolute_error(y_test_original, attention_pred_original)
        
        simple_r2 = r2_score(y_test_original, simple_pred_original)
        simple_rmse = np.sqrt(mean_squared_error(y_test_original, simple_pred_original))
        simple_mae = mean_absolute_error(y_test_original, simple_pred_original)
        
        print(f"🎯 Attention Model Performance:")
        print(f"   R²:   {attention_r2:.4f}")
        print(f"   RMSE: ${attention_rmse:,.2f}")
        print(f"   MAE:  ${attention_mae:,.2f}")
        
        print(f"\n🎯 Simple Model Performance:")
        print(f"   R²:   {simple_r2:.4f}")
        print(f"   RMSE: ${simple_rmse:,.2f}")
        print(f"   MAE:  ${simple_mae:,.2f}")
        
    def analyze_daily_feature_contributions(self):
        """
        Analyze feature contributions for each date
        """
        print("\n📅 Analyzing daily feature contributions...")
        
        # Prepare full dataset
        X_scaled = self.scalers['feature_scaler'].transform(self.features)
        
        # Get attention weights for all data
        _, attention_weights = self.models['attention'].predict(X_scaled, verbose=0)
        
        # Create DataFrame with dates and attention weights
        daily_data = pd.DataFrame(X_scaled, columns=self.feature_names)
        daily_data['PositionDate'] = self.dates.values
        daily_data['BaseMV'] = self.target.values
        
        # Add attention weights
        for i, feature in enumerate(self.feature_names):
            daily_data[f'{feature}_attention'] = attention_weights[:, i]
        
        # Group by date and calculate average contributions
        daily_contributions = daily_data.groupby('PositionDate').agg({
            **{f'{feature}_attention': 'mean' for feature in self.feature_names},
            'BaseMV': ['sum', 'mean', 'count']
        }).round(4)
        
        # Flatten column names
        daily_contributions.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] 
                                     for col in daily_contributions.columns]
        
        self.daily_contributions = daily_contributions
        
        print(f"✓ Analyzed contributions for {len(daily_contributions)} unique dates")
        
    def calculate_shap_values(self):
        """
        Calculate SHAP values for feature importance
        """
        print("\n🔍 Calculating SHAP values for feature interpretability...")
        
        try:
            # Use a sample of training data for SHAP background
            X_train_scaled, _, _ = self.train_data
            background_sample = X_train_scaled[:100]  # Use 100 samples as background
            
            # Create SHAP explainer
            explainer = shap.DeepExplainer(self.models['simple'], background_sample)
            
            # Calculate SHAP values for test data (sample for efficiency)
            X_test_scaled, _, dates_test = self.test_data
            test_sample = X_test_scaled[:200]  # Use 200 test samples
            dates_sample = dates_test.iloc[:200]
            
            shap_values = explainer.shap_values(test_sample)
            
            # Create SHAP DataFrame - handle different output formats
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_array = shap_values[0]
            else:
                shap_array = shap_values
                
            # Handle different SHAP output shapes
            if len(shap_array.shape) == 3:
                # If shape is (samples, features, 1), squeeze the last dimension
                shap_array = shap_array.squeeze(-1)
            
            # Ensure correct shape
            if len(shap_array.shape) == 2 and shap_array.shape[1] == len(self.feature_names):
                shap_df = pd.DataFrame(shap_array, columns=self.feature_names)
                shap_df['PositionDate'] = dates_sample.values
                
                # Group by date
                daily_shap = shap_df.groupby('PositionDate')[self.feature_names].mean()
                self.daily_shap_values = daily_shap
                print("✓ SHAP values processed successfully")
            else:
                print(f"⚠️ SHAP values shape mismatch: {shap_array.shape} vs {len(self.feature_names)} features")
                self.daily_shap_values = None
            
            print("✓ SHAP values calculated successfully")
            
        except Exception as e:
            print(f"⚠️ SHAP calculation failed: {str(e)}")
            print("Continuing without SHAP analysis...")
            self.daily_shap_values = None
    
    def visualize_daily_contributions(self):
        """
        Create visualizations for daily feature contributions
        """
        print("\n📊 Creating daily contribution visualizations...")
        
        # Get attention-based contributions
        attention_cols = [col for col in self.daily_contributions.columns if '_attention' in col]
        attention_data = self.daily_contributions[attention_cols]
        attention_data.columns = [col.replace('_attention_mean', '') for col in attention_data.columns]
        
        # Plot 1: Heatmap of daily attention weights
        plt.figure(figsize=(16, 10))
        
        # Select top features by average attention
        avg_attention = attention_data.mean().sort_values(ascending=False)
        top_features = avg_attention.head(8).index
        
        sns.heatmap(attention_data[top_features].T, 
                   cmap='viridis', 
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=attention_data.index.strftime('%m-%d'),
                   yticklabels=top_features)
        plt.title('Daily Feature Attention Weights (Top 8 Features)')
        plt.xlabel('Date')
        plt.ylabel('Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'daily_attention_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Time series of top feature contributions
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features[:4]):
            ax = axes[i]
            dates = attention_data.index
            values = attention_data[feature]
            
            ax.plot(dates, values, marker='o', linewidth=2, markersize=4)
            ax.set_title(f'{feature} - Daily Attention Weight')
            ax.set_xlabel('Date')
            ax.set_ylabel('Attention Weight')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'daily_feature_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: BaseMV vs Feature Importance correlation
        plt.figure(figsize=(14, 8))
        
        basemv_daily = self.daily_contributions['BaseMV_sum']
        
        # Calculate correlation between BaseMV and feature attention
        correlations = []
        for feature in top_features:
            corr = np.corrcoef(basemv_daily, attention_data[feature])[0, 1]
            correlations.append(corr)
        
        bars = plt.bar(range(len(top_features)), correlations)
        plt.title('Correlation between Daily BaseMV and Feature Attention Weights')
        plt.xlabel('Features')
        plt.ylabel('Correlation with BaseMV')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        
        # Color bars based on correlation strength
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if corr > 0.1:
                bar.set_color('green')
            elif corr < -0.1:
                bar.set_color('red')
            else:
                bar.set_color('gray')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'basemv_feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_daily_insights(self):
        """
        Generate insights for each date
        """
        print("\n💡 Generating daily insights...")
        
        attention_cols = [col for col in self.daily_contributions.columns if '_attention' in col]
        attention_data = self.daily_contributions[attention_cols]
        attention_data.columns = [col.replace('_attention_mean', '') for col in attention_data.columns]
        
        # Find most important features for each date
        daily_insights = {}
        
        for date in attention_data.index:
            date_data = attention_data.loc[date]
            top_3_features = date_data.nlargest(3)
            basemv_sum = self.daily_contributions.loc[date, 'BaseMV_sum']
            deal_count = self.daily_contributions.loc[date, 'BaseMV_count']
            
            daily_insights[date] = {
                'top_features': top_3_features.to_dict(),
                'basemv_total': basemv_sum,
                'deal_count': deal_count,
                'dominant_factor': top_3_features.index[0],
                'dominance_score': top_3_features.iloc[0]
            }
        
        # Print summary insights
        print(f"\n📋 DAILY INSIGHTS SUMMARY:")
        print("=" * 60)
        
        # Most volatile dates
        basemv_volatility = self.daily_contributions['BaseMV_sum'].abs()
        high_impact_dates = basemv_volatility.nlargest(3)
        
        print(f"🔥 Highest Impact Dates:")
        for date, impact in high_impact_dates.items():
            insights = daily_insights[date]
            print(f"   {date.strftime('%Y-%m-%d')}: ${impact:,.0f}")
            print(f"      Dominant Factor: {insights['dominant_factor']} ({insights['dominance_score']:.3f})")
            print(f"      Deal Count: {insights['deal_count']:.0f}")
        
        # Feature dominance patterns
        feature_dominance = {}
        for date, insights in daily_insights.items():
            dominant = insights['dominant_factor']
            if dominant not in feature_dominance:
                feature_dominance[dominant] = 0
            feature_dominance[dominant] += 1
        
        print(f"\n🏆 Feature Dominance Frequency:")
        sorted_dominance = sorted(feature_dominance.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_dominance[:5]:
            percentage = (count / len(daily_insights)) * 100
            print(f"   {feature}: {count} days ({percentage:.1f}%)")
        
        self.daily_insights = daily_insights
        
        return daily_insights
    
    def save_results(self):
        """
        Save analysis results to CSV files
        """
        print("\n💾 Saving analysis results...")
        
        # Save daily contributions
        contributions_path = os.path.join(self.output_dir, 'daily_feature_contributions.csv')
        self.daily_contributions.to_csv(contributions_path)
        print(f"✓ Saved {contributions_path}")
        
        # Save daily insights summary
        insights_path = os.path.join(self.output_dir, 'daily_insights_summary.csv')
        insights_df = pd.DataFrame.from_dict(self.daily_insights, orient='index')
        insights_df.to_csv(insights_path)
        print(f"✓ Saved {insights_path}")
        
        # Save SHAP values if available
        if hasattr(self, 'daily_shap_values') and self.daily_shap_values is not None:
            shap_path = os.path.join(self.output_dir, 'daily_shap_values.csv')
            self.daily_shap_values.to_csv(shap_path)
            print(f"✓ Saved {shap_path}")
            
        print(f"\n📂 All results saved in folder: {self.output_dir}")

def main():
    """
    Main execution function
    """
    print("🚀 BASEMV DEEP LEARNING ANALYSIS")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = BaseMVDeepLearningAnalyzer()
    
    try:
        # Load and prepare data
        analyzer.load_and_prepare_data()
        
        # Train deep learning models
        analyzer.train_models()
        
        # Analyze daily feature contributions
        analyzer.analyze_daily_feature_contributions()
        
        # Calculate SHAP values
        analyzer.calculate_shap_values()
        
        # Create visualizations
        analyzer.visualize_daily_contributions()
        
        # Generate insights
        analyzer.generate_daily_insights()
        
        # Save results
        analyzer.save_results()
        
        print(f"\n✅ Deep learning analysis completed successfully!")
        print(f"📊 Generated files in folder '{analyzer.output_dir}':")
        print(f"   • daily_attention_heatmap.png")
        print(f"   • daily_feature_timeseries.png")
        print(f"   • basemv_feature_correlation.png")
        print(f"   • daily_feature_contributions.csv")
        print(f"   • daily_insights_summary.csv")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
