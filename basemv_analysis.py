"""
BaseMV Analysis using Traditional AI/ML
Analyzes factors contributing to BaseMV ups and downs in forex deals portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
warnings.filterwarnings('ignore')

class BaseMVAnalyzer:
    """
    Comprehensive BaseMV analysis using traditional AI/ML techniques
    """
    
    def __init__(self, data_path='data/Cashflows_FX_V3.csv'):
        """
        Initialize the analyzer with forex data
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for analysis
        """
        print("📊 Loading and preparing forex data...")
        
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
        
        print(f"✓ Prepared {len(self.features.columns)} features")
        # print(f"✓ Target variable: BaseMV (range: ${self.target.min():,.2f} to ${self.target.max():,.2f})")
        
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
        Prepare feature matrix and target variable
        """
        print("🎯 Preparing features and target...")
        
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
        
        # Handle categorical variables
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
        

    def analyze_feature_correlations(self):
        """
        Analyze correlations between features and BaseMV
        """
        print("📈 Analyzing feature correlations...")
        
        # Calculate correlations
        feature_target_corr = self.features.corrwith(self.target).abs().sort_values(ascending=False)
        
        # Display top correlations
        print("\n🔝 Top Features Correlated with BaseMV:")
        print("=" * 50)
        for i, (feature, corr) in enumerate(feature_target_corr.head(15).items(), 1):
            print(f"{i:2d}. {feature:<25} : {corr:.4f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        top_features = feature_target_corr.head(10).index.tolist()
        corr_matrix = self.features[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap (Top 10 Features)')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_target_corr
        
    def train_models(self):
        """
        Train multiple traditional AI/ML models
        """
        print("🤖 Training traditional AI/ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                # Use scaled features for linear models
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"   RMSE: ${rmse:,.2f}")
            print(f"   MAE:  ${mae:,.2f}")
            print(f"   R²:   {r2:.4f}")
            
            # Store model
            self.models[name] = model
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance from tree-based models
        """
        print("\n🌳 Analyzing Feature Importance...")
        
        # Get feature importance from Random Forest
        rf_model = self.models['Random Forest']
        rf_importance = pd.DataFrame({
            'feature': self.features.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get feature importance from Gradient Boosting
        gb_model = self.models['Gradient Boosting']
        gb_importance = pd.DataFrame({
            'feature': self.features.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Display top important features
        print("\n🔝 Top 15 Important Features (Random Forest):")
        print("=" * 55)
        for i, row in rf_importance.head(15).iterrows():
            print(f"{row.name+1:2d}. {row['feature']:<25} : {row['importance']:.4f}")
        
        # Plot feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Random Forest importance
        top_rf = rf_importance.head(10)
        ax1.barh(range(len(top_rf)), top_rf['importance'])
        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels(top_rf['feature'])
        ax1.set_xlabel('Importance')
        ax1.set_title('Random Forest Feature Importance')
        ax1.invert_yaxis()
        
        # Gradient Boosting importance
        top_gb = gb_importance.head(10)
        ax2.barh(range(len(top_gb)), top_gb['importance'])
        ax2.set_yticks(range(len(top_gb)))
        ax2.set_yticklabels(top_gb['feature'])
        ax2.set_xlabel('Importance')
        ax2.set_title('Gradient Boosting Feature Importance')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.feature_importance = {
            'Random Forest': rf_importance,
            'Gradient Boosting': gb_importance
        }
        
        return rf_importance, gb_importance
    
    def analyze_basemv_drivers(self):
        """
        Detailed analysis of BaseMV drivers
        """
        print("\n💰 Analyzing BaseMV Drivers...")
        
        # Analyze by currency
        currency_impact = self.data.groupby('Currency')['BaseMV'].agg([
            'count', 'mean', 'std', 'sum'
        ]).round(2)
        
        print("\n📊 BaseMV Impact by Currency:")
        print("=" * 60)
        print(currency_impact)
        
        # Analyze by deal direction
        direction_impact = self.data.groupby('PayorReceive')['BaseMV'].agg([
            'count', 'mean', 'std', 'sum'
        ]).round(2)
        
        print("\n📊 BaseMV Impact by Deal Direction:")
        print("=" * 60)
        print(direction_impact)
        
        # Analyze by valuation model
        model_impact = self.data.groupby('ValuationModel')['BaseMV'].agg([
            'count', 'mean', 'std', 'sum'
        ]).round(2)
        
        print("\n📊 BaseMV Impact by Valuation Model:")
        print("=" * 60)
        print(model_impact)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Currency impact
        currency_sum = currency_impact['sum'].sort_values()
        axes[0,0].barh(range(len(currency_sum)), currency_sum.values)
        axes[0,0].set_yticks(range(len(currency_sum)))
        axes[0,0].set_yticklabels(currency_sum.index)
        axes[0,0].set_xlabel('Total BaseMV (USD)')
        axes[0,0].set_title('Total BaseMV by Currency')
        
        # Deal direction impact
        direction_sum = direction_impact['sum']
        axes[0,1].bar(direction_sum.index, direction_sum.values)
        axes[0,1].set_xlabel('Deal Direction')
        axes[0,1].set_ylabel('Total BaseMV (USD)')
        axes[0,1].set_title('Total BaseMV by Deal Direction')
        
        # BaseMV distribution
        axes[1,0].hist(self.data['BaseMV'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('BaseMV (USD)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('BaseMV Distribution')
        
        # Time series of BaseMV
        daily_basemv = self.data.groupby('PositionDate')['BaseMV'].sum()
        axes[1,1].plot(daily_basemv.index, daily_basemv.values, marker='o')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Total BaseMV (USD)')
        axes[1,1].set_title('BaseMV Over Time')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('basemv_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return currency_impact, direction_impact, model_impact
    
    def model_comparison(self):
        """
        Compare model performance
        """
        print("\n📊 Model Performance Comparison:")
        print("=" * 60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'RMSE': [results['rmse'] for results in self.results.values()],
            'MAE': [results['mae'] for results in self.results.values()],
            'R²': [results['r2'] for results in self.results.values()]
        })
        
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Plot model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        axes[0].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Model RMSE Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[2].bar(comparison_df['Model'], comparison_df['R²'])
        axes[2].set_ylabel('R²')
        axes[2].set_title('Model R² Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def generate_insights(self):
        """
        Generate actionable insights from the analysis
        """
        print("\n💡 ACTIONABLE INSIGHTS:")
        print("=" * 60)
        
        # Get best performing model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_r2 = self.results[best_model_name]['r2']
        
        print(f"🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
        
        # Top contributing factors
        rf_importance = self.feature_importance['Random Forest']
        top_factors = rf_importance.head(5)
        
        print(f"\n🔝 Top 5 Factors Influencing BaseMV:")
        for i, row in top_factors.iterrows():
            print(f"   {i+1}. {row['feature']} ({row['importance']:.4f})")
        
        # Risk insights
        high_risk_deals = self.data[abs(self.data['BaseMV']) > self.data['BaseMV'].std() * 2]
        print(f"\n⚠️  High Impact Deals: {len(high_risk_deals)} deals (>{2*self.data['BaseMV'].std():,.0f} USD)")
        
        # Currency insights
        currency_totals = self.data.groupby('Currency')['BaseMV'].sum().sort_values(key=abs, ascending=False)
        dominant_currency = currency_totals.index[0]
        print(f"\n💱 Dominant Currency Impact: {dominant_currency} (${currency_totals.iloc[0]:,.0f})")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"1. Focus on {top_factors.iloc[0]['feature']} - highest impact factor")
        print(f"2. Monitor {dominant_currency} exposure closely")
        print(f"3. Use {best_model_name} for BaseMV predictions")
        print(f"4. Review {len(high_risk_deals)} high-impact deals for risk management")
        
def main():
    """
    Main execution function
    """
    print("🚀 BASEMV ANALYSIS USING TRADITIONAL AI/ML")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = BaseMVAnalyzer()
    
    try:
        # Load and prepare data
        analyzer.load_and_prepare_data()
        
        # Analyze correlations
        # correlations = analyzer.analyze_feature_correlations()
        
        # Train models
        results = analyzer.train_models()
        
        # Analyze feature importance
        rf_imp, gb_imp = analyzer.analyze_feature_importance()
        
        # Analyze BaseMV drivers
        curr_impact, dir_impact, model_impact = analyzer.analyze_basemv_drivers()
        
        # Compare models
        comparison = analyzer.model_comparison()
        
        # Generate insights
        analyzer.generate_insights()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Generated visualizations:")
        print(f"   • feature_correlations.png")
        print(f"   • feature_importance.png") 
        print(f"   • basemv_analysis.png")
        print(f"   • model_comparison.png")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
