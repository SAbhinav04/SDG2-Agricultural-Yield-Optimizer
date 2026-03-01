"""
Advanced Model with Feature Engineering
Creates interaction features and polynomial features to improve R² score
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load original data (before scaling)
df = pd.read_csv("../Dataset/crop_yield_with_crops.csv")

print("=" * 70)
print("ADVANCED MODEL WITH FEATURE ENGINEERING")
print("=" * 70)

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n🔧 Creating Advanced Features...")

# Extract basic features
X_basic = df.drop("Yield", axis=1)
y = df["Yield"]

# 1. Create interaction features
X_basic['N_P_interaction'] = X_basic['Nitrogen'] * X_basic['Phosphorus']
X_basic['N_K_interaction'] = X_basic['Nitrogen'] * X_basic['Potassium']
X_basic['P_K_interaction'] = X_basic['Phosphorus'] * X_basic['Potassium']
X_basic['NPK_total'] = X_basic['Nitrogen'] + X_basic['Phosphorus'] + X_basic['Potassium']
X_basic['NPK_ratio'] = X_basic['Nitrogen'] / (X_basic['Phosphorus'] + 1)  # Avoid div by zero

# 2. Create rainfall-nutrient interactions
X_basic['Rainfall_N'] = X_basic['Rainfall'] * X_basic['Nitrogen']
X_basic['Rainfall_P'] = X_basic['Rainfall'] * X_basic['Phosphorus']
X_basic['Rainfall_K'] = X_basic['Rainfall'] * X_basic['Potassium']

# 3. Create squared features (polynomial)
X_basic['Nitrogen_sq'] = X_basic['Nitrogen'] ** 2
X_basic['Phosphorus_sq'] = X_basic['Phosphorus'] ** 2
X_basic['Potassium_sq'] = X_basic['Potassium'] ** 2
X_basic['Rainfall_sq'] = X_basic['Rainfall'] ** 2

# 4. One-hot encode crop types
X_encoded = pd.get_dummies(X_basic, columns=['Crop'], drop_first=True)

print(f"\n📊 Feature Engineering Summary:")
print(f"  • Original features: {len(X_basic.columns) - len([c for c in X_basic.columns if 'Crop_' in c and c != 'Crop'])}")
print(f"  • Total features after engineering: {X_encoded.shape[1]}")
print(f"  • New interaction features: 8")
print(f"  • New polynomial features: 4")

# ============================================
# TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\n📦 Dataset Split:")
print(f"  • Training samples: {len(X_train)}")
print(f"  • Test samples: {len(X_test)}")

# ============================================
# MODEL 1: IMPROVED RANDOM FOREST
# ============================================

print(f"\n🌲 Training Random Forest with Feature Engineering...")

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,  # No limit - let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n  Random Forest Results:")
print(f"    MAE  : {rf_mae:.4f}")
print(f"    RMSE : {rf_rmse:.4f}")
print(f"    R²   : {rf_r2:.4f} ({rf_r2*100:.2f}%)")

# ============================================
# MODEL 2: GRADIENT BOOSTING (Often performs better)
# ============================================

print(f"\n🚀 Training Gradient Boosting with Feature Engineering...")

gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print(f"\n  Gradient Boosting Results:")
print(f"    MAE  : {gb_mae:.4f}")
print(f"    RMSE : {gb_rmse:.4f}")
print(f"    R²   : {gb_r2:.4f} ({gb_r2*100:.2f}%)")

# ============================================
# SELECT BEST MODEL
# ============================================

print("\n" + "=" * 70)
print("FINAL RESULTS:")
print("=" * 70)

if gb_r2 > rf_r2:
    print(f"\n🏆 WINNER: Gradient Boosting")
    best_model = gb_model
    best_pred = gb_pred
    best_mae = gb_mae
    best_rmse = gb_rmse
    best_r2 = gb_r2
    model_name = "Gradient Boosting"
else:
    print(f"\n🏆 WINNER: Random Forest")
    best_model = rf_model
    best_pred = rf_pred
    best_mae = rf_mae
    best_rmse = rf_rmse
    best_r2 = rf_r2
    model_name = "Random Forest"

print(f"\n📈 Best Model Performance:")
print(f"  • MAE  : {best_mae:.4f} kg/hectare")
print(f"  • RMSE : {best_rmse:.4f} kg/hectare")
print(f"  • R²   : {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")

print(f"\n💡 Performance Assessment:")
if best_r2 >= 0.9:
    print(f"   ✅ Outstanding! (R² ≥ 0.9)")
elif best_r2 >= 0.8:
    print(f"   ✅ Excellent! (0.8 ≤ R² < 0.9)")
elif best_r2 >= 0.7:
    print(f"   ✅ Good! (0.7 ≤ R² < 0.8)")
elif best_r2 >= 0.6:
    print(f"   ⚠️  Fair (0.6 ≤ R² < 0.7)")
else:
    print(f"   ❌ Poor (R² < 0.6)")

# ============================================
# FEATURE IMPORTANCE
# ============================================

print(f"\n📊 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")

# ============================================
# SAVE RESULTS
# ============================================

# Save the best model
joblib.dump(best_model, f'../Models/advanced_{model_name.lower().replace(" ", "_")}_model.pkl')
print(f"\n✅ Best model saved as 'advanced_{model_name.lower().replace(' ', '_')}_model.pkl'")

# Save predictions
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_pred
})
results_df.to_csv('../Evaluation/advanced_model_results.csv', index=False)
print(f"✅ Predictions saved to 'advanced_model_results.csv'")

# Save feature list
feature_list_df = pd.DataFrame({'Feature': X_encoded.columns})
feature_list_df.to_csv('../Models/feature_list.csv', index=False)
print(f"✅ Feature list saved to 'feature_list.csv'")

print("\n" + "=" * 70)
print("ADVANCED MODEL TRAINING COMPLETE! ✨")
print("=" * 70)
