"""
Improved Random Forest Model with Hyperparameter Tuning
Goal: Achieve R² Score > 0.7
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load preprocessed feature matrix
X = pd.read_csv("../Preprocessing/cleaned_X.csv")
y = pd.read_csv("../Preprocessing/y.csv")
y = y.squeeze()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 70)
print("IMPROVED RANDOM FOREST MODEL WITH HYPERPARAMETER TUNING")
print("=" * 70)
print(f"\nDataset Size:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ============================================
# HYPERPARAMETER TUNING
# ============================================

print(f"\n🔧 Starting Hyperparameter Tuning...")
print("   This may take a few minutes...\n")

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 15, 20, 25],  # None = unlimited depth
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Create base model
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS FOUND:")
print("=" * 70)
for param, value in best_params.items():
    print(f"  • {param}: {value}")
print(f"\n✅ Best Cross-Validation R² Score: {best_cv_score:.4f}")

# ============================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================

print("\n🚀 Training final model with best parameters...")

# Use the best estimator from grid search
best_rf_model = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_rf_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ============================================
# PRINT RESULTS
# ============================================

print("\n" + "=" * 70)
print("IMPROVED MODEL PERFORMANCE:")
print("=" * 70)
print(f"\n📈 Test Set Metrics:")
print(f"  • MAE  (Mean Absolute Error)      : {mae:.4f} kg/hectare")
print(f"  • MSE  (Mean Squared Error)       : {mse:.4f}")
print(f"  • RMSE (Root Mean Squared Error)  : {rmse:.4f} kg/hectare")
print(f"  • R²   (R-Squared Score)          : {r2:.4f} ({r2*100:.2f}% variance explained)")

print(f"\n💡 Performance Assessment:")
if r2 >= 0.9:
    print(f"   ✅ Outstanding! (R² ≥ 0.9)")
elif r2 >= 0.8:
    print(f"   ✅ Excellent! (0.8 ≤ R² < 0.9)")
elif r2 >= 0.7:
    print(f"   ✅ Good! (0.7 ≤ R² < 0.8)")
elif r2 >= 0.6:
    print(f"   ⚠️  Fair (0.6 ≤ R² < 0.7) - Can be improved")
else:
    print(f"   ❌ Poor (R² < 0.6) - Needs significant improvement")

# ============================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================

print(f"\n📊 Feature Importance Ranking:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")

# ============================================
# SAVE RESULTS
# ============================================

# Save the improved model
joblib.dump(best_rf_model, '../Models/improved_rf_model.pkl')
print(f"\n✅ Model saved as 'improved_rf_model.pkl'")

# Save predictions for evaluation
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results_df.to_csv('../Evaluation/improved_rf_results.csv', index=False)
print(f"✅ Predictions saved to 'improved_rf_results.csv'")

# Save best parameters
params_df = pd.DataFrame([best_params])
params_df.to_csv('../Models/best_hyperparameters.csv', index=False)
print(f"✅ Best parameters saved to 'best_hyperparameters.csv'")

print("\n" + "=" * 70)
print("MODEL IMPROVEMENT COMPLETE! ✨")
print("=" * 70)
