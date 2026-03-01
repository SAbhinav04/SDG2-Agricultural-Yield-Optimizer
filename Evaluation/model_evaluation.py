"""
Model Evaluation Script
Calculates comprehensive metrics: MAE, MSE, RMSE, R2 Score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results file with actual vs predicted values
results_df = pd.read_csv('rf_results.csv')

# Extract actual and predicted values
y_actual = results_df['Actual']
y_predicted = results_df['Predicted']

# ============================================
# CALCULATE EVALUATION METRICS
# ============================================

# 1. Mean Absolute Error (MAE)
# Average of absolute differences between predictions and actual values
mae = mean_absolute_error(y_actual, y_predicted)

# 2. Mean Squared Error (MSE)
# Average of squared differences between predictions and actual values
mse = mean_squared_error(y_actual, y_predicted)

# 3. Root Mean Squared Error (RMSE)
# Square root of MSE, in same units as target variable
rmse = np.sqrt(mse)

# 4. R² Score (Coefficient of Determination)
# Proportion of variance in target variable explained by the model
r2 = r2_score(y_actual, y_predicted)

# ============================================
# PRINT EVALUATION RESULTS
# ============================================

print("=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"\n📊 Dataset Statistics:")
print(f"   • Number of test samples: {len(y_actual)}")
print(f"   • Actual yield range: {y_actual.min():.2f} - {y_actual.max():.2f}")
print(f"   • Predicted yield range: {y_predicted.min():.2f} - {y_predicted.max():.2f}")

print(f"\n📈 Performance Metrics:")
print(f"   • MAE  (Mean Absolute Error)      : {mae:.4f}")
print(f"   • MSE  (Mean Squared Error)       : {mse:.4f}")
print(f"   • RMSE (Root Mean Squared Error)  : {rmse:.4f}")
print(f"   • R²   (R-Squared Score)          : {r2:.4f}")

print(f"\n💡 Interpretation:")
print(f"   • On average, predictions are off by ±{mae:.2f} kg/hectare")
print(f"   • RMSE penalizes large errors: {rmse:.2f} kg/hectare")
print(f"   • Model explains {r2*100:.2f}% of variance in crop yield")

if r2 > 0.8:
    print(f"   ✅ Model performance: Excellent!")
elif r2 > 0.6:
    print(f"   ⚠️  Model performance: Good, but has room for improvement")
else:
    print(f"   ❌ Model performance: Needs improvement")

print("=" * 60)

# ============================================
# ADDITIONAL ANALYSIS
# ============================================

# Calculate residuals (errors)
residuals = y_actual - y_predicted

print(f"\n📉 Residual Analysis:")
print(f"   • Mean residual: {residuals.mean():.4f}")
print(f"   • Std of residuals: {residuals.std():.4f}")
print(f"   • Max overestimation: {residuals.min():.2f} kg/hectare")
print(f"   • Max underestimation: {residuals.max():.2f} kg/hectare")

# ============================================
# SAVE METRICS TO FILE
# ============================================

metrics_summary = {
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2_Score'],
    'Value': [mae, mse, rmse, r2]
}

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print(f"\n✅ Metrics saved to 'evaluation_metrics.csv'")

# ============================================
# VISUALIZATION (Optional)
# ============================================

# Create visualization plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_actual, y_predicted, alpha=0.6, edgecolors='k')
axes[0, 0].plot([y_actual.min(), y_actual.max()], 
                [y_actual.min(), y_actual.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Yield')
axes[0, 0].set_ylabel('Predicted Yield')
axes[0, 0].set_title('Actual vs Predicted Yield')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals Distribution
axes[0, 1].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals vs Predicted
axes[1, 0].scatter(y_predicted, residuals, alpha=0.6, edgecolors='k')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Yield')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Metrics Bar Chart
metrics_names = ['MAE', 'RMSE', 'R² Score']
metrics_values = [mae, rmse, r2]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 1].bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Key Evaluation Metrics')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(metrics_values):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
print(f"✅ Evaluation plots saved to 'evaluation_plots.png'")

print("\n" + "=" * 60)
print("Evaluation Complete! ✨")
print("=" * 60)
