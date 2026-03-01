"""
Model Comparison Script
Compares Random Forest and Linear Regression models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt

# ============================================
# LOAD BOTH MODELS' RESULTS
# ============================================

rf_results = pd.read_csv('rf_results.csv')
lr_results = pd.read_csv('lr_results.csv')

# ============================================
# CALCULATE METRICS FOR BOTH MODELS
# ============================================

def calculate_metrics(actual, predicted, model_name):
    """Calculate all evaluation metrics for a model"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2
    }

# Calculate metrics
rf_metrics = calculate_metrics(rf_results['Actual'], rf_results['Predicted'], 'Random Forest')
lr_metrics = calculate_metrics(lr_results['Actual'], lr_results['Predicted'], 'Linear Regression')

# Create comparison dataframe
comparison_df = pd.DataFrame([rf_metrics, lr_metrics])

# ============================================
# PRINT COMPARISON
# ============================================

print("=" * 80)
print("MODEL COMPARISON - RANDOM FOREST vs LINEAR REGRESSION")
print("=" * 80)

print("\n📊 EVALUATION METRICS COMPARISON:\n")
print(comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("DETAILED ANALYSIS:")
print("=" * 80)

print("\n🎯 Mean Absolute Error (MAE) - Lower is better:")
print(f"   Random Forest:      {rf_metrics['MAE']:.4f} kg/hectare")
print(f"   Linear Regression:  {lr_metrics['MAE']:.4f} kg/hectare")
if rf_metrics['MAE'] < lr_metrics['MAE']:
    diff = ((lr_metrics['MAE'] - rf_metrics['MAE']) / lr_metrics['MAE']) * 100
    print(f"   ❌ Random Forest is WORSE by {diff:.2f}%")
else:
    diff = ((rf_metrics['MAE'] - lr_metrics['MAE']) / rf_metrics['MAE']) * 100
    print(f"   ✅ Linear Regression is BETTER by {diff:.2f}%")

print("\n🎯 Root Mean Squared Error (RMSE) - Lower is better:")
print(f"   Random Forest:      {rf_metrics['RMSE']:.4f} kg/hectare")
print(f"   Linear Regression:  {lr_metrics['RMSE']:.4f} kg/hectare")
if rf_metrics['RMSE'] < lr_metrics['RMSE']:
    diff = ((lr_metrics['RMSE'] - rf_metrics['RMSE']) / lr_metrics['RMSE']) * 100
    print(f"   ❌ Random Forest is WORSE by {diff:.2f}%")
else:
    diff = ((rf_metrics['RMSE'] - lr_metrics['RMSE']) / rf_metrics['RMSE']) * 100
    print(f"   ✅ Linear Regression is BETTER by {diff:.2f}%")

print("\n🎯 R² Score - Higher is better:")
print(f"   Random Forest:      {rf_metrics['R2_Score']:.4f} ({rf_metrics['R2_Score']*100:.2f}% variance explained)")
print(f"   Linear Regression:  {lr_metrics['R2_Score']:.4f} ({lr_metrics['R2_Score']*100:.2f}% variance explained)")
if rf_metrics['R2_Score'] > lr_metrics['R2_Score']:
    diff = ((rf_metrics['R2_Score'] - lr_metrics['R2_Score']) / lr_metrics['R2_Score']) * 100
    print(f"   ❌ Random Forest is WORSE by {abs(diff):.2f}%")
else:
    diff = ((lr_metrics['R2_Score'] - rf_metrics['R2_Score']) / rf_metrics['R2_Score']) * 100
    print(f"   ✅ Linear Regression is BETTER by {diff:.2f}%")

print("\n" + "=" * 80)
print("WINNER:")
print("=" * 80)

# Determine winner based on R² score (most important metric)
if lr_metrics['R2_Score'] > rf_metrics['R2_Score']:
    print(f"\n🏆 LINEAR REGRESSION is the BETTER model!")
    print(f"   • Better R² Score: {lr_metrics['R2_Score']:.4f} vs {rf_metrics['R2_Score']:.4f}")
    print(f"   • Lower MAE: {lr_metrics['MAE']:.2f} vs {rf_metrics['MAE']:.2f}")
    print(f"   • Lower RMSE: {lr_metrics['RMSE']:.2f} vs {rf_metrics['RMSE']:.2f}")
else:
    print(f"\n🏆 RANDOM FOREST is the BETTER model!")
    print(f"   • Better R² Score: {rf_metrics['R2_Score']:.4f} vs {lr_metrics['R2_Score']:.4f}")

print("\n" + "=" * 80)

# ============================================
# SAVE COMPARISON TO CSV
# ============================================

comparison_df.to_csv('model_comparison.csv', index=False)
print(f"\n✅ Comparison saved to 'model_comparison.csv'")

# ============================================
# VISUALIZATION
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison: Random Forest vs Linear Regression', fontsize=18, fontweight='bold')

# Plot 1: MAE Comparison
axes[0, 0].bar(['Random Forest', 'Linear Regression'], 
               [rf_metrics['MAE'], lr_metrics['MAE']], 
               color=['#FF6B6B', '#4ECDC4'], 
               edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('MAE (kg/hectare)')
axes[0, 0].set_title('Mean Absolute Error (Lower is Better)')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([rf_metrics['MAE'], lr_metrics['MAE']]):
    axes[0, 0].text(i, v + 10, f'{v:.2f}', ha='center', fontweight='bold')

# Plot 2: RMSE Comparison
axes[0, 1].bar(['Random Forest', 'Linear Regression'], 
               [rf_metrics['RMSE'], lr_metrics['RMSE']], 
               color=['#FF6B6B', '#4ECDC4'], 
               edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('RMSE (kg/hectare)')
axes[0, 1].set_title('Root Mean Squared Error (Lower is Better)')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([rf_metrics['RMSE'], lr_metrics['RMSE']]):
    axes[0, 1].text(i, v + 10, f'{v:.2f}', ha='center', fontweight='bold')

# Plot 3: R² Score Comparison
axes[1, 0].bar(['Random Forest', 'Linear Regression'], 
               [rf_metrics['R2_Score'], lr_metrics['R2_Score']], 
               color=['#FF6B6B', '#4ECDC4'], 
               edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('R² Score (Higher is Better)')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([rf_metrics['R2_Score'], lr_metrics['R2_Score']]):
    axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 4: Actual vs Predicted for both models
axes[1, 1].scatter(rf_results['Actual'], rf_results['Predicted'], 
                   alpha=0.6, label='Random Forest', color='#FF6B6B', edgecolors='k')
axes[1, 1].scatter(lr_results['Actual'], lr_results['Predicted'], 
                   alpha=0.6, label='Linear Regression', color='#4ECDC4', edgecolors='k')
axes[1, 1].plot([rf_results['Actual'].min(), rf_results['Actual'].max()], 
                [rf_results['Actual'].min(), rf_results['Actual'].max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Yield')
axes[1, 1].set_ylabel('Predicted Yield')
axes[1, 1].set_title('Actual vs Predicted: Both Models')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
print(f"✅ Comparison plots saved to 'model_comparison_plots.png'")

print("\n" + "=" * 80)
print("Comparison Complete! ✨")
print("=" * 80)
