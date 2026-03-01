"""
Model Comparison Script
Compares all available models that have result files in the Evaluation folder.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

RESULT_FILES = [
    ("Random Forest", "rf_results.csv"),
    ("Linear Regression", "lr_results.csv"),
    ("Improved Random Forest", "improved_rf_results.csv"),
    ("Advanced Model", "advanced_model_results.csv"),
]


def calculate_metrics(actual, predicted, model_name):
    """Calculate all evaluation metrics for a model."""
    mse = mean_squared_error(actual, predicted)
    return {
        'Model': model_name,
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2_Score': r2_score(actual, predicted)
    }


def load_available_results():
    """Load only result files that currently exist."""
    loaded = {}
    for model_name, filename in RESULT_FILES:
        if os.path.exists(filename):
            loaded[model_name] = pd.read_csv(filename)
    return loaded


results_by_model = load_available_results()

if len(results_by_model) < 2:
    raise ValueError(
        "At least two result files are required for comparison. "
        "Available files: "
        + ", ".join(
            [filename for _, filename in RESULT_FILES if os.path.exists(filename)]
        )
    )

metrics_rows = []
for model_name, df in results_by_model.items():
    metrics_rows.append(calculate_metrics(df['Actual'], df['Predicted'], model_name))

comparison_df = pd.DataFrame(metrics_rows)
comparison_df = comparison_df.sort_values(
    by=['R2_Score', 'RMSE', 'MAE'],
    ascending=[False, True, True]
).reset_index(drop=True)

print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)
print("\nAvailable models compared:")
for model_name in comparison_df['Model']:
    print(f"  - {model_name}")

print("\nEvaluation Metrics Comparison:\n")
print(comparison_df.to_string(index=False))

winner = comparison_df.iloc[0]
print("\n" + "=" * 90)
print("WINNER (by highest R², then lowest RMSE/MAE):")
print("=" * 90)
print(f"\nBest Model: {winner['Model']}")
print(f"  - R² Score: {winner['R2_Score']:.4f}")
print(f"  - RMSE: {winner['RMSE']:.4f} kg/hectare")
print(f"  - MAE: {winner['MAE']:.4f} kg/hectare")
print("\n" + "=" * 90)

# Save comparison metrics
comparison_df.to_csv('model_comparison.csv', index=False)
print("\nComparison saved to 'model_comparison.csv'")

# ============================================
# VISUALIZATION
# ============================================

models = comparison_df['Model'].tolist()
mae_values = comparison_df['MAE'].tolist()
rmse_values = comparison_df['RMSE'].tolist()
r2_values = comparison_df['R2_Score'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Model Comparison', fontsize=18, fontweight='bold')

# Plot 1: MAE Comparison
axes[0, 0].bar(models, mae_values, color='#F28E8E', edgecolor='black', alpha=0.8)
axes[0, 0].set_ylabel('MAE (kg/hectare)')
axes[0, 0].set_title('Mean Absolute Error (Lower is Better)')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].tick_params(axis='x', rotation=20)
for i, value in enumerate(mae_values):
    axes[0, 0].text(i, value + max(mae_values) * 0.01, f'{value:.2f}', ha='center')

# Plot 2: RMSE Comparison
axes[0, 1].bar(models, rmse_values, color='#7BC8A4', edgecolor='black', alpha=0.8)
axes[0, 1].set_ylabel('RMSE (kg/hectare)')
axes[0, 1].set_title('Root Mean Squared Error (Lower is Better)')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].tick_params(axis='x', rotation=20)
for i, value in enumerate(rmse_values):
    axes[0, 1].text(i, value + max(rmse_values) * 0.01, f'{value:.2f}', ha='center')

# Plot 3: R² Score Comparison
axes[1, 0].bar(models, r2_values, color='#72A0C1', edgecolor='black', alpha=0.8)
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('R² Score (Higher is Better)')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].tick_params(axis='x', rotation=20)
for i, value in enumerate(r2_values):
    axes[1, 0].text(i, value + 0.02, f'{value:.4f}', ha='center')

# Plot 4: Actual vs Predicted for all models
min_actual = None
max_actual = None
for model_name, df in results_by_model.items():
    axes[1, 1].scatter(
        df['Actual'],
        df['Predicted'],
        alpha=0.6,
        label=model_name,
        edgecolors='k'
    )
    if min_actual is None:
        min_actual = df['Actual'].min()
        max_actual = df['Actual'].max()
    else:
        min_actual = min(min_actual, df['Actual'].min())
        max_actual = max(max_actual, df['Actual'].max())

axes[1, 1].plot([min_actual, max_actual], [min_actual, max_actual], 'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Yield')
axes[1, 1].set_ylabel('Predicted Yield')
axes[1, 1].set_title('Actual vs Predicted: All Models')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
print("Comparison plots saved to 'model_comparison_plots.png'")

print("\n" + "=" * 90)
print("Comparison Complete")
print("=" * 90)
