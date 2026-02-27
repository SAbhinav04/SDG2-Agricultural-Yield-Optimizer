import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Load preprocessed feature matrix
X = pd.read_csv("Preprocessing/cleaned_X.csv")
y = pd.read_csv("Preprocessing/y.csv")
y = y.squeeze()

print("Original Data Shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Model Training & Result Export

print("\nStarting Model Training...")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Training Completed.")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

joblib.dump(model, "linear_regression_model.pkl")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("linear_regression_predictions.csv", index=False)

print("\nModel and predictions saved successfully.")
