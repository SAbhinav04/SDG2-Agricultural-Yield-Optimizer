import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================
# Data Loading
# ==============================

# Load preprocessed feature matrix
X = pd.read_csv("Preprocessing/cleaned_X.csv")
y = pd.read_csv("Preprocessing/y.csv")
y = y.squeeze()   # Convert to 1D


print("Original Data Shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ==============================
# Train-Test Split
# ==============================

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


# ==============================
# Next Step (To be implemented by teammate)
# Model Training & Result Export
# ==============================