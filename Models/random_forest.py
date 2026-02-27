import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed feature matrix
X = pd.read_csv("preprocessing/cleaned_X.csv")

# Load target variable
y = pd.read_csv("preprocessing/y.csv")
y = y.squeeze()

# split parameters 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-Test Split :")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)