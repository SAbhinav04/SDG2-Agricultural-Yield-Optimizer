import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load preprocessed feature matrix
X = pd.read_csv("../Preprocessing/cleaned_X.csv")

# Load target variable
y = pd.read_csv("../Preprocessing/y.csv")
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

# 3. INITIALIZE RANDOM FOREST MODEL

rf_model = RandomForestRegressor(n_estimators=300, random_state=42 , max_depth=10)


# 4. TRAIN MODEL


rf_model.fit(X_train, y_train)


# 5. GENERATE PREDICTIONS


y_pred = rf_model.predict(X_test)


# 6. SAVE RESULTS FOR EVALUATION TEAM


results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results_df.to_csv('../Evaluation/rf_results.csv', index=False)


# 7. SAVE TRAINED MODEL


joblib.dump(rf_model, '../Models/rf_model.pkl')


# 8. PRINT METRICS


r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nRandom Forest Model Training Complete")
print("R2 Score:", r2)
print("RMSE:", rmse)
print("Model saved as rf_model.pkl")