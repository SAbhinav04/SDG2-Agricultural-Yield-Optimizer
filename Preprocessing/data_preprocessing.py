# member 1
import pandas as pd
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv("Dataset/crop_yield_with_crops.csv")

print(df.isnull().sum())
print(df.duplicated().sum())
print(df.head())

# member 2
X = df.drop("Yield", axis=1)
y = df["Yield"]
X = pd.get_dummies(X, columns=['Crop'], drop_first=True) 

# member 3
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# member 4
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.to_csv('cleaned_X.csv', index=False)
y.to_csv('y.csv', index=False) 

print(X.shape)
print(y.shape)

