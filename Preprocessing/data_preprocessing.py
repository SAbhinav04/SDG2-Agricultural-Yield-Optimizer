# member 1
import pandas as pd
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv("C:\Users\Vijay kumar\Downloads\crop_yield_with_crops.csv")

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

