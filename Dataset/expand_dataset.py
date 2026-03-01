import pandas as pd
import numpy as np

np.random.seed(42)

n = 4000

crops = ["Rice", "Wheat", "Maize", "Cotton"]

crop = np.random.choice(crops, n)

# Anchor-based nutrient ranges
nitrogen = np.random.uniform(40, 120, n)
phosphorus = np.random.uniform(20, 60, n)
potassium = np.random.uniform(20, 80, n)
rainfall = np.random.uniform(100, 300, n)

# Anchor-based yield logic
yield_kg = (
    18 * nitrogen +
    12 * phosphorus +
    10 * potassium +
    5 * rainfall +
    np.random.normal(0, 250, n)
)

# Clip yield to realistic bounds
yield_kg = np.clip(yield_kg, 1500, 7000)

df = pd.DataFrame({
    "Crop": crop,
    "Nitrogen": nitrogen,
    "Phosphorus": phosphorus,
    "Potassium": potassium,
    "Rainfall": rainfall,
    "Yield": yield_kg
})

df.to_csv("anchor_based_crop_dataset.csv", index=False)

print("Anchor-based dataset created:", df.shape)