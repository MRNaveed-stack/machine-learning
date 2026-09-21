import pandas as pd

df = pd.read_csv('iris.csv')

print("--- DataFrame Successfully Created ---")

print("\n[Activity 2] Initial observations (First 5 records):")
print(df.head())

print("\n[Activity 3] Final observations (Last 5 records):")
print(df.tail())

print("\n[Activity 4 & 5] Overall Structure, Dimensions, and Data Types:")
print(f"• Dataset Dimensions (Rows, Columns): {df.shape}")
print(f"• Column Names: {list(df.columns)}")
print("\n• Dataset Info Details:")
df.info()

print("\n[Activity 6] Statistical Summary of Numerical Features:")
measurement_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
numerical_features = df[measurement_columns]
print(numerical_features.describe())

print("\n[Activity 7] Calculating Feature Means:")
feature_means = numerical_features.mean()
print(feature_means)

highest_mean_feature = feature_means.idxmax()
highest_mean_value = feature_means.max()

print(f"\n🌟 The measurement feature with the highest average value is '{highest_mean_feature}' with a mean of {highest_mean_value:.4f} cm.")
