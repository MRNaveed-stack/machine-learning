
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("TASK 3.3: HEIGHT AND WEIGHT SIMPLE LINEAR REGRESSION")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv("weight-height.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Columns: Height (in inches), Weight (in lbs)
height_col = [c for c in df.columns if 'height' in c.lower()][0]
weight_col = [c for c in df.columns if 'weight' in c.lower()][0]

X = df[[height_col]]
y = df[weight_col]

# 2. Divide dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 3. Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluation on test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"Mean Absolute Error (MAE) : {mae:.4f} lbs")
print(f"Mean Squared Error (MSE)  : {mse:.4f} lbs^2")
print(f"Root Mean Squared Error   : {rmse:.4f} lbs")
print(f"R-squared (R2 Score)      : {r2:.4f}")

# 6. Display regression coefficient and intercept
print("\n" + "=" * 50)
print("MODEL PARAMETERS")
print("=" * 50)
print(f"Regression Coefficient (Slope) : {model.coef_[0]:.4f}")
print(f"Model Intercept                : {model.intercept_:.4f}")
print(f"Equation: Weight(lbs) = {model.coef_[0]:.4f} * Height(inches) + ({model.intercept_:.4f})")

# 4. Predict the weight of a person whose height is 170 cm
# 1 inch = 2.54 cm -> 170 cm = 170 / 2.54 inches
height_170cm_inches = 170.0 / 2.54
pred_input = pd.DataFrame({height_col: [height_170cm_inches]})
pred_weight_lbs = model.predict(pred_input)[0]
pred_weight_kg = pred_weight_lbs * 0.45359237

print("\n" + "=" * 50)
print("PREDICTION FOR HEIGHT = 170 cm")
print("=" * 50)
print(f"Height in inches: {height_170cm_inches:.2f} inches")
print(f"Predicted Weight: {pred_weight_lbs:.2f} lbs ({pred_weight_kg:.2f} kg)")

# 7. Plot graph showing actual data points and regression line
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(10, 6))
# Sample 1000 points for clearer visualization without clutter
sample_df = df.sample(n=min(1000, len(df)), random_state=42)
plt.scatter(sample_df[height_col], sample_df[weight_col], color='#4A90E2', alpha=0.4, label='Actual Data Points (Sample)')

# Regression line
x_vals = np.linspace(df[height_col].min(), df[height_col].max(), 100).reshape(-1, 1)
x_vals_df = pd.DataFrame(x_vals, columns=[height_col])
y_line = model.predict(x_vals_df)
plt.plot(x_vals, y_line, color='red', linewidth=2.5, label='Fitted Regression Line')

# Highlight 170 cm point
plt.scatter([height_170cm_inches], [pred_weight_lbs], color='green', s=120, zorder=5,
            label=f'Prediction (170cm): {pred_weight_lbs:.1f} lbs')

plt.title("Height vs Weight Simple Linear Regression", fontsize=14, fontweight='bold')
plt.xlabel("Height (inches)", fontsize=12)
plt.ylabel("Weight (lbs)", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join("outputs", "task3_height_weight_regression.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {plot_path}")
