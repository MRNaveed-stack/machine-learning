"""
Task 3.1: Linear Regression
Dataset: Canada Per Capita Income Dataset (Canada_per_capita_income (1).csv)

Requirements:
1. Load the Canada per capita income dataset.
2. Build a regression model using the given data.
3. Train the model using available historical data.
4. Predict Canada's per capita income for the year 2020.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the dataset
dataset_path = "Canada_per_capita_income (1).csv"
if not os.path.exists(dataset_path):
    # Fallback to alternate naming if needed
    dataset_path = "Canada_per_capita_income.csv"

df = pd.read_csv(dataset_path)

print("=" * 60)
print("TASK 3.1: CANADA PER CAPITA INCOME PREDICTION (LINEAR REGRESSION)")
print("=" * 60)
print("\nFirst 5 rows of dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Identify columns
# Typically: 'year' and 'income' or 'per capita income (US$)'
year_col = [c for c in df.columns if 'year' in c.lower()][0]
income_col = [c for c in df.columns if 'income' in c.lower() or 'capita' in c.lower()][0]

X = df[[year_col]]
y = df[income_col]

# 2. Build and train the linear regression model on available historical data
model = LinearRegression()
model.fit(X, y)

# 3. Model parameters and performance on historical data
y_pred_historical = model.predict(X)
r2 = r2_score(y, y_pred_historical)
mae = mean_absolute_error(y, y_pred_historical)
rmse = np.sqrt(mean_squared_error(y, y_pred_historical))

print("\nModel Parameters:")
print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Regression Equation: Income = {model.coef_[0]:.4f} * Year + ({model.intercept_:.4f})")
print(f"R-squared Score on historical data: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 4. Predict Canada's per capita income for the year 2020
year_2020 = pd.DataFrame({year_col: [2020]})
predicted_income_2020 = model.predict(year_2020)[0]

print("\n" + "-" * 50)
print(f"Predicted Canada's per capita income for 2020: ${predicted_income_2020:,.2f}")
print("-" * 50)

# Optional: Generate and save plot
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.scatter(df[year_col], df[income_col], color='blue', label='Actual Historical Data')
plt.plot(df[year_col], y_pred_historical, color='red', linewidth=2, label='Regression Trendline')
plt.scatter([2020], [predicted_income_2020], color='green', s=100, zorder=5, label=f'2020 Prediction: ${predicted_income_2020:,.2f}')
plt.title("Canada Per Capita Income Prediction (1970 - 2020)", fontsize=13, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Per Capita Income (US$)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plot_path = os.path.join("outputs", "task1_canada_income_regression.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nVisualization saved to: {plot_path}")
