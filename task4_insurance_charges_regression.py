

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("TASK 3.4: MEDICAL INSURANCE CHARGES VS AGE (LINEAR REGRESSION)")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv("insurance.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 2. Independent variable = age, Dependent variable = charges
X = df[['age']]
y = df['charges']

# 3. Divide into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Train regression model using training data
model = LinearRegression()
model.fit(X_train, y_train)

# Model predictions on test set
y_pred = model.predict(X_test)

# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"Mean Absolute Error (MAE) : ${mae:,.2f}")
print(f"Mean Squared Error (MSE)  : {mse:,.2f}")
print(f"Root Mean Squared Error   : ${rmse:,.2f}")
print(f"R-squared (R2 Score)      : {r2:.4f}")

# 7. Display regression coefficient and intercept
print("\n" + "=" * 50)
print("MODEL PARAMETERS")
print("=" * 50)
print(f"Regression Coefficient (Slope) : {model.coef_[0]:.4f}")
print(f"Model Intercept                : {model.intercept_:.4f}")
print(f"Equation: Charges ($) = {model.coef_[0]:.4f} * Age + ({model.intercept_:.4f})")

# 5. Predict medical insurance charges for a person who is 40 years old
pred_40 = pd.DataFrame({'age': [40]})
predicted_charge_40 = model.predict(pred_40)[0]

print("\n" + "=" * 50)
print("PREDICTION FOR AGE = 40 YEARS")
print("=" * 50)
print(f"Predicted Medical Insurance Charges: ${predicted_charge_40:,.2f}")

# 8. Plot actual data points and regression line
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['charges'], color='#5C6BC0', alpha=0.5, label='Actual Insurance Charges')

# Regression line
age_range = np.linspace(df['age'].min(), df['age'].max(), 100).reshape(-1, 1)
age_range_df = pd.DataFrame(age_range, columns=['age'])
charges_line = model.predict(age_range_df)
plt.plot(age_range, charges_line, color='red', linewidth=2.5, label='Fitted Regression Line')

# Highlight age 40 prediction
plt.scatter([40], [predicted_charge_40], color='green', s=120, zorder=5,
            label=f'Age 40 Prediction: ${predicted_charge_40:,.2f}')

plt.title("Medical Insurance Charges vs Age (Simple Linear Regression)", fontsize=14, fontweight='bold')
plt.xlabel("Age (Years)", fontsize=12)
plt.ylabel("Insurance Charges ($)", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join("outputs", "task4_insurance_charges_regression.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {plot_path}")
