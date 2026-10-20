

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# If word2number is available, we use it; otherwise fallback mapping
try:
    from word2number import w2n
    def parse_experience(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        try:
            return w2n.word_to_num(str(val))
        except Exception:
            return 0
except ImportError:
    word_dict = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12
    }
    def parse_experience(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        val_str = str(val).strip().lower()
        return word_dict.get(val_str, 0)

print("=" * 60)
print("TASK 3.2: HIRING SALARY PREDICTION (MULTIPLE LINEAR REGRESSION)")
print("=" * 60)

# 1. Load Dataset
df = pd.read_csv("hiring.csv")
print("\nOriginal Dataset:")
print(df)

# Column names cleaning/mapping
# ['experience', 'test_score(out of 10)', 'interview_score(out of 10)', 'salary($)']
exp_col = [c for c in df.columns if 'exp' in c.lower()][0]
test_col = [c for c in df.columns if 'test' in c.lower()][0]
interview_col = [c for c in df.columns if 'interview' in c.lower()][0]
salary_col = [c for c in df.columns if 'salary' in c.lower()][0]

# 2. Data Preprocessing / Handling Missing Values
# - Experience NaN converted to 0, and string numbers converted to integer
df[exp_col] = df[exp_col].apply(parse_experience)

# - Test score missing value imputed with median / mean (floor)
test_median = math.floor(df[test_col].median())
df[test_col] = df[test_col].fillna(test_median)

# - Interview score (if any missing) filled with median
interview_median = math.floor(df[interview_col].median())
df[interview_col] = df[interview_col].fillna(interview_median)

print("\nProcessed Dataset (Cleaned & Imputed):")
print(df)

# Features and Target
X = df[[exp_col, test_col, interview_col]]
y = df[salary_col]

# 3. Build & Train Multiple Linear Regression Model
model = LinearRegression()
model.fit(X, y)

print("\nModel Parameters:")
print(f"Intercept: {model.intercept_:.2f}")
feature_names = ["Experience (years)", "Test Score (out of 10)", "Interview Score (out of 10)"]
for name, coef in zip(feature_names, model.coef_):
    print(f"Coefficient for {name}: {coef:.2f}")

# 4. Predict salary for Candidates A and B
# Candidate A: 2 years experience, 9 test score, 6 interview score
# Candidate B: 12 years experience, 10 test score, 10 interview score
candidate_a_features = pd.DataFrame([[2, 9, 6]], columns=[exp_col, test_col, interview_col])
candidate_b_features = pd.DataFrame([[12, 10, 10]], columns=[exp_col, test_col, interview_col])

pred_a = model.predict(candidate_a_features)[0]
pred_b = model.predict(candidate_b_features)[0]

print("\n" + "=" * 50)
print("PREDICTION RESULTS")
print("=" * 50)
print(f"Candidate A (2 yrs exp, 9 test, 6 interview): ${pred_a:,.2f}")
print(f"Candidate B (12 yrs exp, 10 test, 10 interview): ${pred_b:,.2f}")

# 5. Interpretation & Feature Influence
# Compute normalized coefficients (standardized) to rigorously determine strongest influence
X_std = (X - X.mean()) / X.std()
y_std = (y - y.mean()) / y.std()
std_model = LinearRegression().fit(X_std, y_std)

print("\n" + "=" * 50)
print("INTERPRETATION & FEATURE INFLUENCE EXPLANATION")
print("=" * 50)
print("Raw Coefficients (Dollar impact per 1-unit increase):")
for name, coef in zip(feature_names, model.coef_):
    print(f"  • {name}: +${coef:,.2f}")

print("\nStandardized Coefficients (Beta weights - directly comparable):")
for name, beta in zip(feature_names, std_model.coef_):
    print(f"  • {name}: {beta:.4f}")

strongest_feature_idx = np.argmax(np.abs(std_model.coef_))
print(f"\nStrongest Influence: '{feature_names[strongest_feature_idx]}' has the strongest influence on salary.")
print("Reasoning: Both in raw unit change and in standardized terms, each additional unit leads to the highest increase in expected salary.")
