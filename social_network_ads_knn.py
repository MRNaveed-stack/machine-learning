"""KNN classification for the Social Network Ads dataset."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent
data = pd.read_csv(DATA_DIR / "Social_Network_Ads.csv")
feature_columns = ["Age", "EstimatedSalary"]

x_train, x_test, y_train, y_test = train_test_split(
    data[feature_columns],
    data["Purchased"],
    test_size=0.2,
    random_state=42,
    stratify=data["Purchased"],
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("Lab Task 2: Social Network Ads Dataset")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

new_customers = pd.DataFrame(
    {
        "Age": [25, 35, 50],
        "EstimatedSalary": [30000, 60000, 100000],
    }
)
new_predictions = model.predict(new_customers)
print("New customer predictions (1 = purchase, 0 = no purchase):")
for customer, prediction in zip(new_customers.to_dict("records"), new_predictions):
    print(f"  {customer} -> {prediction}")