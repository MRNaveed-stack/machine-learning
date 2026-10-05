"""KNN classification for the Mobile Price Classification dataset."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent
training_data = pd.read_csv(DATA_DIR / "train.csv")
feature_columns = [column for column in training_data.columns if column != "price_range"]

x_train, x_test, y_train, y_test = train_test_split(
    training_data[feature_columns],
    training_data["price_range"],
    test_size=0.2,
    random_state=42,
    stratify=training_data["price_range"],
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("Lab Task 3: Mobile Price Classification Dataset")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

provided_test_data = pd.read_csv(DATA_DIR / "test.csv")
provided_predictions = model.predict(provided_test_data[feature_columns])
output = provided_test_data.copy()
output["predicted_price_range"] = provided_predictions
output.to_csv(DATA_DIR / "mobile_test_predictions.csv", index=False)
print("Predictions for test.csv saved to mobile_test_predictions.csv")