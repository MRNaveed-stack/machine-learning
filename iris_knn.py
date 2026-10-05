"""KNN classification for the Iris dataset."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent

data = pd.read_csv(DATA_DIR / "Iris.csv")
feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

x_train, x_test, y_train, y_test = train_test_split(
    data[feature_columns],
    data["Species"],
    test_size=0.2,
    random_state=42,
    stratify=data["Species"],
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("Lab Task 1: Iris Dataset")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))