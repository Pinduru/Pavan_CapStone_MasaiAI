"""Load the saved pipeline and make a prediction from raw passenger data."""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "best_pipeline.joblib")

passenger = pd.DataFrame(
    [
        {
            "pclass": 2,
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 25.0,
            "sex": "female",
            "embarked": "S",
        }
    ]
)

prediction = int(model.predict(passenger)[0])
probability = float(model.predict_proba(passenger)[0, 1])

print(f"Predicted class: {prediction}")
print(f"Survival probability: {probability:.4f}")
