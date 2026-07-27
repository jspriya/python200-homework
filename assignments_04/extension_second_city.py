# Train weather data on Miami

import requests
import pandas as pd
import json
import platform
import sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import os


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 25.7617,
    "longitude": -80.1918,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(df.head())
print(df.info())
print(df.describe())

# --- Step 2: Engineer Labels ---

# "Good for running" criteria for Fremont, CA
# - Daily high between 7°C and 28°C
# - Daily low at least 0°C
# - Less than 3 mm of precipitation
# - Maximum wind speed less than 30 km/h

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7)
    & (df["temperature_2m_max"] <= 28)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3.0)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

# Print class counts
print("Class counts:")
print(df["good_for_running"].value_counts())

print("\nClass percentages:")
print(df["good_for_running"].value_counts(normalize=True) * 100)

# Calculate the fraction of "good" days

good_fraction = df["good_for_running"].mean()

print(f"\nFraction of good running days: {good_fraction:.2%}")

# --------- Step 3: Train and Tune ----------

os.makedirs("outputs", exist_ok=True)

# Features and target
X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
])

# Parameter grid
param_grid = {
    "logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

# Print results
print("Best C:", grid_search.best_params_["logreg__C"])
print("Best CV AUC:", grid_search.best_score_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

test_auc = roc_auc_score(y_test, y_probs)
print("Test AUC:", test_auc)

# Plot ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.title("Weather Classifier ROC Curve")
plt.savefig("outputs/weather_roc.png")
plt.show()

# --- Step 4: Reflection on Evaluation ---
#
# The model achieved a test AUC of 0.9937, which indicates excellent
# classification performance. This means the model is very effective at
# distinguishing between days that are good for running and days that are not.
# The AUC is higher than I expected, suggesting that the four weather features
# are strong predictors of the target label.
#
# The classification report shows that false negatives are more common than
# false positives. The model correctly identified all good running days
# (recall = 1.00 for class 1), but it missed some of the not-good running
# days (recall = 0.76 for class 0). In practice, this means the app is more
# likely to recommend running on a day that is actually not ideal than to miss
# recommending a day that is good. I would prefer this behavior because users
# can still decide not to run if conditions are slightly worse than expected.
#
# For a real application, I would not necessarily use the default threshold
# of 0.5. If I wanted to be more conservative and avoid recommending running
# on poor-weather days, I would increase the threshold (for example, to around
# 0.6). This would reduce false positives, even if it meant occasionally
# missing some days that are still suitable for running.

# --- Step 5: Save the Model ---

# Save the trained pipeline
joblib.dump(best_model, "models/weather_classifier.pkl")

# Metadata
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": test_auc,
    "city": {
        "name": "Fremont, CA",
        "latitude": 37.5483,
        "longitude": -121.9886,
    },
    "label_thresholds": {
        "temperature_2m_max": "7°C to 28°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum": "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

# Save metadata as JSON
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Weather classifier saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")

# Comparison
# Metric	            Fremont, CA	    Miami, FL
# Good running days	    76.44%	        27.95%
# Best C	            10.0	        100.0
# Best CV AUC	        0.9574	        0.9918
# Test AUC	            0.9937	        0.9925

# --- Extension A: Second City Comparison ---
#
# Fremont had many more good running days (76.44%) than Miami (27.95%).
# This difference is expected because Fremont has a mild Mediterranean climate
# with moderate temperatures and relatively little rainfall, while Miami is
# much hotter and wetter. As a result, many more Miami days exceed the
# temperature threshold or receive enough precipitation to be labeled as
# unsuitable for running.
#
# Despite the large difference in class distribution, the model's test AUC
# remained very high (0.9937 for Fremont and 0.9925 for Miami). This suggests
# that the four weather features are strong predictors of the running label in
# both cities. The slight difference in AUC is likely due to differences in the
# underlying weather patterns and the distribution of examples in each dataset.

