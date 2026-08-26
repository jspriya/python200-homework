import json
import joblib
import pandas as pd

# --- Task 1: Load and Verify ---

# Load the trained model
model = joblib.load("models/weather_classifier.pkl")

# Load metadata
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("=== Model Metadata ===")
print("City:", metadata["city"]["name"])
print(
    f"Latitude: {metadata['city']['latitude']}, "
    f"Longitude: {metadata['city']['longitude']}"
)
print("Features:", metadata["feature_names"])
print("Test AUC:", metadata["test_auc"])
print()

# --- Task 2: Predict on New Data ---

# Five hypothetical weather days
new_days = pd.DataFrame([
    # Clearly good
    [22, 12, 0.0, 12],

    # Clearly bad (hot)
    [36, 22, 0.0, 10],

    # Clearly bad (heavy rain)
    [18, 10, 12.0, 15],

    # Clearly bad (very windy)
    [20, 8, 0.0, 45],

    # Borderline
    [27, 2, 2.8, 28],
], columns=metadata["feature_names"])

predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)[:, 1]

print("=== Predictions ===")

for i, row in new_days.iterrows():
    print(f"\nDay {i + 1}")
    print(f"temperature_2m_max: {row['temperature_2m_max']} °C")
    print(f"temperature_2m_min: {row['temperature_2m_min']} °C")
    print(f"precipitation_sum: {row['precipitation_sum']} mm")
    print(f"wind_speed_10m_max: {row['wind_speed_10m_max']} km/h")

    label = "Good for running" if predictions[i] == 1 else "Skip running"

    print("Prediction:", label)
    print(f"Confidence: {probabilities[i]:.3f}")

# --- Step 5: Reflection ---
#
# I intended Day 5 to be a borderline case because its weather values are close
# to the thresholds used to define a good running day. However, the model
# predicted a probability of 0.000, indicating that it was very confident the
# day was not suitable for running. This suggests that the combination of
# temperature, precipitation, and wind was similar to examples labeled as
# "skip" during training.
#
# If the model produced a probability around 0.52 instead, I would consider
# that an uncertain prediction because it is very close to the default
# threshold of 0.5. In a real application, I might display a message such as
# "Conditions are borderline—check the detailed forecast before deciding."
#
# If someone ran predict_weather.py before train_weather_classifier.py, the
# model and metadata files would not exist, causing a FileNotFoundError when
# the script attempted to load them. A more helpful approach would be to catch
# the exception and display a message such as:
# "Model files not found. Please run train_weather_classifier.py first."
#
# In a production system, the prediction script would retrieve tomorrow's
# weather forecast from a weather API instead of using manually created sample
# data. It would then create a DataFrame with the required features, load the
# saved model, and generate predictions automatically each day.