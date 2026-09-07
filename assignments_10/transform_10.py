# video link - https://youtu.be/BMU8S4zTuN4

import os
import json
from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# Part 2: Project — The Double-Transform Pipeline
# Step 1: Incremental Read
# ============================================================

# Load environment variables from .env
load_dotenv()

# Connect to Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Load the model metadata
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]

print(f"Model features: {FEATURES}")


# Fetch all rows from weather_raw
raw_rows = (
    supabase
    .table("weather_raw")
    .select("*")
    .execute()
    .data
)

print(f"Raw records: {len(raw_rows)}")


# Fetch dates that are already present in weather_enriched
enriched_rows = (
    supabase
    .table("weather_enriched")
    .select("date")
    .execute()
    .data
)

already_done = {row["date"] for row in enriched_rows}

print(f"Already enriched: {len(already_done)}")


# Keep only records that have not been enriched yet
to_process = [
    row for row in raw_rows
    if row["date"] not in already_done
]

print(f"Records to process this run: {len(to_process)}")

if not to_process:
    print("No new records to process.")

    # ============================================================
    # Step 5: Verify
    # ============================================================

    verify_rows = (
        supabase
        .table("weather_enriched")
        .select("date, good_for_running, confidence, llm_summary")
        .execute()
        .data
    )

    print(f"Total rows in weather_enriched: {len(verify_rows)}")

    print("Five sample rows:")
    for row in verify_rows[:5]:
        print(row)

    good_days = sum(
        row["good_for_running"]
        for row in verify_rows
    )

    print(f"Days classified as good for running: {good_days}")

    # The LLM summaries should generally reflect the weather features
    # and the ML prediction. A weaker summary could result from the LLM
    # overemphasizing one weather condition or misinterpreting the prediction.

    exit()

# The LLM summaries generally reflect the weather features and ML predictions.
# The January 4 summary is a particularly good example because the model
# predicted that the day was not good for running, and the summary explains
# this using high precipitation and strong winds. The January 2 summary is
# weaker because it still recommends running despite some precipitation and
# wind. This may be because the LLM gave more weight to the mild temperatures
# than to the less favorable weather conditions.

# ============================================================
# Step 2: ML Transform
# ============================================================

import joblib
import pandas as pd

# Load the trained weather classifier
model = joblib.load("models/weather_classifier.pkl")

# Build a DataFrame from the records that need processing
df = pd.DataFrame(to_process)

# Select features in the exact order used during model training
X = df[FEATURES]

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Get the probability of the "good" class
good_class_index = list(model.classes_).index(1)
confidence = probabilities[:, good_class_index]

# Build enrichment records
enrichment_records = []

for row, prediction, conf in zip(to_process, predictions, confidence):
    enrichment_records.append({
        "date": row["date"],
        "good_for_running": bool(prediction),
        "confidence": float(conf)
    })

# Print prediction summary
good_count = sum(record["good_for_running"] for record in enrichment_records)

if enrichment_records:
    confidence_values = [
        record["confidence"] for record in enrichment_records
    ]

    print(f"Good running days: {good_count}")
    print(
        f"Confidence range: "
        f"{min(confidence_values):.4f} - {max(confidence_values):.4f}"
    )
else:
    print("No new records to process.")


# ============================================================
# Step 3: LLM Transform
# ============================================================

from openai import OpenAI

# Create OpenAI client
client = OpenAI()

# System prompt for the LLM
SYSTEM_PROMPT = """
You are a weather assistant. Based on the day's weather features
and the machine learning prediction, provide a one-sentence
recommendation about whether the day is good for running.
Keep the recommendation concise and practical.
"""

def create_user_message(row, prediction):
    """Create the user message containing weather data and ML prediction."""
    return f"""
Weather for {row['date']}:
- Maximum temperature: {row['temperature_2m_max']} °C
- Minimum temperature: {row['temperature_2m_min']} °C
- Precipitation: {row['precipitation_sum']} mm
- Maximum wind speed: {row['wind_speed_10m_max']} km/h
- ML prediction: {"Good for running" if prediction else "Not good for running"}

Give a one-sentence recommendation about whether this is a good day for running.
"""


# Call the LLM for each enrichment record
for i, (row, prediction) in enumerate(zip(to_process, predictions), start=1):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": create_user_message(row, prediction)
                }
            ],
            temperature=0
        )

        summary = response.choices[0].message.content.strip()

    except Exception as e:
        # Use a fallback instead of stopping the pipeline
        print(f"LLM API error: {e}")
        summary = "Unable to generate an LLM recommendation."
        break

    # Add the LLM summary to the enrichment record
    enrichment_records[i - 1]["llm_summary"] = summary

    # Show progress every 50 records
    if i % 50 == 0:
        print(f"Processed {i} LLM records")

print(f"LLM transformation complete: {len(enrichment_records)} records")
print(enrichment_records[:3])

# ============================================================
# Step 4: Load
# ============================================================

# Upsert enrichment records into weather_enriched
if enrichment_records:
    response = (
        supabase
        .table("weather_enriched")
        .upsert(enrichment_records)
        .execute()
    )

    print(f"Rows upserted: {len(response.data)}")
else:
    print("No enrichment records to upsert.")



# ============================================================
# Step 6: Reflect
# ============================================================
#
# The ML classifier was trained on weather data from Charlotte, NC,
# so its predictions may not be as accurate when applied to a different
# city because weather patterns and conditions can vary by location.
# The classifier learned relationships from Charlotte's climate, which
# may not apply as well to a city such as Fremont, CA.
#
# The LLM does not have the ability to override the ML classifier in this
# pipeline because the classifier's prediction is passed to the LLM as
# an input. The LLM is therefore purely additive: it provides a
# human-readable recommendation based on the weather features and the
# classifier's prediction. This makes the LLM useful for explaining the
# prediction, but it does not change the classifier's actual result.
#
# If the pipeline processed 50,000 records instead of 365, my main
# concern would be cost and latency from making an LLM API call for
# every record. I would address this by batching requests where possible,
# limiting unnecessary API calls, and potentially using a queue or
# asynchronous processing to improve throughput.

