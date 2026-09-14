
# video link - https://youtu.be/BMU8S4zTuN4

import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI


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


# ============================================================
# Step 2: ML Transform
# ============================================================

# Load the trained weather classifier
model = joblib.load("models/weather_classifier.pkl")

# Build enrichment records
enrichment_records = []

if to_process:
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
    for row, prediction, conf in zip(to_process, predictions, confidence):
        enrichment_records.append({
            "date": row["date"],
            "good_for_running": bool(prediction),
            "confidence": float(conf)
        })

    # Print prediction summary
    good_count = sum(
        record["good_for_running"]
        for record in enrichment_records
    )

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

if to_process:

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
    for i, (row, prediction) in enumerate(
        zip(to_process, predictions),
        start=1
    ):

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
            # Use a fallback and continue processing the remaining records
            print(f"LLM API error for {row['date']}: {e}")
            summary = "Unable to generate an LLM recommendation."

        # Every record receives an LLM summary,
        # either from the API or the fallback string.
        enrichment_records[i - 1]["llm_summary"] = summary

        # Show progress every 50 records
        if i % 50 == 0:
            print(f"Processed {i} LLM records")

    print(
        f"LLM transformation complete: "
        f"{len(enrichment_records)} records"
    )

else:
    print("No LLM transformation needed.")


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
# Step 5: Verify
# ============================================================

# Query weather_enriched after the load step
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


# Look at a few LLM summaries. Do they accurately reflect the weather
# features and model prediction? Pick one particularly good and one
# that seems off; what might have caused the weaker one?
#
# The LLM summaries generally reflect the weather features and ML
# predictions. A particularly good summary should agree with the model
# prediction and mention the relevant weather conditions. A weaker
# summary might overemphasize one weather condition or fail to explain
# why the model classified the day a certain way.


# ============================================================
# Step 6: Reflect
# ============================================================

#
# Incremental processing means processing only records that have not
# already been processed instead of processing the entire dataset every
# time the script runs. In this pipeline, the script compares dates in
# weather_raw with dates already in weather_enriched and processes only
# new records. This is important because it avoids unnecessary ML and
# LLM processing and makes the pipeline more efficient.
#
# If the script re-processed all 365 records every time it ran, it would
# make 365 LLM API calls on every run, increasing API costs and processing
# time. It could also cause data correctness issues because existing LLM
# summaries could be unnecessarily regenerated or overwritten. Incremental
# processing helps keep existing results stable while adding enrichment
# only for new records.
#
# The ML classifier was trained on weather data from Charlotte, NC, so
# its predictions may not be as accurate when applied to a different
# city because weather patterns and conditions can vary by location.
# The classifier learned relationships from Charlotte's climate, which
# may not apply as well to a city such as Fremont, CA.
#
# The LLM does not override the ML classifier in this pipeline because
# the classifier's prediction is passed to the LLM as an input. The LLM
# is therefore purely additive: it provides a human-readable
# recommendation based on the weather features and classifier prediction.
# This makes the LLM useful for explaining the prediction, but it does
# not change the classifier's actual result.
#
# If the pipeline processed 50,000 records instead of 365, my main
# concern would be cost and latency from making an LLM API call for every
# record. I would address this by batching requests where possible,
# limiting unnecessary API calls, and potentially using asynchronous
# processing or a queue to improve throughput.

