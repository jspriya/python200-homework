# video link - https://youtu.be/T8EIMNbGZZc


import requests
from prefect import task ,flow

import os
import json
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

LATITUDE = 35.23
LONGITUDE = -80.84

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# **********************  Extract ********************

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": FEATURES,
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()["daily"]

    records = [
        {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]

    print(f"Extracted {len(records)} daily records from Open-Meteo")

    return records

# ***********************  Load weather_raw **********************

@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw")

# # ***********************  Transform **********************

@task
def transform(raw_records: list) -> list:
    # Find dates that have already been enriched
    response = (
        supabase.table("weather_enriched")
        .select("date")
        .execute()
    )

    already_done = {row["date"] for row in response.data}

    # Keep only records that have not been enriched yet
    to_process = [
        record for record in raw_records
        if record["date"] not in already_done
    ]

    print(
        f"Found {len(already_done)} already enriched records. "
        f"Processing {len(to_process)} new records."
    )

    if not to_process:
        return []

    # Load the saved sklearn Pipeline
    model = joblib.load("models/weather_classifier.pkl")

    # Load feature names from metadata
    with open("models/weather_classifier_metadata.json") as f:
        metadata = json.load(f)

    features = metadata["feature_names"]

    # Create feature data in the order expected by the model
    X = [
        [record[feature] for feature in features]
        for record in to_process
    ]

    # Run ML predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    enrichment_records = []

    for i, record in enumerate(to_process):
        good_for_running = bool(predictions[i])

        # Probability of the predicted class
        confidence = float(max(probabilities[i]))

        messages = [
            {
                "role": "system",
                "content": (
                    "Give a one-sentence recommendation about whether "
                    "this weather is good for running."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Temperature maximum: {record['temperature_2m_max']} C, "
                    f"temperature minimum: {record['temperature_2m_min']} C, "
                    f"precipitation: {record['precipitation_sum']} mm, "
                    f"wind speed: {record['wind_speed_10m_max']} km/h. "
                    f"ML prediction: "
                    f"{'good' if good_for_running else 'not good'}."
                ),
            },
        ]

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            llm_summary = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM error for {record['date']}: {e}")
            llm_summary = "LLM recommendation unavailable."

        enrichment_records.append(
            {
                "date": record["date"],
                "good_for_running": good_for_running,
                "confidence": confidence,
                "llm_summary": llm_summary,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} of {len(to_process)} records.")

    print(f"Created {len(enrichment_records)} enrichment records.")

    return enrichment_records

# ********************  Load weather_enriched  *******************

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    if not enrichment_records:
        print("No enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )

    print(
        f"Upserted {len(response.data)} rows into weather_enriched"
    )

# ********************* Flow  *****************************

@flow(log_prints=True)
def weather_pipeline():
    records = extract()
    load_raw(records)

    enrichment_records = transform(records)
    load_enriched(enrichment_records)

    print("Weather pipeline completed successfully; skipping weather_enriched.")

if __name__ == "__main__":
    weather_pipeline()














