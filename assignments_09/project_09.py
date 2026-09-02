# video link - https://youtu.be/cj6lvAiEkmE

# ============================================================
# Project 09 - Extract + Load Weather Pipeline
# ============================================================
import os
from datetime import date

import requests
from dotenv import load_dotenv
from supabase import create_client

# ============================================================
# Supabase Connection
# ============================================================

def get_client():
    """Create and return a Supabase client using credentials from .env."""

    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from the environment variables.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY is missing from the environment variables.")

    return create_client(supabase_url, supabase_key)

# ============================================================
# Step 1: Extract
# ============================================================

def extract_weather_data():
    """Fetch daily weather data for Fremont, CA for 2023."""

    # Fremont, California coordinates
    latitude = 37.5485
    longitude = -121.9886

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "wind_speed_10m_max"
        ),
        "timezone": "America/Los_Angeles"
    }

    response = requests.get(url, params=params)

    # Catch HTTP errors early
    response.raise_for_status()

    data = response.json()

    print("Weather data extracted successfully.")
    print("Response keys:", data.keys())
    print("Daily data keys:", data["daily"].keys())
    print("Number of dates:", len(data["daily"]["time"]))

    return data

# ============================================================
# Step 2: Transform
# ============================================================

def transform_weather_data(data):
    """Convert Open-Meteo columnar data into row dictionaries."""

    daily = data["daily"]

    records = []

    for i in range(len(daily["time"])):
        record = {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i]
        }

        records.append(record)

    print("First record:", records[0])
    print("Last record:", records[-1])
    print("Number of transformed records:", len(records))

    # A full year of 2023 has 365 days because 2023 was not a leap year.
    # We expect 365 records. If the number differs, the API date range,
    # missing data, or another API response issue could explain the difference.

    return records

# ============================================================
# Step 3: Load
# ============================================================

def load_weather_data(supabase, records):
    """Upsert weather records into the weather_raw table."""

    response = (
        supabase
        .table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Rows upserted: {len(response.data)}")

    return response.data

# ============================================================
# Step 4: Verify
# ============================================================

def verify_weather_data(supabase):
    """Verify the weather data loaded into Supabase."""

    # 1. Get all rows to determine total count
    response = (
        supabase
        .table("weather_raw")
        .select("*")
        .execute()
    )

    records = response.data

    print(f"Total rows in weather_raw: {len(records)}")

    # 2. Get earliest and latest dates
    if records:
        dates = [record["date"] for record in records]

        print("Earliest date:", min(dates))
        print("Latest date:", max(dates))

    # 3. Get the row for 2023-07-04
    july_4_response = (
        supabase
        .table("weather_raw")
        .select("*")
        .eq("date", "2023-07-04")
        .execute()
    )

    if july_4_response.data:
        print("Record for 2023-07-04:")
        print(july_4_response.data[0])
    else:
        print("No record found for 2023-07-04.")
        
# ============================================================
# Main Pipeline
# ============================================================

supabase = get_client()

print("Supabase client connected successfully!")

# Step 1: Extract
weather_data = extract_weather_data()

# Step 2: Transform
weather_records = transform_weather_data(weather_data)

# Step 3: Load
load_weather_data(supabase, weather_records)

# Step 4: Verify
verify_weather_data(supabase)

