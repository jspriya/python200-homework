# ------------ Supabase Connection ---------------

# ------------ Connection Question 1 ---------------

#The two pieces of information supabase-py needs to connect to a
#Supabase project are the Supabase project URL and the Supabase API key.

#They are foud in the Supabase dashboard under:
#Project Settings → API.

#They should never be hardcoded in a Python script because the API key
#is sensitive information. Hardcoding credentials can expose them if
#the code is shared or pushed to a public GitHub repository. Instead,
#they should be stored securely in a .env file and loaded using
#environment variables.

# ------------------  Connection Question 2 -----------------

from dotenv import load_dotenv
import os
from supabase import create_client


def get_client():
    """Create and return a Supabase client using credentials from .env."""
    
    load_dotenv()
#   print("Supabase URL:", os.getenv("SUPABASE_URL"))

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from the environment variables.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY is missing from the environment variables.")

    return create_client(supabase_url, supabase_key)

supabase= get_client()

print("Supabase client connected successfully!")

# ----------------- Connection Question 3 ----------------

#Row Level Security (RLS) is a security feature in Supabase/PostgreSQL
#that controls which rows a user is allowed to access or modify based
#on security policies.

#are focused on learning how to connect to and work with Supabase rather
#than implementing user authentication and row-level security policies.
#Disabling RLS makes it easier for the Python code to read and write data
#during these exercises.

#In a real-world application, I would keep RLS enabled when the database
#contains user-specific or sensitive data. For example, in a banking,
#healthcare, or social media application, RLS could ensure that each
#authenticated user can only access the rows they are authorized to see
#or modify.

#========================================================================================================

# ---------------  supabase-py CRUD  --------------------

# -------------- CRUD Question 1 ------------

from datetime import date


def insert_test_record(supabase):
    """Insert one test weather record for today's date."""
   
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 75.0,
        "temperature_2m_min": 55.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0
    }

    response = supabase.table("weather_raw").insert(record).execute()

    print("Inserted record:", response.data)
    return response.data

# supabase = get_client()


# Test the function
# insert_test_record(supabase)

#If I ran this function twice, the second insert would fail because
#date is the primary key, and today's date would already exist in the
#weather_raw table.

#To make the operation safe to run multiple times, I could use upsert
#instead of insert. Since date is the primary key, Supabase can use date
#as the conflict key and update the existing record instead of creating
#a duplicate.

# ------------- CRUD Question 2 --------------

def get_records_by_date_range(supabase, start, end):
    """Return all weather records between start and end dates, inclusive."""

    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )

    return response.data

supabase = get_client()

# Q1
# insert_test_record(supabase) 

# Q2
# Test with today's date
today = date.today().isoformat()

records = get_records_by_date_range(
    supabase,
    today,
    today
)

print("Records found:")
print(records)

# ---------------- CRUD Question 3 ---------------------

#The difference between insert and upsert is that insert attempts to
#create a new row. If a row with the same primary key already exists,
#the insert will fail.

#Upsert means "update or insert." If a row with the same conflict key
#already exists, upsert updates that row. If no matching row exists,
#it inserts a new row.

#For example, I would use insert when adding a new weather record for
#a date that does not already exist in weather_raw.

#I would use upsert when loading weather data repeatedly. If weather data
#for a particular date already exists, I would want to update that record
#instead of creating a duplicate.

#Since date is the primary key in weather_raw, I can use date as the
#conflict key for the upsert operation.


def safe_upsert(supabase, records):
    """Upsert weather records using date as the conflict key."""

    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Rows affected: {len(response.data)}")
    return response.data

test_records = [
    {
        "date": date.today().isoformat(),
        "temperature_2m_max": 78.0,
        "temperature_2m_min": 58.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.0
    }
]

safe_upsert(supabase, test_records)

# ===================================================================================================

# --------------  Idempotency  --------------------
# ------------Idempotency Question 1 ---------------

#Idempotency means that running the same operation multiple times produces
#the same final result as running it once.

#Idempotency matters in a data pipeline because pipelines can fail, be
#interrupted, or need to be restarted. If an operation is idempotent, we
#can safely rerun it without creating duplicate or incorrect data.

#For example, suppose a weather pipeline loads 30 days of weather data
#into the weather_raw table. If the pipeline successfully loads 15 days
#and then crashes, restarting a non-idempotent pipeline might insert those
#same 15 days again. This could create duplicate records and incorrect
#data.

#In this project, using the date as the primary key and using upsert makes
#the loading process idempotent. If the pipeline is restarted, existing
#dates are updated rather than duplicated, while missing dates are inserted.


