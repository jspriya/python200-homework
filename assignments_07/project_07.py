from pathlib import Path
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import CodeAgent, OpenAIServerModel, tool

# ------------------ Setup ------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

df = None

def get_loaded_dataframe():
    """Return the currently loaded happiness DataFrame."""
    return df

# ----------------   Task 1 — Define the four tools -------------

# Tool 1a: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged CSV from DATA_PATH if it exists. If the merged
    file does not exist, loads and merges the yearly CSV files from
    the happiness_project resources directory.

    Returns:
        A dictionary indicating whether the data was loaded successfully.
    """
    global df

    path = Path(DATA_PATH)

    try:
        # Try the merged dataset first
        if path.exists():
            df = pd.read_csv(path)

        else:
            # Fall back to yearly CSV files
            resources_path = Path(
                "../assignments/resources/happiness_project/"
            )

            csv_files = sorted(resources_path.glob("*.csv"))

            if not csv_files:
                return {
                    "error": (
                        f"Could not find the merged dataset at {DATA_PATH} "
                        "and no yearly CSV files were found."
                    )
                }

            yearly_data = []

            for csv_file in csv_files:
                yearly_df = pd.read_csv(csv_file)
                yearly_data.append(yearly_df)

            df = pd.concat(yearly_data, ignore_index=True)

        return {
            "status": "success",
            "message": "Happiness data loaded successfully."
        }

    except Exception as e:
        return {
            "error": (
                f"Could not load the dataset: "
                f"{type(e).__name__}: {e}"
            )
        }
# Tool 1b : get_dataset_info

@tool
def get_dataset_info() -> dict:
    """Return the shape and column names of the loaded happiness dataset.

    Returns:
        A dictionary containing the number of rows, number of columns,
        dataset shape, and column names.
    """
    if df is None:
        return {
            "error": "No happiness data is loaded. Run load_happiness_data first."
        }

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }

# Tool 2: summarize_column

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics for the column,
        or an error dictionary if the data is not loaded or the column
        does not exist.
    """
    if df is None:
        return {
            "error": "No happiness data is loaded. Run load_happiness_data first."
        }

    if column not in df.columns:
        return {
            "error": (
                f"Column '{column}' was not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        }

    return df[column].describe().to_dict()

# Tool 3 — compute_correlation

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient, and p-value, or an error dictionary if the input
        is invalid.
    """
    if df is None:
        return {
            "error": "No happiness data is loaded. Run load_happiness_data first."
        }

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' was not found."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' was not found."}

    try:
        # Remove rows where either column has a missing value
        valid_data = df[[col1, col2]].dropna()

        if len(valid_data) < 2:
            return {
                "error": "Not enough valid data to compute correlation."
            }

        correlation, p_value = pearsonr(
            valid_data[col1],
            valid_data[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(correlation, 4),
            "p_value": round(p_value, 4),
        }
    except Exception as e:
        return {
            "error": (
                f"Could not compute correlation: "
                f"{type(e).__name__}: {e}"
            )
        }
    
# Tool 4 — get_top_n_countries

@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The numeric column used to rank the countries.
        year: The year to filter the dataset.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dictionary containing the year, ranking column, and a list
        of countries with their corresponding values, or an error
        dictionary if the input is invalid.
    """
    if df is None:
        return {
            "error": "No happiness data is loaded. Run load_happiness_data first."
        }

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found."
        }

    if "year" not in df.columns:
        return {
            "error": "The dataset does not contain a 'year' column."
        }
    if "country" not in df.columns:
        return {
            "error": "The dataset does not contain a 'country' column."
        }

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {
            "error": f"No data was found for the year {year}."
        }

    try:
        top_rows = (
            year_data
            .sort_values(column, ascending=False)
            .head(n)
        )

        results = top_rows[["country", column]].to_dict(
            orient="records"
        )

        return {
            "year": year,
            "column": column,
            "results": results,
        }

    except Exception as e:
        return {
            "error": (
                f"Could not rank countries: "
                f"{type(e).__name__}: {e}"
            )
        }
    
# Task 2 — Create the CodeAgent

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, getting dataset information,
summarizing columns, computing correlations, and ranking countries.

The load_happiness_data tool loads the dataset into memory and returns
a status message. It does not return the DataFrame.

Use get_dataset_info to obtain the dataset shape and column names.

Use summarize_column for descriptive statistics.

Use compute_correlation for Pearson correlation and statistical
significance. The tool automatically handles missing values.

Write Python code directly when the tools are not sufficient,
for example when creating custom plots or computing something
the tools do not cover.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        get_dataset_info,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats",
    ],
    max_steps=8,
)

# Task 3 — Five required queries

if __name__ == "__main__":

    queries = [
        "Load the happiness data and tell me its shape and column names.",

        "Summarize the happiness_score column.",

        "What is the correlation between gdp_per_capita and happiness_score? "
        "Is it statistically significant?",

        "Show me the top 5 happiest countries in 2020.",

       "Load the happiness data. Create a line chart showing average "
        "happiness_score by year, with one line for each regional_indicator. "
        "Use pandas to read ../assignments_01/outputs/merged_happiness.csv "
        "and save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")

        response = agent.run(
            query,
            reset=True
        )

        print(response)

    # My query 1
    my_query_1 = (
        "What are the average and standard deviation of happiness_score?"
    )

    response_1 = agent.run(
        my_query_1,
        reset=False
    )

    print("\n--- My Query 1 ---")
    print(response_1)

    # Comment:
    # This triggered the summarize_column tool.


    # My query 2
    my_query_2 = (
        "Load the happiness data from "
        "../assignments_01/outputs/merged_happiness.csv. "
        "Create a histogram of the actual happiness_score values and "
        "save it as outputs/happiness_distribution.png."
    )

    # Comment:
    # This query requires code generation because there is no histogram tool.

    response_2 = agent.run(
        my_query_2,
        reset=False
    )

    print("\n--- My Query 2 ---")
    print(response_2)

    # Comment:
    # This triggered code generation because the available tools do not
    # include a histogram tool.

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#    The agent reported a Pearson correlation of 0.6218 and a p-value of 0.0. It
#    determined that the correlation was statistically significant because the p-value
#    was less than 0.05. Therefore, it correctly applied the common significance
#    threshold of 0.05 and concluded that there is a statistically significant
#    positive correlation between gdp_per_capita and happiness_score.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    I was surprised by how well the agent was able to recover from an error. When
#    answering the question about the top 5 happiest countries in 2020, it first used
#    the incorrect column name "Happiness Score" and received an error. It then
#    inspected the available columns, recognized that the correct column was
#    "happiness_score", and successfully completed the query. This showed me that
#    the agent can use tools, recognize errors, and adjust its approach.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    An additional data-filtering tool would make this agent more useful. It could
#    filter the dataset by specific countries, regions, years, or ranges of values.
#    This would allow the agent to answer questions such as, "What was the average
#    happiness score for European countries between 2018 and 2021?" or "Which
#    countries had a GDP per capita above 8 and a happiness score above 7 in 2021?"