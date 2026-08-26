from pathlib import Path
import os

import pandas as pd
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

# ----------------   Task 1 — Define the four tools -------------

# Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Returns:
        A dictionary containing the dataset shape and column names.
        Returns an error dictionary if the data cannot be loaded.
    """
    global df

    path = Path(DATA_PATH)

    if not path.exists():
        return {
            "error": f"Could not find the merged dataset at {DATA_PATH}."
        }

    try:
        df = pd.read_csv(path)

        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
        }

    except Exception as e:
        return {
            "error": f"Could not load the dataset: {type(e).__name__}: {e}"
        }
"""   
# TEST TASK 1
print("\n--- Testing load_happiness_data ---")

result = load_happiness_data()

print(result)

"""
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
        correlation, p_value = pearsonr(df[col1], df[col2])

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

Use the available tools for loading data, summarizing columns,
computing correlations, and ranking countries.

Write Python code directly only when the tools are not sufficient,
for example when creating custom plots or computing something the
tools do not cover.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
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

        "Plot happiness_score over the years as a line chart, "
        "with one line per region. "
        "Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")

        response = agent.run(
            query,
            reset=False
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
    # This should primarily trigger the summarize_column tool.


    # My query 2
    my_query_2 = (
        "Create a histogram of happiness_score and save it as "
        "outputs/happiness_distribution.png."
    )

    response_2 = agent.run(
        my_query_2,
        reset=False
    )

    print("\n--- My Query 2 ---")
    print(response_2)

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was
#    statistically significant? Did it use the p-value correctly? What threshold
#    did it apply?
#
#    The agent attempted to determine statistical significance using the p-value.
#    However, in my run the correlation and p-value were returned as NaN because
#    of missing values in the data. Therefore, the agent could not make a valid
#    statistical significance determination. A common significance threshold is
#    p < 0.05.
#
# 2. Did any of the agent's responses surprise you — either by being more capable
#    than you expected, or less? Describe one specific example.
#
#    I was surprised that the CodeAgent was able to generate its own matplotlib
#    code for creating a histogram without having a specific histogram tool.
#    However, the generated code did not execute successfully because the agent
#    had difficulty accessing the happiness_score data correctly and encountered
#    a Matplotlib backend error. This showed me that an agent can generate code
#    for a task but the generated code may still need to be adjusted for the
#    structure of the data and the execution environment.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent
#    answer. (You do not need to implement it.)
#
#    An additional filtering tool would make the agent more useful. It could allow
#    the agent to filter the dataset by year, region, or other conditions before
#    performing an analysis. This would help answer questions such as "What is the
#    average happiness score for European countries in 2020?"
