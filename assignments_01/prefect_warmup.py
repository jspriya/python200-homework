import numpy as np
import pandas as pd
from prefect import flow, task

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary

if __name__ == "__main__":
    pipeline_flow()

"""

1. Why might Prefect be more overhead than it is worth here?

This pipeline is very small and only processes a few values. It consists of
three simple functions that run quickly and in sequence. Adding Prefect
requires installing another package, decorating functions with @task and
@flow, and understanding the framework, which adds complexity without much
benefit for such a simple script.

2. When could Prefect still be useful?

Prefect is useful for larger or production data pipelines. For example, it can
schedule workflows to run automatically, retry failed tasks, monitor workflow
execution, log task results, manage dependencies between tasks, and process
large datasets from databases, APIs, or cloud storage. Even if each individual
task is simple, Prefect helps make complex, automated workflows more reliable
and easier to manage.

"""