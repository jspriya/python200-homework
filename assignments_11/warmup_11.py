# --------------------------------------------------
# Prefect Question 1
# --------------------------------------------------

# A @task represents a single, focused unit of work in a Prefect pipeline.
# A @flow is the orchestrator that calls tasks in order and manages the
# overall flow run.

# For a helper function that converts a temperature from Celsius to
# Fahrenheit, I would not decorate it with @task because it is a simple,
# pure, in-memory calculation with no I/O. It does not need Prefect's
# orchestration features such as retries or task-level monitoring.

# I would keep this helper as a regular Python function and use @task for
# larger pipeline steps that need to be tracked, retried, logged, or
# interact with external systems.

# --------------------------------------------------
# Prefect Question 2
# --------------------------------------------------
from prefect import task
@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass

# --------------------------------------------------
# Prefect Question 3
# --------------------------------------------------

# If a task fails, I would open the failed flow run in the Prefect UI and select the failed task to view its 
# task-run details. I would look at the Logs section for the exception message and error output, including the 
# stack trace when available. This would show me what caused the task to fail and help me determine whether the 
# problem was with the API request, data processing, or another part of the task.

# **********************  Production Patterns ********************

# --------------------------------------------------
# Production Question 1
# --------------------------------------------------


# raise_for_status() checks whether the HTTP response was successful. If the
# API returns an error such as a 500 status code, it raises an exception.

# This is better than using:
# if response.status_code != 200:
#     print("error")

# because simply printing an error does not stop the pipeline. The pipeline
# could continue with empty, incomplete, or invalid data.

# With raise_for_status(), the exception is propagated to Prefect. Prefect
# marks the task as Failed, the error appears clearly in the logs, and
# downstream tasks do not run. This prevents bad data from moving through
# the pipeline.

# With the print("error") approach, the error is only printed, so the task
# can continue running and downstream tasks may also run with bad or
# missing data.

# Therefore, in a data pipeline, a visible task failure is safer than
# silently allowing incorrect data to continue through the pipeline.

# --------------------------------------------------
# Production Question 2
# --------------------------------------------------


# Using upsert with on_conflict="date" protects the pipeline from creating
# duplicate records when the pipeline is re-run. If the same date already
# exists in the weather_raw table, upsert updates the existing row instead
# of trying to insert a duplicate.

# In this scenario, after the pipeline crashes during the transform step,
# I can fix the bug and safely re-run the pipeline from the beginning.
# The existing raw weather records will be updated rather than duplicated.

# If I had used plain insert instead, re-running the pipeline would try to
# insert the same dates again. Because date is the primary key, the insert
# would cause a conflict/error for records that already exist, making the
# re-run fail instead of safely updating the existing records.

# Therefore, upsert with on_conflict supports idempotency and makes it safe
# to re-run the pipeline after a failure.

# --------------------------------------------------
# Production Question 3
# --------------------------------------------------

from prefect import task
from prefect.logging import get_run_logger

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(
        f"Upserted {len(enrichment_records)} enrichment records into weather_enriched."
    )

# --------------------------------------------------
# Production Question 4
# --------------------------------------------------

# The incremental processing check contributes to idempotency by checking
# which records have already been processed before running the ML and LLM
# steps. It skips records that already exist in the weather_enriched table,
# so re-running the pipeline does not process the same records again.

# If the incremental check were removed, the ML and LLM steps would run on
# all 365 records every time the pipeline ran. This would increase costs
# because the LLM API would be called repeatedly for records that were
# already processed. It would also take more time because every record
# would need to go through the ML and LLM processing again.

# Re-processing could also affect data correctness because existing
# enrichment results could be overwritten with different results, rather
# than preserving the results that were already generated. The incremental
# check therefore makes the pipeline safer to re-run and avoids unnecessary
# API calls, processing time, and cost.

