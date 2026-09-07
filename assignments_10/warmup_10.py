# ===============  ML vs. LLM in the Pipeline =========================

# -------------------- Q1 -----------------------

#The ML classifier produces a binary prediction, such as "good" or "skip".
#It is designed to analyze structured input features and make a consistent
#classification based on patterns it learned from training data.

#The LLM produces the recommendation or explanation. It can understand and
#generate natural language, so it is better suited for turning the item's
#information and the ML result into a useful, human-readable recommendation.

#Each tool has a different purpose. The ML model is good at making a specific,
#repeatable prediction from structured data, while the LLM is good at
#understanding context and generating natural-language responses.

#If we used the LLM to make the binary good/skip prediction, the result could
#be less consistent and harder to reproduce because LLM responses can vary
#and are not specifically trained for this classification task. It could also
#ignore the patterns learned from the labeled training data.

#If we used the ML model to write the recommendation, it would not be able
#to generate a meaningful natural-language explanation because a classifier
#is designed to output a prediction, not create text.

#Therefore, the ML classifier is used for the decision, and the LLM is used
#to communicate that decision as a recommendation.

# ------------------  Q2 ------------------------

#1. Converting a date string like "2023-07-04" to day-of-week:
#   Use deterministic code because the result follows a fixed, known rule and does not require learning from data.

#2. Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text:
#   Use an LLM because it can understand and interpret the context and language in unstructured job descriptions.

#3. Predicting customer churn given 15 numeric features and a labeled training dataset:
#   Use a trained ML model because it can learn patterns from the labeled historical data and make predictions on new customers.

#4. Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form:
#   Use deterministic code because the known variations can be mapped directly to a standard city name.

#5. Summing a column of revenue figures:
#   Use deterministic code because addition is a fixed calculation that does not require an ML model or LLM.

# --------------------  Q3 ---------------------------

##records instead of re-processing all existing records each time the script runs.

#It is important because it makes the pipeline more efficient by reducing
#processing time and cost, especially when the dataset grows.

#If the transform script re-processed all 365 records every time it ran, it
#would waste computing resources and, if the pipeline includes LLM calls,
#could significantly increase API usage and cost. It could also cause data
#correctness problems, such as creating duplicate records or repeatedly
#overwriting existing transformed results 

#=================================  Prompt Design ===================================

# -------------------- Q1 -----------------------

#"Write exactly two sentences about whether the weather is good for running.
#The first sentence must clearly state the prediction (good for running or
#skip). The second sentence must briefly explain the reasoning using the
#weather conditions. Be direct and practical. Do not use bullet points or
#headers."

#Validation change:

#The validation logic would need to check that the LLM response contains
#exactly two sentences instead of exactly one. It should also verify that the
#first sentence states the prediction and the second sentence provides the
#reasoning. If the response does not contain exactly two sentences, the
#pipeline should treat it as invalid and handle the error appropriately.

# ------------------------  Q3 -----------------------------------

import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    return None

#In a production pipeline, I would use this function around LLM API calls
#because temporary problems such as network errors, timeouts, or service
#unavailability can cause an individual request to fail. Retrying gives the
#request a chance to succeed without stopping the entire pipeline. If all
#retry attempts fail, returning None allows the pipeline to handle the failed
#record appropriately.









