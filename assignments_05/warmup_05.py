from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()
client = OpenAI()

"""
# check if the api key is laoded
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key is not None)
"""
"""
#load_dotenv()
#client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

# Print the response text
print("Response:")
print(response.choices[0].message.content)

# Print the model name
print("\nModel:")
print(response.model)

# Print the total tokens used
print("\nTotal Tokens:")
print(response.usage.total_tokens)

# --------- Q2 --------------

#load_dotenv()
#client = OpenAI()

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temp,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)


# Observation:
# At temperature 0, the response is simple and consistent.
# At temperature 0.7, the response is still relevant but a bit more creative.
# At temperature 1.5, the response is much more creative and includes an explanation
# in addition to the suggested name.
#
# If I needed a consistent, reproducible output, I would use temperature = 0
# because it produces the most predictable and repeatable responses.

# ---------- Q3 ------------

#load_dotenv()

#client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

# Print all three completions
for i, choice in enumerate(response.choices, start=1):
    print(f"Response {i}:")
    print(choice.message.content)
    print()

# --------- Q4 -------------

from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv()

#client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("Response:")
print(response.choices[0].message.content)

# The response was truncated because max_tokens was set to 15.
# Using max_tokens is useful for controlling response length,
# reducing API costs, improving response time, and preventing
# unnecessarily long outputs.

# -------------   System Messages and Personas   --------------

# ------------ Q1 -------------

# First system message: Patient Python tutor
messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Response from the Python Tutor:")
print(response.choices[0].message.content)

# Second system message: cowboy
messages = [
    {
        "role": "system",
        "content": "You are a cowboy who explains everything using his style."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }   
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nResponse from the cowboy:")
print(response.choices[0].message.content)

# Observation:
# The system message changed the model's personality and style.
# The first response was patient, simple, and encouraging.
# The second response explained the same concept using cowboy-themed language.
# The underlying information stayed similar, but the tone and wording changed.

# --------- Q2 -----------

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Response:")
print(response.choices[0].message.content)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Response:")
print(response.choices[0].message.content)

# Observation:
# The model knows Jordan's name because the entire conversation
# history was included in the messages list. Although the API is
# stateless and does not remember previous API calls, it can use
# any context that is provided in the current request.

"""

# ------------- Prompt Engineering ------------

# --------------Q1----------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as Positive, Negative, or Mixed.

Return your answers in exactly this format:
Review 1: <sentiment>
Review 2: <sentiment>
Review 3: <sentiment>

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# ----------- Q2 ----------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as Positive, Negative, or Mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: Mixed

Now classify these reviews:

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"

Return your answers in this format:
Review 1: <sentiment>
Review 2: <sentiment>
Review 3: <sentiment>
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# Observation:
# Adding one example improved the consistency of the output format.
# The model followed the example and returned the results in the
# requested structure, making the output more predictable than in Q1.

# ------------ Q3 -------------
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as Positive, Negative, or Mixed.

Examples:

Review: "The customer service was friendly and solved my problem quickly."
Sentiment: Positive

Review: "The app freezes every time I try to save my work."
Sentiment: Negative

Review: "The product is affordable, but the instructions are confusing."
Sentiment: Mixed

Now classify these reviews:

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"

Return your answers in this format:
Review 1: <sentiment>
Review 2: <sentiment>
Review 3: <sentiment>
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)


# Observation:
# Zero-shot works well for simple tasks when the instructions are clear.
# One-shot provides one example, helping the model understand the
# desired format and improving consistency.
# Few-shot gives multiple examples, making the model even more reliable,
# especially for tasks with different categories or more complex patterns.
#
# I would use:
# - Zero-shot for straightforward tasks.
# - One-shot when I want a specific output format.
# - Few-shot when the task is more complex or when I need the most
#   consistent and accurate results.

# ------------- Q4 -----------------
prompt = """
Solve the following problem. Show your reasoning step by step before giving the final answer.
Label the final answer clearly.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# Observation:
# Asking the model to reason step by step can improve accuracy because
# it encourages the model to break the problem into smaller calculations
# instead of trying to produce the answer in one step. This reduces the
# chance of missing a step or making an arithmetic mistake.

# ---------------- Q5 --------------------------

review = """I've been using this tool for three months. It handles large datasets well,
but the UI is clunky and the export options are limited."""

prompt = f"""
Analyze the following review.

Return ONLY valid JSON.
Do not include any explanation or markdown.

The JSON must have exactly these keys:
- sentiment
- confidence
- reason

The confidence value must be a float between 0 and 1.

Review:
"{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Get the raw response text
raw_response = response.choices[0].message.content

print("Raw Response:")
print(raw_response)

# Parse the JSON safely
try:
    result = json.loads(raw_response)

    print("\nParsed Results:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nError: Response was not valid JSON.")
    print("Raw response:")
    print(raw_response)

# ------------ Q6 -----------------

# First example: Instructions
user_text = """First boil a pot of water. Once boiling, add a handful of salt and the
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."""

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("First Response:")
print(response.choices[0].message.content)

# Second example: Regular prose
user_text = """The weather was beautiful all weekend, and we enjoyed spending time
walking through the park and visiting local shops."""

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nSecond Response:")
print(response.choices[0].message.content)

# Observation:
# Delimiters help clearly separate the instructions from the user's text.
# This prevents the model from confusing the prompt with the content it
# is supposed to analyze, resulting in more accurate and reliable responses.

# ---------- Local Models with Ollama -------------

# ------------- Q1 -------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("OpenAI Response:")
print(response.choices[0].message.content)


"""
Ollama Output:

Paste your actual Ollama response here.

Example:
A large language model is an AI model capable of understanding and generating human language, trained on vast datasets to learn patterns and phrases. It processes and 
generates text by analyzing context and understanding the underlying language structure.  

The model's training data allows it to learn from a wide range of linguistic information, enabling it to perform tasks like translation, summarization, or creative 
writing with accuracy and fluency.
"""

# Observation:
# Both Ollama and OpenAI provided accurate explanations of what a large
# language model is. The Ollama response focused more on language patterns
# and included examples of tasks like creative writing, while the OpenAI
# response emphasized AI systems, machine learning, and deep learning.
# The OpenAI response was more concise and directly followed the two-sentence
# instruction, while the Ollama response provided more detail.
#
# One advantage of running a model locally is that user data stays on the
# local machine, improving privacy and reducing dependence on external APIs.
#
# One disadvantage is that local models may require significant computer
# resources and may not match the performance or accuracy of larger
# cloud-based models.