from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
import os
import string

"""
# checkin if API key loads successfully
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
"""

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print("Key exists:", api_key is not None)
print("Key length:", len(api_key))

"""
print("First 10 characters:", api_key[:10])
print("Last 4 characters:", api_key[-4:])
print("Has leading/trailing spaces:", api_key != api_key.strip())
"""

# ----------- Concepts Question 1 -------------

# Scenario A — RAG (Retrieval-Augmented Generation)
# A legal team should use RAG because the assistant needs to answer questions
# from hundreds of internal PDFs that are updated regularly. RAG allows the
# system to retrieve the most relevant, up-to-date information from the
# policy library without retraining the model each time the documents change.

# Scenario B — Fine-tuning
# Fine-tuning is the best approach because the startup has 3,000 examples of
# their desired writing style. Training the model on these examples can help
# it consistently reproduce their specific dry, minimalist brand voice.

# Scenario C — Prompt Engineering
# Prompt engineering is the best approach because the analyst only needs to
# ask questions about one short, two-page report. She can provide the report
# as context in the prompt without needing to build a retrieval system or
# fine-tune the model.

# ----------- Concepts Question 2 -------------

# A confidently wrong answer can be more harmful than an answer that says
# "I am not sure" because people may trust the confident answer and act on
# incorrect information without checking it. An answer that shows uncertainty
# encourages the user to verify the information before taking action.

# For example, if someone asks an AI for medical advice and it confidently
# gives the wrong medication dosage, the person could follow the advice and
# experience serious health consequences.

# The tone of the response affects how much we trust it. A confident,
# authoritative tone can make incorrect information sound reliable, even when
# there is no evidence behind it. When an AI communicates uncertainty, users
# are more likely to recognize that the answer may need to be verified.


# ----------- Concepts Question 3 -----------------

# 1. Extract text from source documents
#    - Extract readable text from PDFs, documents, or other source files.
#
# 2. Split text into chunks
#    - Break the extracted text into smaller pieces so they can be searched efficiently.
#
# 3. Convert text chunks into embeddings
#    - Convert each text chunk into a numerical vector that represents its meaning.
#
# 4. Receive the user's query
#    - The system receives the question or request from the user.
#
# 5. Embed the user's query
#    - Convert the user's question into an embedding so it can be compared with the document chunks.
#
# 6. Retrieve the most relevant chunks
#    - Search for the document chunks whose embeddings are most similar to the user's query.
#
# 7. Inject retrieved chunks into the prompt
#    - Add the relevant retrieved information to the prompt that will be sent to the LLM.
#
# 8. Generate a response from the LLM
#    - The LLM uses the user's question and the retrieved information to generate an answer.
#
# -------------- Keyword RAG -------------

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# ----------- Keyword Question 1 --------------

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {results[0][0]}")

# The selected document is loyalty.txt. This happened because the simple
# keyword retrieval method only counts exact word matches. The word "your"
# appears in both the query and loyalty.txt, and because documents with equal
# scores are sorted in reverse alphabetical order, loyalty.txt was selected.
# This shows a limitation of simple keyword retrieval because it may select
# a less relevant document when keyword scores are tied.

# ---------- Keyword RAG 2 -----------

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {results[0][0]}")

# No document was selected because there were no exact keyword matches between
# the query and the documents. Keyword RAG did not handle this query well
# because the menu may contain relevant information, but it does not use the
# same words as the query, such as "caffeine" or "without caffeine."
# Semantic retrieval using embeddings would work better because it can
# understand the meaning and relationship between words rather than relying
# only on exact keyword matches.

# ------------ Keyword RAG 3 -------------

# Prediction:
# I predict that loyalty.txt will be selected because the query contains
# "sign up" and "rewards," which are related to joining a loyalty program
# and earning points.

query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {results[0][0]}")

# Result:
# My prediction was not correct. No document was selected because the
# keyword retrieval system looks only for exact word matches. Although
# loyalty.txt is semantically related to the query, it contains words like
# "join," "loyalty," and "points" instead of the exact words "sign up" and
# "rewards." Therefore, there was no keyword overlap.

# ------------- Semantic RAG Q1 ----------------

# 1. A vector embedding  represents data like words, sentences, or images in a mathematical space as numbers. 
# Texts with similar meanings usually have embeddings that are closer together.


# 2. The chunk with a cosine similarity score of 0.85 is more relevant than
# the one with a score of 0.30. A higher cosine similarity score means the
# meaning of that chunk is more closely related to the meaning of the query.

# 3. 3. Semantic search can find relevant chunks without exact word matches
# because it compares the meaning of the query and the text rather than just
# looking for identical words. 

# ----------- Semantic Q2 -------------------

#Semantic Question 2
#Keyword RAG and semantic RAG handle the same problem differently. Copy this table into your code as a comment and fill in the right column:

#| Feature                    | Keyword RAG                       | Semantic RAG                |
#|----------------------------|-----------------------------------|-----------------------------|
#| What is compared?          | Exact word overlap                | Meaning/semantic similarity |
#| What is retrieved?         | Full document                     | Most relevant text chunk(s) |
#| Can it handle synonyms?    | No                                | Yes                         |
#| Storage format             | Plain text dictionary             | Vector embeddings           |
#| Relevance score            | Number of overlapping keywords    | Cosine similarity score     |

# -------------- LlamaIndex -------------------

print("--------------- Q1 LlamaIndex -------------------")

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

load_dotenv()

# Adjust this path based on your local folder structure
PDF_PATH = "./brightleaf_pdfs"

# Check that the folder exists before trying to load it
import os

assert os.path.exists(PDF_PATH), f"Directory not found: {PDF_PATH}"

# Load the Brightleaf PDFs
documents = SimpleDirectoryReader(PDF_PATH).load_data()

# Build the vector index
index = VectorStoreIndex.from_documents(documents)

# Create a query engine that retrieves the top 3 chunks
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    # Send the question to the RAG pipeline
    response = query_engine.query(question)

    # Print the model's answer
    print("\nAnswer:")
    print(response)

    # Print information about the 3 retrieved source nodes
    print("\nRetrieved source nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        score = node.score
        text = node.text[:150]

        print(f"\nSource {i}")
        print(f"Similarity score: {score}")
        print(f"Chunk text: {text}")

# Query 1 observations:
# - The first retrieved chunk is highly relevant because it discusses
#   BrightLeaf's employee benefits program. The other two retrieved chunks,
#   about security and the company mission, are less relevant to this question.
# - The model's response is confident and specific. It directly lists the
#   benefits offered and does not hedge with phrases such as "based on the
#   context" or "I'm not sure."
# - An unexpected result was that chunks from the security policy and mission
#   statement were retrieved even though the question was specifically about
#   employee benefits.

# Query 2 observations:
# - The first retrieved chunk is highly relevant because it directly discusses
#   BrightLeaf's network and data security policies. The other retrieved chunks,
#   about employee benefits and the company mission, are less relevant.
# - The model's response is confident, detailed, and specific. It provides
#   concrete information about authentication, encryption, access control,
#   incident response, training, and compliance without noticeable hedging.
# - An unexpected result was that chunks from the employee benefits and mission
#   documents were included in the top three retrieved results.

# -------------- LlamaIndex Question 2 -------------------

print("--------------- Q2 LlamaIndex -------------------")

question = "What employee benefits does BrightLeaf offer?"

# Run with similarity_top_k = 1
print("\n" + "=" * 70)
print("similarity_top_k = 1")
print("=" * 70)

query_engine_1 = index.as_query_engine(similarity_top_k=1)

response_1 = query_engine_1.query(question)

print("\nQuestion:", question)
print("\nResponse:")
print(response_1)

print("\nSource node scores:")

for i, source_node in enumerate(response_1.source_nodes, start=1):
    print(f"Source {i} score:", source_node.score)


# Run with similarity_top_k = 5
print("\n" + "=" * 70)
print("similarity_top_k = 5")
print("=" * 70)

query_engine_5 = index.as_query_engine(similarity_top_k=5)

response_5 = query_engine_5.query(question)

print("\nQuestion:", question)
print("\nResponse:")
print(response_5)

print("\nSource node scores:")

for i, source_node in enumerate(response_5.source_nodes, start=1):
    print(f"Source {i} score:", source_node.score)

# Observations:
# With similarity_top_k=1, the model used only the most relevant source node
# and gave a shorter, more general summary of BrightLeaf's employee benefits.
#
# With similarity_top_k=5, the model had access to five retrieved chunks and
# produced a much more detailed response. It included specific information
# such as preventive care, telemedicine, mental health services, fitness
# memberships, the Wellness Reimbursement Plan, life and disability insurance,
# a 401(k) company match, parental leave, professional development, and the
# Learning Hub.
#
# More retrieved context is not always better. In this example, the response
# became more detailed with similarity_top_k=5, but the additional source nodes
# had lower similarity scores than the first node and may contain less relevant
# information. Too much unrelated context could potentially distract the model
# or lead to a less focused answer.

# -------------- LlamaIndex Question 3 -------------------

question = "What is BrightLeaf's policy on remote work for employees?"

print("\n" + "=" * 70)
print("LlamaIndex Question 3")
print("=" * 70)

print("\nQuestion:", question)

response = query_engine.query(question)

print("\nResponse:")
print(response)

print("\nRetrieved chunks:")

for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print("Similarity score:", source_node.score)
    print("Chunk text:")
    print(source_node.node.get_content())

# Observations:
# I expected the pipeline to struggle with this question because I was asking
# specifically about BrightLeaf's remote work policy, while I was not sure that
# the documents contained a formal remote work policy.
#
# However, the pipeline found relevant information in the employee benefits
# document. The first retrieved chunk discussed flexible scheduling and hybrid
# work options, so the model was able to provide a clear and specific answer.
#
# Something unexpected was that the security policy and mission statement were
# also retrieved, even though they were not directly related to remote work.
#
# To handle more difficult or vague queries better, I would use a relevance
# threshold or reranking step to reduce unrelated retrieved chunks. I would
# also instruct the model to clearly state when the documents do not contain
# enough information to answer a question instead of making assumptions.

# -------------- LlamaIndex Question 4 -------------------

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms.openai import OpenAI

# Create the judge LLM
judge_llm = OpenAI(model="gpt-4o-mini")

# Create evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


# ---------- Query 1: Expected to be high quality ----------

q1 = "What employee benefits does BrightLeaf offer?"

response1 = query_engine.query(q1)

print("\n" + "=" * 70)
print("Question 1:", q1)
print("=" * 70)

print("\nResponse:")
print(response1)

faithfulness_result1 = faithfulness_evaluator.evaluate_response(
    response=response1
)
relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1
)

print("\nFaithfulness score:", faithfulness_result1.score)
print("Relevancy score:", relevancy_result1.score)


# ---------- Query 2: Expected to be lower quality ----------

q2 = "What is BrightLeaf's policy on providing employees with free company cars?"

response2 = query_engine.query(q2)

print("\n" + "=" * 70)
print("Question 2:", q2)
print("=" * 70)

print("\nResponse:")
print(response2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\nFaithfulness score:", faithfulness_result2.score)
print("Relevancy score:", relevancy_result2.score)

# ---------------- Observations ----------------
#
# A faithfulness score of 1.0 means that the response is fully supported by
# the retrieved context and does not contain unsupported or hallucinated
# information. A score of 0.0 would indicate that the response includes
# information that cannot be supported by the retrieved context.
#
# A relevancy score measures whether the response appropriately addresses the
# user's question. This is different from faithfulness because a response can
# be supported by the retrieved context but still fail to answer the user's
# question directly.
#
# The scores did not change between the two queries. Both queries received
# faithfulness and relevancy scores of 1.0. The first query was answered using
# information directly available in the employee benefits document. For the
# second query, even though information about free company cars was not in the
# documents, the model correctly stated that the policy did not mention them
# instead of inventing information. Therefore, the response was still both
# faithful to the context and relevant to the question.
#
# LLM-as-a-judge is an approach where a language model evaluates another
# model's response using criteria such as faithfulness and relevancy. It is
# useful for RAG evaluation because these qualities involve understanding the
# meaning and relationship between the question, retrieved context, and
# response. A simple accuracy metric or exact-match comparison cannot easily
# evaluate these semantic qualities.
