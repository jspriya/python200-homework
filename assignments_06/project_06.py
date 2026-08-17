# Mini-Project: Groundwork Coffee Co. Q&A Assistant
# ------------------------------------------------------------

from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# ------------------------------------------------------------
# Step 1: Setup
# ------------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded successfully.")
else:
    print("Warning: API key was not found.")

# Path to the Groundwork documents
docs_dir = Path("./groundwork_docs")

# Verify that the directory exists
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# ------------------------------------------------------------
# Step 2: Load the Documents
# ------------------------------------------------------------

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"\nNumber of documents loaded: {len(documents)}")

print("\nDocuments:")

for document in documents:
    print("-", document.metadata["file_name"])


# ------------------------------------------------------------
# Step 3: Build the Index and Query Engine
# ------------------------------------------------------------

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")

# ------------------------------------------------------------
# Step 4: Query the Assistant
# ------------------------------------------------------------

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:

    print("\n" + "=" * 70)
    print("Question:", question)
    print("=" * 70)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    # Get the top retrieved source node
    top_source = response.source_nodes[0]

    print("\nTop Retrieved Source:")
    print(
        "Document:",
        top_source.node.metadata.get("file_name")
    )
    print(
        "Similarity score:",
        top_source.score
    )
    print(
        "Chunk text:",
        top_source.node.get_content()[:200]
    )

# Reflection:
# Overall, the assistant sounded confident and accurate in its responses.
# The answers matched the information in the retrieved documents, and the
# top retrieved source was relevant to each question. The response about
# Groundwork's history had the highest similarity score and closely matched
# the information in the "our_story.txt" document.
#
# One surprising result was the dairy-free milk question. The assistant gave
# a specific and useful answer about oat milk, almond milk, and soy milk being
# available at no extra charge, but the top retrieved source was
# "seasonal_specials.txt" rather than the main menu or FAQ. This shows that
# even when the final answer is accurate, the retrieved document may not
# always be the source I would initially expect.
#
# Overall, the assistant appeared confident and the answers were supported by
# relevant information from the Groundwork documents.

# Step 5: Find a Failure
# ------------------------------------------------------------

failure_question = (
    "Does Groundwork Coffee offer free Wi-Fi to customers?"
)

print("\n" + "=" * 70)
print("Step 5: Find a Failure")
print("=" * 70)

print("\nQuestion:", failure_question)

failure_response = query_engine.query(failure_question)

print("\nFull Response:")
print(failure_response)

print("\nAll Retrieved Source Nodes:")

for i, source_node in enumerate(
    failure_response.source_nodes, start=1
):
    print(f"\nSource {i}")
    print(
        "Document:",
        source_node.node.metadata.get("file_name")
    )
    print(
        "Similarity score:",
        source_node.score
    )
    print(
        "Chunk text:",
        source_node.node.get_content()[:200]
    )

# ------------------------------------------------------------
# Step 5 Reflection
# ------------------------------------------------------------
#
# I asked, "Does Groundwork Coffee offer free Wi-Fi to customers?"
# I expected this question to be difficult because none of the Groundwork
# documents contain information about Wi-Fi.
#
# The retrieval system still returned three documents: "our_story.txt",
# "wholesale_catering.txt", and "menu.txt". These documents were not actually
# relevant to the question about Wi-Fi. This shows that semantic search will
# still retrieve the closest matching documents even when the requested
# information does not exist in the knowledge base.
#
# However, the model did not guess or invent an answer. Instead, it said that
# Groundwork Coffee does not mention anything about free Wi-Fi in the provided
# context information. The model's tone became more cautious and less
# confident than in the earlier questions where the answer was directly
# available in the documents.
#
# This suggests that AI-generated responses should not automatically be trusted
# just because they sound confident. A system can retrieve unrelated documents,
# and a different model or prompt might still generate a confident but incorrect
# answer.
#
# To improve the system, I could add a similarity score threshold so that
# documents with scores below a certain level are not used as evidence. I could
# also add instructions telling the model to clearly state when the answer is
# not available in the retrieved documents rather than guessing. Adding better
# metadata, reranking, or more relevant documents could also improve retrieval.

# ------------------------------------------------------------
# Step 6: Final Reflection
# ------------------------------------------------------------
#
# 1. In this project, the equivalent LlamaIndex implementation for loading
# documents, building a vector index, and creating a query engine only took a
# few lines of code:
#
#     documents = SimpleDirectoryReader(str(docs_dir)).load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     query_engine = index.as_query_engine(similarity_top_k=3)
#
# The manual semantic RAG implementation from the lesson required many more
# lines for chunking, creating embeddings, storing vectors, and retrieving
# relevant chunks. This shows the value of using a framework like LlamaIndex.
# It abstracts much of the underlying complexity and allows developers to build
# a working RAG system more quickly. However, understanding the underlying
# process is still important for debugging and improving the system.
#
# 2. A useful application of this approach would be an internal company
# knowledge assistant. For example, a company could use RAG with employee
# handbooks, HR policies, benefits information, technical documentation, and
# company procedures. Employees could ask questions in natural language instead
# of manually searching through many documents. The assistant could retrieve
# the most relevant information and provide a quick answer.
#
# 3. One failure mode that RAG cannot fully prevent is an incorrect or
# misleading answer generated by the language model even when the correct
# information was successfully retrieved. The model may misunderstand the
# retrieved context, combine information incorrectly, or add details that were
# not supported by the documents. This means that good retrieval alone does not
# guarantee a completely accurate answer.