import math  # Import math so retrieval can measure vector length.
import os  # Import os so we can check if OPENAI_API_KEY exists.

from dotenv import load_dotenv  # Import dotenv so scripts can read the .env file.
from langchain_core.output_parsers import StrOutputParser  # Import a parser that returns plain text from LangChain.
from langchain_core.prompts import ChatPromptTemplate  # Import LangChain prompt templates for chat models.
from langchain_openai import ChatOpenAI  # Import the OpenAI chat model wrapper from LangChain.

from src.rag_pipeline.config import ARTIFACT_INDEX, load_params, read_json, tokenize  # Import paths and helpers.

load_dotenv(".env")  # Load OPENAI_API_KEY from the real .env file for command-line and Flask runs.

def query_vector(question, vocabulary):  # Define a helper that converts the question into a vector.
    tokens = tokenize(question)  # Tokenize the student question.
    counts = {word: 0 for word in vocabulary}  # Start every vocabulary word with count zero.
    for token in tokens:  # Loop through every question token.
        if token in counts:  # Count only words that exist in the index vocabulary.
            counts[token] += 1  # Add one count for the matching word.
    length = math.sqrt(sum(count * count for count in counts.values())) or 1  # Calculate question vector length.
    return [counts[word] / length for word in vocabulary]  # Return the normalized question vector.

def dot_product(left, right):  # Define a helper that compares two vectors.
    return sum(a * b for a, b in zip(left, right))  # Return the cosine-style similarity score.

def retrieve(question):  # Define the retrieval part of RAG.
    params = load_params()  # Load settings from params.yaml.
    top_k = params["retrieval"]["top_k"]  # Read how many chunks we want to retrieve.
    index = read_json(ARTIFACT_INDEX)  # Load the JSON vector index artifact.
    q_vector = query_vector(question, index["vocabulary"])  # Convert the question into a vector.
    scored = []  # Create an empty list for scored chunks.
    for chunk, vector in zip(index["chunks"], index["vectors"]):  # Compare the question to every chunk.
        score = dot_product(q_vector, vector)  # Calculate similarity between question and chunk.
        scored.append({**chunk, "score": round(score, 4)})  # Store chunk data plus retrieval score.
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)  # Sort best chunks first.
    return [item for item in ranked[:top_k] if item["score"] > 0]  # Return useful top chunks only.

def make_chain():  # Define the LangChain answer generation chain.
    params = load_params()  # Load model settings from params.yaml.
    model_name = params["openai"]["model"]  # Read the OpenAI model name.
    model = ChatOpenAI(model=model_name, temperature=0)  # Create GPT-4o-mini chat model with stable output.
    prompt = ChatPromptTemplate.from_messages([  # Create a chat prompt with system and user messages.
        ("system", "Answer only from the context. If the context is missing, say you do not know."),  # Set grounding rule.
        ("user", "Question: {question}\n\nContext:\n{context}"),  # Insert the question and retrieved context.
    ])  # Finish the prompt template.
    return prompt | model | StrOutputParser()  # Connect prompt, model, and text parser into one chain.

def answer_question(question):  # Define the full RAG function used by Flask.
    sources = retrieve(question)  # First retrieve useful chunks from the JSON index.
    context = "\n\n".join(source["text"] for source in sources)  # Combine retrieved chunks into one context string.
    steps = ["Load JSON index", "Retrieve top chunks", "Send context to LangChain", "Ask gpt-4o-mini", "Show answer"]  # Save visible pipeline steps.
    if not os.getenv("OPENAI_API_KEY"):  # Check whether the API key is missing.
        answer = "OPENAI_API_KEY is missing. Add it to your .env file before asking GPT."  # Explain the missing key.
    elif not sources:  # Check whether retrieval found no useful context.
        answer = "I do not know from the knowledge base."  # Give a grounded answer when context is empty.
    else:  # Run GPT only when key and context both exist.
        answer = make_chain().invoke({"question": question, "context": context})  # Ask LangChain to call GPT-4o-mini.
    return {"answer": answer, "sources": sources, "steps": steps}  # Return everything the frontend needs.
