import math
from src.rag_pipeline.config import ARTIFACT_INDEX,CHUNKS_FILE,read_json,write_json,tokenize

def make_vector(tokens,vocabulary):
    counts = {word: 0 for word in vocabulary}
    for token in tokens:
        if token in counts:
            counts[token]+=1
    length = math.sqrt(sum(count**2 for count in counts.values() )) or 1
    return [round(counts[word]/length,6) for word in vocabulary] 


def build_index():  # Define the indexing pipeline stage.
    chunks = read_json(CHUNKS_FILE)  # Load preprocessed chunks from JSON.
    all_tokens = [tokenize(chunk["text"]) for chunk in chunks]  # Tokenize every chunk text.
    vocabulary = sorted(set(token for tokens in all_tokens for token in tokens))  # Build one sorted word list.
    vectors = [make_vector(tokens, vocabulary) for tokens in all_tokens]  # Convert every chunk into a vector.
    index = {"vocabulary": vocabulary, "chunks": chunks, "vectors": vectors}  # Store all retrieval data together.
    write_json(ARTIFACT_INDEX, index)  # Save the index as a readable JSON artifact.
    print(f"Vector index saved to {ARTIFACT_INDEX}")  # Print a simple success message.

if __name__ == "__main__":  # Run only when this file is executed as a script.
    build_index()  # Start the indexing stage.
