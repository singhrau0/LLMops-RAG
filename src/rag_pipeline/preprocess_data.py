import re
from src.rag_pipeline.config import CHUNKS_FILE,CLEAN_FILE,load_params,write_json,read_text

def split_words(words,max_words,overlap_words):
    chunks = []
    start = 0
    while start<len(words):
        end = min(start+ max_words,len(words))
        chunks.append(words[start:end])
        if end==len(words):
            break
        start = end-overlap_words
    return chunks

def process_data():
    params = load_params()
    max_words = params["chunking"]["max_words"]
    overlap_words = params["chunking"]["overlap_words"]
    clean_text = read_text(CLEAN_FILE)
    sections = re.split(r"\n##",clean_text)
    rows = []
    for section in sections[1:]:
        doc_id,body = section.split("\n",1)
        words = body.split()
        for chunk_number,chunk_words in enumerate(split_words(words,max_words,overlap_words)):
            rows.append(
                {
                    "chunk_id":f"{doc_id.strip()}_chunk_{chunk_number}",
                    "doc_id":doc_id.strip(),
                    "text":" ".join(chunk_words)
                }
            )
    write_json(CHUNKS_FILE,rows)
    print(f"Created {len(rows)} chunks at {CHUNKS_FILE}")

if __name__=="__main__":
    process_data()