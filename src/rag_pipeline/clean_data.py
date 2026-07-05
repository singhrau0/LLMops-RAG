from src.rag_pipeline.config import CLEAN_FILE,RAW_FILE,read_text,write_text,clean_spaces

def clean_data():
    raw_text = read_text(RAW_FILE)
    clean_text = clean_spaces(raw_text)
    write_text(CLEAN_FILE,clean_text)
    print(f"Cleaned Data Saved to {CLEAN_FILE}")

if __name__=="__main__":
    clean_data()