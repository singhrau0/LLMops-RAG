import json
import re
from pathlib import Path
import yaml

RAW_FILE = Path("data/raw/space_kb.md")
CLEAN_FILE = Path("data/clean/space_kb_clean.md")
CHUNKS_FILE = Path("artifacts/chunks.json")

# clean data 
def read_text(path):
    return path.read_text(encoding ="utf-8")

def clean_spaces(text):
    text = text.replace("\r\n","\n")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3}","\n\n",text)
    return text.strip()+"\n"

def write_text(path,text):
    path.parent.mkdir(parents = True,exist_ok = True)
    path.write_text(text,encoding = "utf-8")

# preprocess data
def load_params():
    return yaml.safe_load(PARAMS_FILE.read_)