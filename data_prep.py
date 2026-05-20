import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

def init_environment():
    """Load environment variables and authenticate with Hugging Face."""
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_API_KEY")
    if hf_token:
        print("Logging into Hugging Face...")
        login(token=hf_token)
    else:
        print("WARNING: HUGGING_FACE_API_KEY not found in .env")

def clean_text(text: str) -> str:
    """Basic text cleaning: remove zero-width spaces, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b\u200e\u200f\ufeff\xa0]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_claims(path: str, is_labelled: bool = True) -> pd.DataFrame:
    """Load and clean claims dataset."""
    print(f"Loading claims from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for cid, info in raw.items():
        row = {
            "claim_id": cid,
            "claim_text": clean_text(info.get("claim_text", ""))
        }
        if is_labelled:
            row["label"] = info.get("claim_label")
            row["evid_ids"] = info.get("evidences", [])
        rows.append(row)

    df = pd.DataFrame(rows)
    if is_labelled and "label" in df.columns:
        df["label"] = df["label"].astype("category")
    return df

def load_evidence(path: str):
    """Load and clean evidence dataset."""
    print(f"Loading evidence from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    evid_dict = {}
    rows = []
    for evid_id, text in raw.items():
        cleaned_text = clean_text(text)
        evid_dict[evid_id] = cleaned_text
        rows.append({"evid_id": evid_id, "evid_text": cleaned_text})
        
    df = pd.DataFrame(rows)
    return df, evid_dict

if __name__ == "__main__":
    init_environment()
    DATA_DIR = "./data"
    
    df_train = load_claims(os.path.join(DATA_DIR, "train-claims.json"), is_labelled=True)
    df_dev = load_claims(os.path.join(DATA_DIR, "dev-claims.json"), is_labelled=True)
    df_test = load_claims(os.path.join(DATA_DIR, "test-claims-unlabelled.json"), is_labelled=False)
    df_evid, evid_dict = load_evidence(os.path.join(DATA_DIR, "evidence.json"))
    
    print(f"Train size: {len(df_train)}")
    print(f"Dev size: {len(df_dev)}")
    print(f"Test size: {len(df_test)}")
    print(f"Evidence size: {len(df_evid)}")
    print("Data preparation test complete!")
