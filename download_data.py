import os
import urllib.request
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "indotoxic2024_annotated_data_v2_final.csv": "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/indotoxic2024_annotated_data_v2_final.csv",
    "indotoxic2024_annotated_data_v2_final.jsonl": "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/indotoxic2024_annotated_data_v2_final.jsonl",
    "indotoxic2024_annotator_demographic_data_v2_final.csv": "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/indotoxic2024_annotator_demographic_data_v2_final.csv",
    "indotoxic2024_annotator_demographic_data_v2_final.jsonl": "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/indotoxic2024_annotator_demographic_data_v2_final.jsonl",
    "IndoDiscourse_Toxicity_Related_Experiment_Code.ipynb": "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/IndoDiscourse%20-%20Toxicity%20Related%20Experiment%20Code.ipynb"
}

def download_file(filename, url):
    target_path = os.path.join(DATA_DIR, filename)
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"Downloaded {filename} ({size_mb:.2f} MB)")
    return target_path

def inspect_dataset():
    for name, url in FILES.items():
        download_file(name, url)

    print("\n--- Inspecting Downloaded Files ---")
    csv_annotated = os.path.join(DATA_DIR, "indotoxic2024_annotated_data_v2_final.csv")
    csv_demographic = os.path.join(DATA_DIR, "indotoxic2024_annotator_demographic_data_v2_final.csv")

    df_annotated = pd.read_csv(csv_annotated)
    print(f"\n[Annotated Data] Shape: {df_annotated.shape}")
    print("Columns:", list(df_annotated.columns))
    print("Head:\n", df_annotated.head(3))

    df_demographic = pd.read_csv(csv_demographic)
    print(f"\n[Demographic Data] Shape: {df_demographic.shape}")
    print("Columns:", list(df_demographic.columns))
    print("Head:\n", df_demographic.head(3))

if __name__ == "__main__":
    inspect_dataset()
