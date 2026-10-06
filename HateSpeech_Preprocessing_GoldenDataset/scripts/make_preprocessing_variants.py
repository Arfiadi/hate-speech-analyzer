import re
from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/preprocessing_split")
OUTPUT_DIR = Path("data/preprocessing_variants")

def clean_raw(text):
    if not isinstance(text, str):
        return ""
    return text


def clean_heavy(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"#(\w+)", r" \1 ", text)

    text = re.sub(
        r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
        " ",
        text
    )

    text = re.sub(r"[^\w\s]", " ", text)
    
    text = re.sub(r"\d+", " ", text)

    stopwords = {
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "ini",
        "itu",
        "untuk",
        "dengan",
        "atau",
        "pada",
        "sebagai",
        "oleh",
        "dalam",
        "kami",
        "kita",
        "saya",
        "kamu",
        "dia",
        "mereka",
        "ada",
        "akan",
        "sudah",
        "belum",
        "saja",
        "juga",
        "karena",
        "agar",
    }

    tokens = [
        token
        for token in re.split(r"\s+", text)
        if token and token not in stopwords
    ]

    return " ".join(tokens)


def clean_minimal(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " [URL] ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"@\w+", " [USER] ", text)

    text = re.sub(r"#(\w+)", r" \1 ", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"([!?.,])\1{2,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_split(split_name):
    input_path = INPUT_DIR / f"{split_name}.csv"

    df = pd.read_csv(input_path)

    df["text_raw"] = (
        df["text_raw"]
        .fillna("")
        .astype(str)
    )

    # P-A: Raw
    df["text"] = df["text_raw"].apply(clean_raw)

    df.to_csv(
        OUTPUT_DIR / "p_a_raw" / f"{split_name}.csv",
        index=False
    )

    # P-B: Heavy Cleaning
    df["text"] = df["text_raw"].apply(clean_heavy)

    df.to_csv(
        OUTPUT_DIR / "p_b_heavy" / f"{split_name}.csv",
        index=False
    )

    # P-C: Minimal Transformer Cleaning
    df["text"] = df["text_raw"].apply(clean_minimal)

    df.to_csv(
        OUTPUT_DIR / "p_c_minimal" / f"{split_name}.csv",
        index=False
    )

    return len(df)


def main():
    print("=== MEMBUAT VARIAN PREPROCESSING ===")

    for folder in [
        "p_a_raw",
        "p_b_heavy",
        "p_c_minimal",
    ]:
        (
            OUTPUT_DIR / folder
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    total = 0

    for split in [
        "train",
        "validation",
        "test",
    ]:
        count = process_split(split)

        total += count

        print(
            f"{split:10s}: {count:,} data"
        )

    print()
    print("=== SELESAI ===")
    print(
        f"Total data diproses: {total:,}"
    )

    print()
    print("Output:")
    print(
        OUTPUT_DIR / "p_a_raw"
    )
    print(
        OUTPUT_DIR / "p_b_heavy"
    )
    print(
        OUTPUT_DIR / "p_c_minimal"
    )


if __name__ == "__main__":
    main()