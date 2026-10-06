from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/preprocessing_consensus_subset.csv")
OUTPUT_DIR = Path("data/preprocessing_split")

RANDOM_STATE = 42


def main():
    print("=== MEMBUAT FIXED SPLIT ===")

    # 1. Baca strict consensus subset
    df = pd.read_csv(INPUT_PATH)

    print(f"Data consensus: {len(df):,} baris")

    # 2. Pastikan tidak ada data kosong
    df = df[
        df["text_raw"].notna()
        & (df["text_raw"].astype(str).str.strip() != "")
    ].copy()

    # 3. Split pertama: 80% train, 20% sementara
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["toxicity_label"],
        random_state=RANDOM_STATE,
    )

    # 4. Split 20% menjadi:
    #    10% validation
    #    10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["toxicity_label"],
        random_state=RANDOM_STATE,
    )

    # 5. Buat folder output
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 6. Simpan
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # 7. Tampilkan hasil
    print()
    print("=== HASIL SPLIT ===")
    print(f"Train      : {len(train_df):,} baris")
    print(f"Validation : {len(val_df):,} baris")
    print(f"Test       : {len(test_df):,} baris")
    print(f"Total      : {len(train_df) + len(val_df) + len(test_df):,} baris")

    print()
    print("=== DISTRIBUSI LABEL ===")

    for name, data in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print()
        print(name)
        print(data["toxicity_label"].value_counts().sort_index())

    print()
    print("=== FILE TERSIMPAN ===")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")


if __name__ == "__main__":
    main()