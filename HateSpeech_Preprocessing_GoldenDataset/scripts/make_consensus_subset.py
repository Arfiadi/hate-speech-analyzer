import ast
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/indotoxic2024_annotated_data_v2_final.csv")
OUTPUT_PATH = Path("data/preprocessing_consensus_subset.csv")


def parse_list(value):
    """Mengubah string list menjadi list integer."""
    try:
        result = ast.literal_eval(value)

        if not isinstance(result, list):
            return []

        # ['0', '1'] -> [0, 1]
        return [int(x) for x in result]

    except (ValueError, SyntaxError, TypeError):
        return []


def main():
    print("=== MEMBUAT STRICT CONSENSUS SUBSET ===")

    # 1. Baca dataset
    df = pd.read_csv(INPUT_PATH)

    print(f"Data awal: {len(df):,} baris")

    # 2. Parse annotator dan toxicity
    df["annotators_list"] = df["annotators_id"].apply(parse_list)
    df["toxicity_votes"] = df["toxicity"].apply(parse_list)

    # 3. Jumlah annotator
    df["n_annotators"] = df["annotators_list"].apply(len)

    # 4. Minimal 2 annotator
    df = df[df["n_annotators"] >= 2].copy()

    print(f"Setelah minimal 2 annotator: {len(df):,} baris")

    # 5. Pastikan jumlah vote toxicity sama dengan jumlah annotator
    df = df[
        (df["toxicity_votes"].apply(len) == df["n_annotators"])
        & (df["toxicity_votes"].apply(
            lambda x: all(v in [0, 1] for v in x)
        ))
    ].copy()

    # 6. Strict consensus:
    # semua annotator harus memberikan label yang sama
    df["is_strict_consensus"] = df["toxicity_votes"].apply(
        lambda x: len(set(x)) == 1
    )

    df = df[df["is_strict_consensus"]].copy()

    print(f"Setelah strict consensus: {len(df):,} baris")

    # 7. Label akhir
    df["toxicity_label"] = df["toxicity_votes"].apply(
        lambda x: x[0]
    )

    # 8. Validasi teks
    df["text_raw"] = df["text"]

    df = df[
        df["text_raw"].notna()
        & (df["text_raw"].astype(str).str.strip() != "")
    ].copy()

    print(f"Setelah validasi teks: {len(df):,} baris")

    # 9. Pilih kolom output
    output = df[
        [
            "text_id",
            "text_raw",
            "toxicity_label",
            "n_annotators",
            "annotators_id",
            "toxicity",
        ]
    ].copy()

    # 10. Pastikan label hanya 0 atau 1
    output = output[
        output["toxicity_label"].isin([0, 1])
    ].copy()

    # 11. Simpan
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 12. Ringkasan
    print()
    print("=== SELESAI ===")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Jumlah data : {len(output):,} baris")

    print()
    print("Distribusi label:")
    print(
        output["toxicity_label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Distribusi jumlah annotator:")
    print(
        output["n_annotators"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()