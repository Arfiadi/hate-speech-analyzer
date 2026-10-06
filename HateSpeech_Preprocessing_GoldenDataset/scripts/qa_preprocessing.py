from pathlib import Path
import pandas as pd


BASE_DIR = Path("data/preprocessing_variants")
REPORT_DIR = Path("reports")


def load_data(variant, split):
    path = BASE_DIR / variant / f"{split}.csv"
    return pd.read_csv(path)


def main():
    print("=== QA PREPROCESSING ===")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    variants = ["p_a_raw", "p_b_heavy", "p_c_minimal"]
    splits = ["train", "validation", "test"]

    quality_results = []
    sample_results = []
    statistics_results = []

    # =========================================================
    # 1. QA ROW, ID, LABEL
    # =========================================================
    print("\n[1] Mengecek konsistensi row, ID, dan label...")

    for split in splits:
        data = {
            variant: load_data(variant, split)
            for variant in variants
        }

        base = data["p_a_raw"]

        for variant in variants:
            df = data[variant]

            same_rows = len(df) == len(base)
            same_ids = df["text_id"].equals(base["text_id"])
            same_labels = df["toxicity_label"].equals(
                base["toxicity_label"]
            )

            quality_results.append({
                "split": split,
                "variant": variant,
                "row_count": len(df),
                "same_row_count_as_raw": same_rows,
                "same_text_id_as_raw": same_ids,
                "same_label_as_raw": same_labels,
                "status": (
                    "PASS"
                    if same_rows and same_ids and same_labels
                    else "FAIL"
                ),
            })

            print(
                f"{split:10s} | "
                f"{variant:12s} | "
                f"rows={len(df):5d} | "
                f"ID={'PASS' if same_ids else 'FAIL'} | "
                f"label={'PASS' if same_labels else 'FAIL'}"
            )

    # =========================================================
    # 2. BEFORE-AFTER SAMPLE
    # =========================================================
    print("\n[2] Membuat contoh before-after...")

    train_raw = load_data("p_a_raw", "train")
    train_heavy = load_data("p_b_heavy", "train")
    train_minimal = load_data("p_c_minimal", "train")

    for i in range(min(10, len(train_raw))):
        sample_results.append({
            "sample_number": i + 1,
            "text_id": train_raw.iloc[i]["text_id"],
            "toxicity_label": train_raw.iloc[i]["toxicity_label"],
            "text_raw": train_raw.iloc[i]["text_raw"],
            "p_a_raw": train_raw.iloc[i]["text"],
            "p_b_heavy": train_heavy.iloc[i]["text"],
            "p_c_minimal": train_minimal.iloc[i]["text"],
        })

    # =========================================================
    # 3. STATISTIK TEKS
    # =========================================================
    print("\n[3] Menghitung statistik preprocessing...")

    for variant in variants:
        for split in splits:
            df = load_data(variant, split)

            text = df["text"].fillna("").astype(str)

            char_lengths = text.str.len()
            word_counts = text.str.split().str.len()

            statistics_results.append({
                "variant": variant,
                "split": split,
                "row_count": len(df),
                "avg_char_length": round(char_lengths.mean(), 2),
                "median_char_length": round(char_lengths.median(), 2),
                "avg_word_count": round(word_counts.mean(), 2),
                "empty_text_count": int((text.str.strip() == "").sum()),
                "url_count": int(
                    text.str.contains(
                        r"https?://\S+|www\.\S+",
                        regex=True,
                        case=False
                    ).sum()
                ),
                "mention_count": int(
                    text.str.contains(
                        r"@\w+",
                        regex=True
                    ).sum()
                ),
            })

    # =========================================================
    # 4. SIMPAN REPORT
    # =========================================================
    quality_path = REPORT_DIR / "preprocessing_quality_check.csv"
    sample_path = REPORT_DIR / "preprocessing_before_after_samples.csv"
    statistics_path = REPORT_DIR / "preprocessing_statistics.csv"

    pd.DataFrame(quality_results).to_csv(
        quality_path,
        index=False
    )

    pd.DataFrame(sample_results).to_csv(
        sample_path,
        index=False
    )

    pd.DataFrame(statistics_results).to_csv(
        statistics_path,
        index=False
    )

    # =========================================================
    # 5. FINAL STATUS
    # =========================================================
    quality_df = pd.DataFrame(quality_results)

    all_pass = (quality_df["status"] == "PASS").all()

    print("\n=== QA SELESAI ===")
    print(f"Quality Check : {quality_path}")
    print(f"Before-After  : {sample_path}")
    print(f"Statistics    : {statistics_path}")

    print()

    if all_pass:
        print("STATUS AKHIR: PASS")
        print("Semua row, text_id, dan toxicity_label konsisten.")
    else:
        print("STATUS AKHIR: FAIL")
        print("Ada perbedaan row, text_id, atau label.")


if __name__ == "__main__":
    main()