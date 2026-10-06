import re
from pathlib import Path
import pandas as pd


BASE_DIR = Path("data/preprocessing_variants/p_c_minimal")
OUTPUT_PATH = Path("reports/p_c_rule_check.csv")


def check_url_masking(df):
    original = df["text_raw"].fillna("").astype(str)
    processed = df["text"].fillna("").astype(str)

    original_has_url = original.str.contains(
        r"https?://\S+|www\.\S+",
        regex=True,
        case=False
    )

    processed_has_raw_url = processed.str.contains(
        r"https?://\S+|www\.\S+",
        regex=True,
        case=False
    )

    processed_has_url_token = processed.str.contains(
        r"\[URL\]",
        regex=False
    )

    return int(
        (original_has_url & processed_has_url_token).sum()
    ), int(
        (original_has_url & processed_has_raw_url).sum()
    )


def check_mention_masking(df):
    original = df["text_raw"].fillna("").astype(str)
    processed = df["text"].fillna("").astype(str)

    original_has_mention = original.str.contains(
        r"@\w+",
        regex=True
    )

    processed_has_raw_mention = processed.str.contains(
        r"@\w+",
        regex=True
    )

    processed_has_user_token = processed.str.contains(
        r"\[USER\]",
        regex=False
    )

    return int(
        (original_has_mention & processed_has_user_token).sum()
    ), int(
        (original_has_mention & processed_has_raw_mention).sum()
    )


def main():
    print("=== QA KHUSUS P-C MINIMAL ===")

    results = []

    for split in ["train", "validation", "test"]:
        path = BASE_DIR / f"{split}.csv"
        df = pd.read_csv(path)

        print(f"\n--- {split.upper()} ---")
        print(f"Jumlah data: {len(df):,}")

        # URL
        url_masked, url_remaining = check_url_masking(df)

        # Mention
        mention_masked, mention_remaining = check_mention_masking(df)

        # Repeated characters
        text = df["text"].fillna("").astype(str)

        repeated_chars_remaining = int(
            text.str.contains(
                r"(.)\1{2,}",
                regex=True
            ).sum()
        )

        repeated_punctuation_remaining = int(
            text.str.contains(
                r"([!?.,])\1{2,}",
                regex=True
            ).sum()
        )

        # Empty text
        empty_text = int(
            (text.str.strip() == "").sum()
        )

        # Token P-C
        url_token_count = int(
            text.str.count(r"\[URL\]").sum()
        )

        user_token_count = int(
            text.str.count(r"\[USER\]").sum()
        )

        result = {
            "split": split,
            "row_count": len(df),
            "url_masked_count": url_masked,
            "raw_url_remaining": url_remaining,
            "mention_masked_count": mention_masked,
            "raw_mention_remaining": mention_remaining,
            "repeated_char_remaining": repeated_chars_remaining,
            "repeated_punctuation_remaining": repeated_punctuation_remaining,
            "empty_text_count": empty_text,
            "URL_token_count": url_token_count,
            "USER_token_count": user_token_count,
        }

        results.append(result)

        print(f"URL berhasil dimask      : {url_masked}")
        print(f"URL mentah tersisa       : {url_remaining}")
        print(f"Mention berhasil dimask  : {mention_masked}")
        print(f"Mention mentah tersisa   : {mention_remaining}")
        print(f"Repeated char tersisa    : {repeated_chars_remaining}")
        print(f"Repeated punctuation     : {repeated_punctuation_remaining}")
        print(f"Empty text               : {empty_text}")
        print(f"[URL] token              : {url_token_count}")
        print(f"[USER] token             : {user_token_count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== QA P-C SELESAI ===")
    print(f"Report: {OUTPUT_PATH}")

    failed = result_df[
        (result_df["raw_url_remaining"] > 0)
        | (result_df["raw_mention_remaining"] > 0)
        | (result_df["repeated_char_remaining"] > 0)
        | (result_df["repeated_punctuation_remaining"] > 0)
        | (result_df["empty_text_count"] > 0)
    ]

    if len(failed) == 0:
        print("STATUS AKHIR: PASS")
        print("Aturan utama P-C terpenuhi.")
    else:
        print("STATUS AKHIR: CHECK")
        print("Ada beberapa kondisi yang perlu diperiksa.")


if __name__ == "__main__":
    main()