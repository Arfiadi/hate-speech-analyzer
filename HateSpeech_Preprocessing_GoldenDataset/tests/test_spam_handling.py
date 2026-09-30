import pandas as pd

from src.preprocessing.candidates import CandidateSettings, build_candidate, prepare_source_frame


def test_spam_is_excluded_from_training() -> None:
    config = {
        "input": {"id_column": "id", "text_column": "text", "toxicity_column": "toxicity", "spam_column": "spam"},
        "filters": {"min_characters": 4, "min_tokens": 2, "exclude_spam": True, "exclude_spam_tie": True},
        "candidates": {"single_annotator_weight": 0.7, "tie_downweight": 0.5},
    }
    raw = pd.DataFrame({
        "id": ["a", "b"],
        "text": ["Pesan promosi berulang sekali", "Diskusi yang sopan dan jelas"],
        "toxicity": ["['1','1']", "['0','0']"],
        "spam": ["['1','1']", "['0','0']"],
    })
    source, _ = prepare_source_frame(raw, config)
    output = build_candidate(source, CandidateSettings("minimal", "pool", "drop", "equal"), config)
    spam_row = output.loc[output["text_id"] == "a"].iloc[0]
    assert spam_row["review_reason"] == "spam_or_noise"
    assert not spam_row["use_for_training"]
