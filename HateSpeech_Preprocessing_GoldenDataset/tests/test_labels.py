from src.preprocessing.labels import parse_votes, vote_summary


def test_parse_votes_accepts_dataset_string() -> None:
    assert parse_votes("['1', '0', '1']") == [1, 0, 1]


def test_tie_is_not_silently_hardened() -> None:
    summary = vote_summary([1, 0])
    assert summary["label"] == 0.5
    assert summary["status"] == "tie"
