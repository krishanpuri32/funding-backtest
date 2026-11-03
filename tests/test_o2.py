"""O2 keyword rule set on hand-written why_stopped strings, and the fixed-seed sample."""
import pandas as pd

from fb import o2

CASES = [
    ("Slow accrual", 1),
    ("Poor enrollment", 1),
    ("Study closed due to low enrolment", 1),
    ("Difficulty recruiting participants", 1),
    ("Not enough eligible patients", 1),
    ("Insufficient number of subjects", 1),
    ("Under-enrolled", 1),
    ("PI left the institution", 0),
    ("Lack of efficacy at interim analysis", 0),
    ("Sponsor decision; follow-up not allowed by IRB", 0),   # 'follow' / 'allowed' must not match 'low'
    ("Drug supply ended", 0),
    ("Funding ended", 0),
    ("Insufficient funding", 1),  # known false positive the validation will measure
]


def test_rule_on_hand_written_strings():
    got = o2.rule_label(pd.Series([s for s, _ in CASES])).tolist()
    assert got == [y for _, y in CASES]


def test_validate_confusion_matrix():
    s = pd.DataFrame({"nct_id": ["A", "B", "C", "D", "E"],
                      "why_stopped": ["Slow accrual", "Insufficient funding", "PI left", "no enough patients", "Poor enrollment"],
                      "label": ["1", "0", "0", "1", "1"], "rule_label": [1, 1, 0, 0, 1]})
    r = o2.validate(s)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 1)
    assert r["precision"] == 2 / 3 and r["recall"] == 2 / 3
    assert [x["nct_id"] for x in r["false_positives"]] == ["B"]
    assert [x["nct_id"] for x in r["false_negatives"]] == ["D"]
    assert r["stored_rule_label_agrees"]


def test_blank_is_na():
    got = o2.rule_label(pd.Series([None, "  ", "slow accrual"]))
    assert got.isna().tolist() == [True, True, False]


def test_matched_keywords():
    assert o2.matched_keywords("Slow recruitment of eligible participants") == ["recruit", "participants", "slow", "eligible"]


def test_sample_is_fixed_seed_blind_and_skips_blank():
    ev = pd.DataFrame({"nct_id": [f"NCT{i:08d}" for i in range(300)],
                       "why_stopped": ["slow accrual" if i % 3 else "PI left" for i in range(300)]})
    ev.loc[5, "why_stopped"] = None
    a = o2.draw_sample(ev, n=200, seed=1)
    b = o2.draw_sample(ev.sample(frac=1, random_state=9), n=200, seed=1)  # input order must not matter
    assert a["nct_id"].tolist() == b["nct_id"].tolist()
    assert list(a.columns) == ["nct_id", "why_stopped", "label", "rule_label"]
    assert len(a) == 200 and a["nct_id"].is_unique and "NCT00000005" not in set(a["nct_id"])
    assert (a["label"] == "").all()


def test_write_sample_keeps_labelled_file(tmp_path):
    import pandas as pd
    import pytest

    from fb.features import write_sample
    p = tmp_path / "s.csv"
    pd.DataFrame({"nct_id": ["A", "B"], "why_stopped": ["x", "y"], "label": ["1", "0"], "rule_label": [1, 0]}).to_csv(p, index=False)
    write_sample(pd.DataFrame({"nct_id": ["A", "B"], "why_stopped": ["x", "y"], "label": [None, None], "rule_label": [1, 0]}), p)
    assert pd.read_csv(p, dtype=str)["label"].tolist() == ["1", "0"]
    with pytest.raises(SystemExit):
        write_sample(pd.DataFrame({"nct_id": ["A", "C"], "why_stopped": ["x", "z"], "label": [None, None], "rule_label": [1, 0]}), p)
