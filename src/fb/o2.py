"""O2 — recruitment-related failure (plan §4): keyword rule set on `why_stopped`, and the
200-string blind-labelling sample.

The rule set is exactly the plan's keyword list and its variants, nothing added:
accrual, enrol, recruit, participants, slow, low, insufficient, eligible. Matching is on
word starts, case-insensitive (so 'enrol' matches enrolment / enrollment / enrolled /
under-enrollment, but 'low' does not match 'follow' or 'allow'). The rule is NOT
validated until the blind labels exist; precision / recall are computed in Phase 4 step 1 (fb.o2.validate).
Known exposure, left in on purpose (the plan fixes the list, validation measures it):
'low' and 'insufficient' also fire on funding reasons ('insufficient funding').
"""
from __future__ import annotations

import re

import pandas as pd

from . import config

KEYWORD_PATTERNS = {
    "accrual": r"\baccru\w*",            # accrual, accrue, accruing, poor-accrual
    "enrol": r"\benrol\w*",              # enrol, enroll, enrolment, enrollment, enrolled
    "recruit": r"\brecruit\w*",          # recruit, recruitment, recruiting
    "participants": r"\bparticipants?\b",
    "slow": r"\bslow\w*",                # slow, slower, slowly
    "low": r"\blow(er|est)?\b",          # low, lower, lowest (not 'follow', 'allow', 'below')
    "insufficient": r"\binsufficient\w*",
    "eligible": r"\b(in)?eligib\w*",     # eligible, eligibility, ineligible
}
RULE = re.compile("|".join(f"(?:{p})" for p in KEYWORD_PATTERNS.values()), flags=re.IGNORECASE)
SAMPLE_COLUMNS = ["nct_id", "why_stopped", "label", "rule_label"]


def rule_label(why_stopped: pd.Series) -> pd.Series:
    """1 if any keyword matches, 0 if the string is present and none match, NA if blank."""
    s = why_stopped.astype("object")
    blank = s.isna() | s.astype(str).str.strip().eq("")
    hit = s.fillna("").astype(str).map(lambda t: int(bool(RULE.search(t))))
    return hit.astype("Int64").mask(blank, pd.NA)


def matched_keywords(text: str) -> list[str]:
    return [k for k, p in KEYWORD_PATTERNS.items() if re.search(p, text or "", flags=re.IGNORECASE)]


def draw_sample(events: pd.DataFrame, n: int = config.O2_SAMPLE_N, seed: int = config.SEED) -> pd.DataFrame:
    """Fixed-seed sample of n training O1 events with a non-blank why_stopped.

    events: nct_id, why_stopped. Sorted by nct_id before sampling so the draw is
    reproducible regardless of input order. `label` is left blank for the blind labeller;
    `rule_label` holds the rule's verdict (hidden while labelling).
    """
    e = events[["nct_id", "why_stopped"]].copy()
    e = e[e["why_stopped"].notna() & e["why_stopped"].astype(str).str.strip().ne("")]
    e = e.sort_values("nct_id").reset_index(drop=True)
    s = e.sample(n=min(n, len(e)), random_state=seed).sort_values("nct_id").reset_index(drop=True)
    s["label"] = ""
    s["rule_label"] = rule_label(s["why_stopped"])
    return s[SAMPLE_COLUMNS]


def validate(sample: pd.DataFrame) -> dict:
    """Phase 4 step 1: the rule (recomputed from the text, not the stored column) against the
    blind labels. Returns the confusion matrix, precision, recall and the FP / FN strings."""
    s = sample.copy()
    s["label"] = pd.to_numeric(s["label"], errors="raise").astype(int)
    s["rule"] = rule_label(s["why_stopped"]).astype(int)
    tp = int(((s["rule"] == 1) & (s["label"] == 1)).sum())
    fp = int(((s["rule"] == 1) & (s["label"] == 0)).sum())
    fn = int(((s["rule"] == 0) & (s["label"] == 1)).sum())
    tn = int(((s["rule"] == 0) & (s["label"] == 0)).sum())
    return {
        "n": int(len(s)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "accuracy": (tp + tn) / len(s) if len(s) else float("nan"),
        "stored_rule_label_agrees": bool((s["rule"] == pd.to_numeric(s["rule_label"])).all()),
        "false_positives": [{"nct_id": r.nct_id, "why_stopped": r.why_stopped, "keywords": matched_keywords(r.why_stopped)}
                            for r in s[(s["rule"] == 1) & (s["label"] == 0)].itertuples()],
        "false_negatives": [{"nct_id": r.nct_id, "why_stopped": r.why_stopped}
                            for r in s[(s["rule"] == 0) & (s["label"] == 1)].itertuples()],
    }


def main() -> dict:
    """Validate on the labelled sample, then label every O1 event in both windows."""
    import json

    from . import tables

    res = validate(pd.read_csv(config.WHY_STOPPED_SAMPLE_PATH, dtype={"label": str}))
    inp = tables.load_inputs()
    counts = {}
    for w in ("train", "test"):
        lab = tables.labels(inp, tables.window_trials(inp, w))
        ev = lab[lab["o1"] == 1]
        counts[w] = {"o1_events": int(len(ev)), "why_stopped_blank": int(ev["o2_rule"].isna().sum()),
                     "o2_events": int((lab["o2"] == 1).sum()), "o1_events_rule_0": int(ev["o2_rule"].eq(0).sum())}
        for rule in ("n20",):
            a = tables.attributable(tables.window_trials(inp, w), rule)
            r = tables.observable_reason(inp, a, rule)
            lm = lab[lab["nct_id"].isin(set(r.loc[r["f_rows_reason"].isin(config.LANDMARK_REASONS), "nct_id"]))]
            counts[w]["o2_events_landmark_n20"] = int((lm["o2"] == 1).sum())
            counts[w]["o2_events_attributable_n20"] = int((lab[lab["nct_id"].isin(set(a["nct_id"]))]["o2"] == 1).sum())
    res["o2_counts"] = counts
    res["keywords"] = KEYWORD_PATTERNS
    config.O2_VALIDATION_PATH.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
