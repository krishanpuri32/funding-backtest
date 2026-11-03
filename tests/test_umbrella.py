import pandas as pd

from fb import umbrella


def _link():
    # P1: 25 NCTs (umbrella at N=20, not at N=50); P2: 60 NCTs (umbrella at both); P3: P30 with 3 NCTs
    # (umbrella under the activity rule only); P4: plain R01 with 2 NCTs; P5: U10 with 60 NCTs (all three).
    rows = []
    rows += [("NCT%08d" % i, "P1", "U01") for i in range(25)]
    rows += [("NCT%08d" % i, "P2", "P01") for i in range(100, 160)]
    rows += [("NCT%08d" % i, "P3", "P30") for i in range(3)]
    rows += [("NCT%08d" % i, "P4", "R01") for i in range(2)]
    rows += [("NCT%08d" % i, "P5", "U10") for i in range(200, 260)]
    return pd.DataFrame(rows, columns=["nct_id", "core_project_num", "activity"])


def test_project_flags_three_rules():
    f = umbrella.project_flags(_link()).set_index("core_project_num")
    assert f.loc["P1", "n_nct_linked"] == 25
    assert f.loc["P1", ["is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]].tolist() == [True, False, False]
    assert f.loc["P2", ["is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]].tolist() == [True, True, False]
    assert f.loc["P3", ["is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]].tolist() == [False, False, True]
    assert f.loc["P4", ["is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]].tolist() == [False, False, False]
    assert f.loc["P5", ["is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]].tolist() == [True, True, True]


def test_n20_threshold_is_strictly_greater():
    rows = [("NCT%08d" % i, "PX", "R01") for i in range(20)]
    f = umbrella.project_flags(pd.DataFrame(rows, columns=["nct_id", "core_project_num", "activity"]))
    assert not f["is_umbrella_n20"].iloc[0]  # exactly 20 is not an umbrella


def test_trial_flags_multi_project():
    flags = umbrella.project_flags(_link())
    pairs = pd.DataFrame([
        ("NCT00000000", "P1"), ("NCT00000000", "P4"),   # umbrella + plain -> attributable under all rules
        ("NCT00000001", "P1"), ("NCT00000001", "P3"),   # n20: P3 ok; n50: both ok; activity: P1 ok
        ("NCT00000100", "P2"),                          # only P2 -> no funding under n20/n50, yes under activity
        ("NCT00000200", "P5"),                          # only P5 -> none under any rule
        ("NCT00000002", "P4"), ("NCT00000002", "P3"),   # n20/n50: both; activity: P4 only
    ], columns=["nct_id", "core_project_num"])
    t = umbrella.trial_flags(pairs, flags).set_index("nct_id")
    assert t.loc["NCT00000000", "linked_projects"] == "P1;P4"
    assert t.loc["NCT00000000", "n_linked_projects"] == 2
    assert t.loc["NCT00000000", ["n_attributable_n20", "n_attributable_n50", "n_attributable_activity"]].tolist() == [1, 2, 2]
    assert t.loc["NCT00000001", ["n_attributable_n20", "n_attributable_n50", "n_attributable_activity"]].tolist() == [1, 2, 1]
    assert t.loc["NCT00000100", ["has_attributable_funding_n20", "has_attributable_funding_n50", "has_attributable_funding_activity"]].tolist() == [False, False, True]
    assert t.loc["NCT00000200", ["has_attributable_funding_n20", "has_attributable_funding_n50", "has_attributable_funding_activity"]].tolist() == [False, False, False]
    assert t.loc["NCT00000002", ["n_attributable_n20", "n_attributable_n50", "n_attributable_activity"]].tolist() == [2, 2, 1]


def test_withdrawn_zero_enroll_flag():
    st = pd.Series(["WITHDRAWN", "WITHDRAWN", "WITHDRAWN", "Withdrawn", "TERMINATED", "COMPLETED", None])
    en = pd.Series(["0", "5", None, "0", "0", "0", "0"])
    assert umbrella.withdrawn_zero_enroll(st, en).tolist() == [True, False, False, True, False, False, False]
