import pandas as pd

from fb import link


def test_bucket_projects_per_nct():
    n = pd.Series([1, 2, 3, 5, 6, 40])
    assert link._bucket_projects_per_nct(n).astype(str).tolist() == ["1", "2", "3-5", "3-5", ">5", ">5"]


def test_bucket_ncts_per_project():
    n = pd.Series([1, 2, 5, 6, 20, 21, 500])
    assert link._bucket_ncts_per_project(n).astype(str).tolist() == ["1", "2-5", "2-5", "6-20", "6-20", ">20", ">20"]


def test_norm_nct_and_regex():
    s = pd.Series([" nct01234567 ", "NCT00000001", "NCT1234"])
    out = link._norm_nct(s)
    assert out.tolist() == ["NCT01234567", "NCT00000001", "NCT1234"]
    assert out.str.match(link.NCT_RE).tolist() == [True, True, False]
