import math

from fb.stats import wilson_ci


def test_wilson_known_values():
    # Textbook: 10/100 -> (0.0552, 0.1744) at 95 %
    lo, hi = wilson_ci(10, 100)
    assert abs(lo - 0.05522) < 2e-4 and abs(hi - 0.17436) < 2e-4
    # 0/10 -> lower bound 0, upper 0.2775
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.27753) < 2e-4
    # n = 0 -> nan
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
