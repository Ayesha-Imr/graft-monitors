"""Pure parts of the labeler (no model code is executed). Runs on the laptop."""
import time
from collections import Counter
from fractions import Fraction

import pytest

from labeling.countdown_labeler import reuse_expression


def _uses(expr):
    import re
    return Counter(int(x) for x in re.findall(r"\d+", expr))


@pytest.mark.parametrize("nums,target", [([3, 5, 7], 15), ([2, 9, 4], 81), ([10, 3, 8, 6], 100),
                                         ([25, 4, 7, 3], 96), ([1, 2, 3], 9)])
def test_reuse_expression_reaches_target_and_reuses(nums, target):
    e = reuse_expression(tuple(nums), target)
    assert e is not None
    assert Fraction(eval(e)) == target
    used, avail = _uses(e), Counter(nums)
    assert set(used) <= set(avail)                       # only the given numbers
    assert any(used[x] > avail[x] for x in used)          # some number reused


def test_reuse_expression_fast_enough():
    t = time.time()
    for k in range(30):
        reuse_expression((k + 2, k + 5, 7, 11), 97 + k)
    assert time.time() - t < 30
