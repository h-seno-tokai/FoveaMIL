"""visualization.render.region_reader.best_level_for_side のユニット

features._best_level_and_read_size と (best_level, read_size) が数値一致することを，
ダミーの level_downsamples を持つ fake openslide で確認する（コピペでなく挙動一致）
"""

import numpy as np
import pytest

from foveamil.preprocessing.features import _best_level_and_read_size
from foveamil.visualization.render.region_reader import best_level_for_side


class _FakeWSI:
    def __init__(self, level_downsamples):
        self.level_downsamples = level_downsamples
        self.level_count = len(level_downsamples)


@pytest.mark.parametrize(
    "downsamples,mag,max_mag,patch",
    [
        ([1.0, 4.0, 16.0, 64.0], 1.25, 40, 224),
        ([1.0, 4.0, 16.0, 64.0], 5.0, 40, 224),
        ([1.0, 4.0, 16.0, 64.0], 40.0, 40, 224),
        ([1.0, 2.0009, 4.0, 8.0], 10.0, 40, 224),
        ([1.0], 20.0, 40, 224),
    ],
)
def test_best_level_matches_features(downsamples, mag, max_mag, patch):
    fake = _FakeWSI(downsamples)
    expected = _best_level_and_read_size(fake, mag, max_mag, patch)
    side0 = int(np.ceil(patch * (max_mag / mag)))
    got = best_level_for_side(downsamples, side0, patch)
    assert got == expected
