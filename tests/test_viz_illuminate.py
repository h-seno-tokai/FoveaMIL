"""visualization.render.illuminate / normalize のユニット"""

import numpy as np

from foveamil.visualization.render.illuminate import (
    build_child_slot_set,
    illuminate_children,
)
from foveamil.visualization.render.normalize import (
    normalize,
    shared_scale,
    to_minmax,
    to_percentile,
)


def test_build_child_slot_set():
    # parent_global=7, cpp=4 -> 子 global 28..31 が slot 0..3
    slots = build_child_slot_set([28, 31], parent_global=7, cpp=4)
    assert slots == {0, 3}
    # 親の子でない global は無視
    assert build_child_slot_set([100], parent_global=7, cpp=4) == set()
    assert build_child_slot_set(None, 7, 4) == set()


def test_illuminate_continuous_brightness():
    zoom_px, ratio = 100, 2  # cell=50, 4 子
    parent = np.full((zoom_px, zoom_px, 3), 200, dtype=np.uint8)
    # slot 0=1.0(原輝度), slot 3=0.0(最も減光), slot 1=0.5, slot 2=0.25
    scores = [1.0, 0.5, 0.25, 0.0]
    out = illuminate_children(parent, scores, ratio, zoom_px, dim_factor=0.3)

    def cell_mean(slot):
        x, y, w, h = (slot % 2) * 50, (slot // 2) * 50, 50, 50
        return out[y:y + h, x:x + w].mean()

    # score 1.0 -> 原輝度 ~200, score 0.0 -> 200*0.3=60
    assert cell_mean(0) > 195
    assert abs(cell_mean(3) - 200 * 0.3) < 2
    # 連続単調: score 大ほど明るい
    assert cell_mean(0) > cell_mean(1) > cell_mean(2) > cell_mean(3)


def test_illuminate_draws_selected_border():
    zoom_px, ratio = 64, 2
    parent = np.full((zoom_px, zoom_px, 3), 100, dtype=np.uint8)
    out = illuminate_children(
        parent, [1, 1, 1, 1], ratio, zoom_px, selected_slots={0}, dim_factor=0.3
    )
    # slot0 の左上隅にマゼンタ枠(255,0,255)
    assert tuple(out[0, 0]) == (255, 0, 255)
    # 非選択 slot3 の隅は枠なし
    assert tuple(out[63, 63]) != (255, 0, 255)


def test_to_percentile_monotone_and_unit():
    out = to_percentile([0.1, 0.9, 0.5, 0.3])
    assert out.max() == 1.0
    # 順位が単調: 引数の大小と percentile の大小が一致
    order = np.argsort([0.1, 0.9, 0.5, 0.3])
    assert list(np.argsort(out)) == list(order)


def test_to_minmax_and_shared_scale():
    assert list(to_minmax([2.0, 4.0, 6.0])) == [0.0, 0.5, 1.0]
    assert to_minmax([3.0, 3.0]).tolist() == [0.0, 0.0]
    assert shared_scale([[0.1, 0.2], [0.05, 0.9]]) == (0.05, 0.9)


def test_normalize_result_keeps_vmin_vmax():
    res = normalize([0.2, 0.8, 0.5], kind="percentile")
    assert res.kind == "percentile"
    assert res.vmin == 0.2 and res.vmax == 0.8
    assert res.values01.max() == 1.0
