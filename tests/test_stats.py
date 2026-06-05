"""evaluation.stats のユニット"""

import math

import numpy as np
import pytest
from scipy import stats

from foveamil.evaluation.stats import (
    mean_ci_bootstrap,
    mean_ci_t,
    nadeau_bengio_corrected_t,
    wilcoxon_signed_rank,
)


def test_mean_ci_t_matches_manual():
    values = [0.80, 0.82, 0.85, 0.83, 0.81]
    mean, low, high = mean_ci_t(values, alpha=0.05)
    arr = np.asarray(values)
    expected_half = stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1)
    assert mean == pytest.approx(arr.mean())
    assert low == pytest.approx(arr.mean() - expected_half)
    assert high == pytest.approx(arr.mean() + expected_half)


def test_mean_ci_t_single_sample_is_nan_interval():
    mean, low, high = mean_ci_t([0.9])
    assert mean == pytest.approx(0.9)
    assert math.isnan(low) and math.isnan(high)


def test_bootstrap_is_reproducible_and_brackets_mean():
    values = [0.7, 0.75, 0.8, 0.85, 0.9]
    a = mean_ci_bootstrap(values, seed=0, n_boot=2000)
    b = mean_ci_bootstrap(values, seed=0, n_boot=2000)
    assert a == b  # 同 seed で完全一致
    mean, low, high = a
    assert low <= mean <= high


def test_wilcoxon_known_difference():
    a = [0.90, 0.91, 0.92, 0.93, 0.94]
    b = [0.80, 0.81, 0.82, 0.83, 0.84]  # 一貫して A > B
    out = wilcoxon_signed_rank(a, b)
    assert out["n"] == 5
    assert out["pvalue"] < 0.1


def test_wilcoxon_all_zero_diff_is_nan():
    a = [0.9, 0.9, 0.9]
    out = wilcoxon_signed_rank(a, a)
    assert math.isnan(out["pvalue"])
    assert out["n"] == 3


def test_nadeau_bengio_correction_inflates_variance():
    diffs = [0.02, 0.01, 0.03, 0.0, 0.02]
    n_train, n_test = 900, 100
    out = nadeau_bengio_corrected_t(diffs, n_train, n_test)
    # 補正 t は通常の対 t より |t| が小さい（分散を増やすため）
    arr = np.asarray(diffs)
    plain_t = arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
    assert abs(out["t"]) < abs(plain_t)
    # 補正分散の式と一致するか
    var = arr.var(ddof=1)
    corrected = var * (1.0 / len(arr) + n_test / n_train)
    assert out["t"] == pytest.approx(arr.mean() / math.sqrt(corrected))
    assert out["df"] == len(arr) - 1


def test_nadeau_bengio_zero_variance():
    out = nadeau_bengio_corrected_t([0.0, 0.0, 0.0], 900, 100)
    assert out["mean_diff"] == 0.0
    assert out["pvalue"] == 1.0


def test_nadeau_bengio_too_few_samples():
    out = nadeau_bengio_corrected_t([0.02], 900, 100)
    assert math.isnan(out["t"])
