"""cv.aggregate_folds / aggregate_folds_ci のユニット"""

import numpy as np

from foveamil.training.cv import aggregate_folds, aggregate_folds_ci


def _per_fold():
    return [
        {"accuracy": 0.90, "macro_auc": 0.95, "weighted_f1": 0.89},
        {"accuracy": 0.88, "macro_auc": 0.93, "weighted_f1": 0.87},
        {"accuracy": 0.92, "macro_auc": 0.96, "weighted_f1": 0.91},
    ]


def test_aggregate_folds_ci_mean_std_match_plain():
    per_fold = _per_fold()
    plain = aggregate_folds(per_fold)
    ci = aggregate_folds_ci(per_fold, n_boot=1000)
    for metric in ("accuracy", "macro_auc", "weighted_f1"):
        assert ci[metric]["mean"] == plain[metric]["mean"]
        assert ci[metric]["std"] == plain[metric]["std"]
        assert ci[metric]["n"] == 3
        # CI 下限 <= mean <= 上限
        assert ci[metric]["ci_t_low"] <= ci[metric]["mean"] <= ci[metric]["ci_t_high"]
        assert ci[metric]["ci_boot_low"] <= ci[metric]["mean"] <= ci[metric]["ci_boot_high"]


def test_aggregate_folds_ci_skips_missing_metric():
    per_fold = [{"accuracy": 0.9}, {"accuracy": 0.8, "macro_auc": 0.95}]
    ci = aggregate_folds_ci(per_fold, n_boot=200)
    assert ci["accuracy"]["n"] == 2
    # macro_auc は 1 fold のみ -> n=1, CI は nan
    assert ci["macro_auc"]["n"] == 1
    assert np.isnan(ci["macro_auc"]["ci_t_low"])
