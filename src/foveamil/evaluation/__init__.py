from __future__ import annotations

from foveamil.evaluation.calibration import (
    apply_calibration,
    calibrate_val_to_test,
    evaluate_predictions,
    extract_logits,
    fit_class_deltas,
    fit_temperature,
)
from foveamil.evaluation.curves import (
    epoch_curve,
    per_class_f1_bars,
    plot_curves,
    plot_per_class_f1,
    plot_summary_bars,
    summary_bars,
)
from foveamil.evaluation.report import (
    compare_combos,
    compute_ece,
    load_predictions,
    plot_calibration,
    plot_pr,
    plot_roc,
    pool_predictions,
)
from foveamil.evaluation.group_metrics import (
    pool_combo_predictions,
    pooled_group_f1,
    pooled_group_f1_from_predictions,
)
from foveamil.evaluation.stats import (
    mean_ci_bootstrap,
    mean_ci_t,
    nadeau_bengio_corrected_t,
    paired_group_f1_permutation_test,
    stratified_bootstrap_group_f1_ci,
    wilcoxon_signed_rank,
)

__all__ = [
    "load_predictions",
    "pool_predictions",
    "compute_ece",
    "extract_logits",
    "fit_temperature",
    "fit_class_deltas",
    "apply_calibration",
    "evaluate_predictions",
    "calibrate_val_to_test",
    "plot_roc",
    "plot_pr",
    "plot_calibration",
    "compare_combos",
    "epoch_curve",
    "plot_curves",
    "summary_bars",
    "plot_summary_bars",
    "per_class_f1_bars",
    "plot_per_class_f1",
    "mean_ci_t",
    "mean_ci_bootstrap",
    "wilcoxon_signed_rank",
    "nadeau_bengio_corrected_t",
    "paired_group_f1_permutation_test",
    "stratified_bootstrap_group_f1_ci",
    "pooled_group_f1",
    "pooled_group_f1_from_predictions",
    "pool_combo_predictions",
]
