import numpy as np
import pytest
from foveamil.evaluation.calibration import (
    optimize_temperature,
    optimize_offsets,
    apply_calibration,
)

def test_optimize_temperature():
    np.random.seed(42)
    n_samples = 100
    n_cls = 3
    y_true = np.random.randint(0, n_cls, n_samples)
    logits = np.random.randn(n_samples, n_cls) * 2.0 # High variance for scaling
    
    t_opt = optimize_temperature(y_true, logits)
    assert t_opt > 0
    
    # Verify that applying it works
    _, y_prob = apply_calibration(logits, temperature=t_opt)
    assert y_prob.shape == (n_samples, n_cls)
    assert np.allclose(y_prob.sum(axis=1), 1.0)

def test_optimize_offsets():
    np.random.seed(42)
    n_samples = 100
    n_cls = 3
    # Create an imbalanced set where minority is class 2
    y_true = np.concatenate([
        np.zeros(45, dtype=int),
        np.ones(45, dtype=int),
        np.full(10, 2, dtype=int)
    ])
    np.random.shuffle(y_true)
    
    # Random logits that are somewhat correlated with y_true
    logits = np.random.randn(len(y_true), n_cls)
    for i in range(len(y_true)):
        logits[i, y_true[i]] += 0.5 # Weak signal
    
    offsets = optimize_offsets(y_true, logits, target_metric="macro_f1")
    assert len(offsets) == n_cls
    
    # Test minority recall optimization
    offsets_rec = optimize_offsets(y_true, logits, target_metric="minority_recall", minority_indices=[2])
    assert len(offsets_rec) == n_cls
    
    # Applying offsets should yield valid predictions
    y_hat, _ = apply_calibration(logits, offsets=offsets_rec)
    assert len(y_hat) == len(y_true)
    assert y_hat.max() < n_cls

def test_calibration_deterministic():
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1])
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5], [0.5, 1.0]])
    
    t1 = optimize_temperature(y_true, logits)
    t2 = optimize_temperature(y_true, logits)
    assert t1 == t2
    
    o1 = optimize_offsets(y_true, logits)
    o2 = optimize_offsets(y_true, logits)
    assert np.all(o1 == o2)
