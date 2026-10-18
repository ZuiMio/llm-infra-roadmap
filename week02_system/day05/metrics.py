import numpy as np


def summarize_step_times(step_times):
    """Summarize a sequence of training step durations in seconds."""
    values = np.asarray(step_times, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("step_times must be one-dimensional.")

    if values.size == 0:
        raise ValueError("step_times must not be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("step_times must contain only finite values.")

    if np.any(values < 0):
        raise ValueError("step_times must be non-negative.")

    return {
        "count": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "max_seconds": float(np.max(values)),
    }
