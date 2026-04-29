"""Depth metrics — RadarMarigold-compatible per-range bins + day/night split + far-only.

Range schema:
    overall   = [(0,50), (0,70), (0,80)]      cumulative ranges
    per_range = [(0,10), (10,20), ..., (70,80)] disjoint 10-m bins (long-range visibility)
    far       = [(50,80)]                       single far-only band (focus 4: long-range)
    day       = same as overall, but evaluated on samples with is_night=False
    night     = same as overall, but evaluated on samples with is_night=True

Metrics: mae, rmse, rel, delta1, delta2, delta3.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


RANGE_BINS_OVERALL: List[Tuple[float, float]] = [(0, 50), (0, 70), (0, 80)]
RANGE_BINS_FINE: List[Tuple[float, float]] = [
    (0, 10), (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70), (70, 80),
]
RANGE_BINS_FAR: List[Tuple[float, float]] = [(50, 80)]
ALL_METRIC_KEYS = ("mae", "rmse", "rel", "delta1", "delta2", "delta3")


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Standard depth metrics over the masked pixels."""
    p = pred[mask]
    g = gt[mask]
    if len(g) == 0:
        return {k: float("nan") for k in ALL_METRIC_KEYS}
    p = np.clip(p, 1e-3, None)
    g = np.clip(g, 1e-3, None)
    mae = float(np.mean(np.abs(p - g)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rel = float(np.mean(np.abs(p - g) / g))
    thresh = np.maximum(p / g, g / p)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))
    return {"mae": mae, "rmse": rmse, "rel": rel,
            "delta1": delta1, "delta2": delta2, "delta3": delta3}


def compute_range_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    range_bins: List[Tuple[float, float]],
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics across multiple depth ranges."""
    pred = np.clip(pred, min_depth, max_depth)
    pred[np.isinf(pred)] = max_depth
    pred[np.isnan(pred)] = min_depth
    out: Dict[str, Dict[str, float]] = {}
    for d_min, d_max in range_bins:
        m = valid_mask & (gt >= d_min) & (gt < d_max)
        metrics = compute_depth_metrics(pred, gt, m)
        if metric_keys is not None:
            metrics = {k: v for k, v in metrics.items() if k in metric_keys}
        out[f"{d_min}-{d_max}m"] = metrics
    return out


class DepthEvaluator:
    """Per-sample evaluation + dataset-level aggregation across day/night/far/etc."""

    def __init__(
        self,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        range_bins_overall: Optional[List[Tuple[float, float]]] = None,
        range_bins_fine: Optional[List[Tuple[float, float]]] = None,
        range_bins_far: Optional[List[Tuple[float, float]]] = None,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.range_bins_overall = range_bins_overall or RANGE_BINS_OVERALL
        self.range_bins_fine = range_bins_fine or RANGE_BINS_FINE
        self.range_bins_far = range_bins_far or RANGE_BINS_FAR

    def evaluate_sample(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
        is_night: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        overall = compute_range_metrics(
            pred, gt, valid_mask, self.range_bins_overall,
            self.min_depth, self.max_depth,
        )
        per_range = compute_range_metrics(
            pred, gt, valid_mask, self.range_bins_fine,
            self.min_depth, self.max_depth,
            metric_keys=["mae", "rmse"],
        )
        far = compute_range_metrics(
            pred, gt, valid_mask, self.range_bins_far,
            self.min_depth, self.max_depth,
        )
        return {
            "overall": overall,
            "per_range": per_range,
            "far": far,
            "is_night": bool(is_night),
        }

    def aggregate_metrics(
        self,
        all_metrics: List[Dict],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate per-sample metrics into dataset-level means.

        Adds two extra categories `day` and `night` that are means computed
        over only the day/night subset of samples (using the per-sample
        `overall` range metrics).
        """
        if not all_metrics:
            return {}

        agg: Dict[str, Dict[str, Dict[str, float]]] = {}

        # categories that come from every sample
        for category in ("overall", "per_range", "far"):
            agg[category] = self._mean_category(all_metrics, category)

        # day / night subsets — copy `overall` from samples filtered by is_night
        day_samples = [m for m in all_metrics if not m.get("is_night", False)]
        night_samples = [m for m in all_metrics if m.get("is_night", False)]
        if day_samples:
            agg["day"] = self._mean_category(day_samples, "overall")
        if night_samples:
            agg["night"] = self._mean_category(night_samples, "overall")
        return agg

    @staticmethod
    def _mean_category(samples: List[Dict], category: str) -> Dict[str, Dict[str, float]]:
        if not samples or category not in samples[0]:
            return {}
        ranges = list(samples[0][category].keys())
        out: Dict[str, Dict[str, float]] = {}
        for rname in ranges:
            metric_names = list(samples[0][category][rname].keys())
            out[rname] = {}
            for mname in metric_names:
                vals = [s[category][rname][mname] for s in samples
                        if not np.isnan(s[category][rname][mname])]
                out[rname][mname] = float(np.mean(vals)) if vals else float("nan")
        return out
