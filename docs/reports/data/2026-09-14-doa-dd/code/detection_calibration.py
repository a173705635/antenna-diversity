"""Record-level score calibration; independent of path truth and angle fitting."""

import math
import numpy as np
from scipy.stats import beta


def residual_peak_score_db(power):
    values=np.asarray(power,dtype=float)
    if values.size==0 or not np.all(np.isfinite(values)) or np.any(values<0):
        return None
    median=float(np.median(values))
    if median<=0:
        return None
    return float(10*np.log10(np.max(values)/median))


def calibrate_threshold(scores, target_false_alarm_rate):
    """Order statistic with strict score > threshold decisions.

    Rank ceil((n+1)*(1-alpha)) provides a marginal exceedance bound for
    exchangeable eligible calibration/test scores. It does not guarantee
    conditional performance in each scenario or confidence on one fitted cutoff.
    """
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("finite eligible calibration scores required")
    if not 0<target_false_alarm_rate<1:
        raise ValueError("false alarm target must be between zero and one")
    rank=math.ceil((values.size+1)*(1-target_false_alarm_rate))
    if rank>values.size:
        raise ValueError("too few calibration scores for the requested tail")
    return float(np.sort(values)[rank-1])


def binomial_interval(positives,total,confidence=.95):
    """Two-sided exact Clopper-Pearson interval, including endpoints."""
    if total<0 or not 0<=positives<=total or not 0<confidence<1:
        raise ValueError("valid counts and confidence required")
    if not total:
        return None
    tail=(1-confidence)/2
    low=0. if positives==0 else float(beta.ppf(tail,positives,total-positives+1))
    high=1. if positives==total else float(beta.ppf(1-tail,positives+1,total-positives))
    return [low,high]
