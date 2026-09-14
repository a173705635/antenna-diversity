"""Per-satellite streaming lifecycle for conditional GF quality decisions.

External tokens identify the complete four-stream arc and calibration context.
This module invalidates stale approvals; it cannot establish their correctness.
The multiplier applies on top of base PVT weights, never replacing base checks.
"""
from dataclasses import dataclass
import math
from gf_quality_gate import classify


@dataclass
class GFRuntime:
    max_gap_s: float
    anomaly_multiplier: float
    last_time_s: float | None = None
    context: tuple | None = None
    approved: bool = False

    def __post_init__(self):
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError('positive maximum gap required')
        if not math.isfinite(self.anomaly_multiplier) or not 0 < self.anomaly_multiplier <= 1:
            raise ValueError('anomaly multiplier must be in (0, 1]')

    def step(self, time_s, residual_m, threshold_m, *, valid, arc_id,
             calibration_id, calibration_valid, nuisance_bound_m,
             branch_confirmation=None):
        """Confirmation is (arc_id, calibration_id), issued freshly by caller.

        IDs must change on any constituent arc/reset or calibration context
        change, including receiver restart. Valid calibration includes its
        temperature/gain/frequency/age envelope. No automatic reapproval occurs.
        """
        if type(valid) is not bool or type(calibration_valid) is not bool:
            raise ValueError('boolean validity flags required')
        if not math.isfinite(time_s) or (self.last_time_s is not None and time_s <= self.last_time_s):
            self.approved = False
            raise ValueError('strictly increasing finite time required')
        if not math.isfinite(threshold_m) or threshold_m <= 0:
            self.approved = False
            raise ValueError('positive threshold required')
        gap = self.last_time_s is not None and time_s-self.last_time_s > self.max_gap_s
        self.last_time_s = time_s
        context = (arc_id, calibration_id)
        changed = context != self.context
        self.context = context
        ready = (valid and calibration_valid and isinstance(arc_id,str) and bool(arc_id)
                 and isinstance(calibration_id,str) and bool(calibration_id)
                 and math.isfinite(residual_m) and math.isfinite(nuisance_bound_m)
                 and nuisance_bound_m >= 0)
        if changed or gap or not ready:
            self.approved = False
        # After a gap, require a later fresh event; do not accept a potentially stale event on that epoch.
        if ready and not gap and branch_confirmation == context:
            self.approved = True
        result = classify([[residual_m]],threshold_m,nuisance_bound_m=nuisance_bound_m,
                          valid=ready,branch_verified=self.approved)
        state = int(result['state'][0,0])
        reason = str(result['reason'][0,0])
        if not ready: reason = 'invalid_observation_or_calibration'
        elif gap: reason = 'time_gap_requires_confirmation'
        elif not self.approved: reason = 'await_current_branch_confirmation'
        return {'state':state,'reason':reason,
                'base_weight_multiplier': self.anomaly_multiplier if state==1 else 1.,
                'branch_approved':self.approved}
