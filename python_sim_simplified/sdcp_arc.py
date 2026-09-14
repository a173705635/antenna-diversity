"""SDCP reference lifecycle, not a phase-branch or signal authenticity verifier."""
from dataclasses import dataclass, field
import math


@dataclass
class SDCPArc:
    session_id: str
    max_gap_s: float
    last_time_s: float | None = None
    arc_number: int = 0
    arc_id: str | None = None
    context: tuple | None = None
    reference_id: str | None = None
    evidence_id: str | None = None
    used_evidence: set = field(default_factory=set)
    break_reason: str = 'startup'

    def invalidate(self,reason):
        self.arc_id=None;self.reference_id=None;self.evidence_id=None
        self.break_reason=reason

    def step(self,time_s,*,quality_valid,phase_state_valid,tracking_ids,calibration_id):
        """One satellite/frequency pair. IDs come from receiver tracking instances.

        phase_state_valid must include external half-cycle/reset validity.
        This does not discover invisible integer slips or identify a true signal.
        """
        if (not math.isfinite(time_s)
                or (self.last_time_s is not None and time_s<=self.last_time_s)):
            self.invalidate('invalid_input')
            raise ValueError('increasing finite time required for the current arc')
        gap=self.last_time_s is not None and time_s-self.last_time_s>self.max_gap_s
        self.last_time_s=time_s
        context=(tracking_ids,calibration_id)
        if context!=self.context:
            self.invalidate('tracking_or_calibration_changed');self.context=context
        if gap:self.invalidate('time_gap')
        if not quality_valid:self.invalidate('quality_invalid')
        if not phase_state_valid:self.invalidate('phase_state_unverified')
        usable=quality_valid and phase_state_valid
        if usable and self.arc_id is None:
            self.arc_number+=1;self.arc_id=f'{self.session_id}:{self.arc_number}'
        state='unavailable' if not usable else ('reference_usable' if self.reference_id else 'await_reference')
        return {'state':state,'arc_id':self.arc_id,'reference_id':self.reference_id,
                'evidence_id':self.evidence_id,'last_break_reason':self.break_reason}

    def accept_reference(self,arc_id,reference_id,evidence_id):
        """Accept caller-validated NEW reference for current arc, never bridge arcs.

        Caller owns evidence correctness; inventing a fresh ID is not validation.
        The session ID must change on receiver restart. Persist evidence elsewhere
        if replay protection must survive a process restart. No offset is fitted.
        """
        if (self.arc_id is None or arc_id!=self.arc_id
                or not reference_id
                or not evidence_id
                or evidence_id in self.used_evidence):
            raise ValueError('current arc and fresh external reference evidence required')
        self.reference_id=reference_id;self.evidence_id=evidence_id
        self.used_evidence.add(evidence_id)
