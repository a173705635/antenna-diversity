"""Leave-one-out solution consistency; not a certified protection level."""
import numpy as np
from signal_recovery import position_solution


def separation_scores(satellites,ranges,selected,positions):
    scores=np.full(len(ranges),np.inf)
    for k,(s,r,mask,p) in enumerate(zip(satellites,ranges,selected,positions)):
        if mask.sum()<6 or not np.isfinite(p).all():continue
        differences=[]
        for star in np.flatnonzero(mask):
            subset=mask.copy();subset[star]=False
            other,_,_=position_solution(s,r,subset)
            differences.append(np.linalg.norm((other-p)[:2]))
        # Unsolvable subset means the solution has not passed this consistency check.
        scores[k]=np.max(differences) if np.isfinite(differences).all() else np.inf
    return scores
