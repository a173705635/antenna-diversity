"""Direct legacy RF replay adapter; no input-validation or admission layer.

Caller supplies aligned records in one frequency and explicit channel order.
Unknown hardware timing, calibration and phase-arc evidence stay unknown.
"""
import numpy as np
from spatial_spoofing import observations_from_records


def adapt_legacy_rf(records_by_channel, channel_ids, *, run_id, wavelength_m):
    """Reuse carrier reconstruction and quality masks without truth/PVT input."""
    records = [records_by_channel[c] for c in channel_ids]
    t, x, v = observations_from_records(records, wavelength_m)
    return dict(run_id=run_id, channel_ids=tuple(channel_ids),
                prns=[h['prn'] for h in records[0]['satellite_tracking']],
                sample_time_s=t.copy(), available_time_s=None, clock_id=None,
                time_basis='legacy_record_seconds', carrier_m=x, valid=v,
                invalid_reason=np.where(v, '', 'legacy_quality_invalid'),
                wavelength_m=wavelength_m, signal='GPS_L1_CA',
                calibration_id=None, arc_ids=None,
                evidence_gaps=['clock_id', 'available_time_s', 'calibration_id',
                               'arc_ids', 'imu', 'hardware_channel_mapping'],
                authenticity_verified=False, provenance='legacy_run_gps_spp_replay')
