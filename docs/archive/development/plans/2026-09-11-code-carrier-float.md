# Code and carrier joint float positioning

User approved single-antenna GPS L1 code/carrier joint estimation with float ambiguities, no integer fixing. Continue ponytail: reuse archived RF histories, existing NumPy/SciPy and the SPP initializer; no extra receiver simulation or filter framework.

- [x] Add common-time carrier range conversion in `positioning.py`: negative accumulated NCO phase times wavelength/(2*pi), using each block's pre-update frequency. Keep unknown phase origins as float biases.
- [x] Test noiseless recovery with arbitrary noninteger per-satellite biases and moving receiver; phase-origin changes must leave positions unchanged. Reject missing/degenerate observations instead of silently spanning gaps.
- [x] Implement offline weighted nonlinear least squares: independent position/clock per epoch, shared per-satellite float biases over one continuous arc. Whiten code/phase residuals with explicit fixed noise scales; use SciPy QR least squares, not normal equations. Center carrier ranges using observed code only to avoid large bias cancellation. No true position or true ambiguity input.
- [x] Add `audit_code_carrier_float.py` to read archived single-antenna SPP records. Split at invalid common epochs; do not interpolate across gaps. Report raw SPP versus joint position mean bias, demeaned 3D RMS, absolute 3D RMS and carrier residuals. Reject arcs with excessive postfit carrier residual, without claiming a complete cycle-slip detector.
- [x] Run three archived 1 s records; perturb phase origins and per-satellite code biases as controls. Demonstrate absolute-bias limitations of fixed LOS geometry. Save reproducible data, source/input hashes, plots and report; run targeted and full simplified tests.

Default weighting is an engineering assumption (code 30 m, carrier 0.01 m), not a calibrated stochastic model; do not infer confidence intervals or integer-fixing validity. Output is retrospective over each complete arc, not causal real-time output. Keep old SPP outputs unchanged.
