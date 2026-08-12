function test_acquisition()
%TEST_ACQUISITION Verify FFT code-phase and Doppler acquisition.

cfg = default_config();
fs = cfg.dfe.fsOutHz;
samplesPerMs = round(fs * 1e-3);
durationMs = cfg.acquisition.coherentMs * cfg.acquisition.noncoherentMs;
n = 0:durationMs*samplesPerMs-1;
trueDelaySamples = 600;
trueDelayChips = trueDelaySamples * cfg.signal.codeRateHz / fs;
code = gnss.ca_code(cfg.signal.prn);
codePhase = mod(n * cfg.signal.codeRateHz / fs - trueDelayChips, 1023);
sampledCode = code(floor(codePhase) + 1);

rng(7, 'twister');
x = sampledCode .* exp(1j * 2*pi * cfg.signal.dopplerHz * n / fs);
x = x + 0.05/sqrt(2) * (randn(size(x)) + 1j*randn(size(x)));

acq = receiver.acquire(x, fs, cfg.signal.prn, cfg.acquisition);
assert(acq.success, 'High-SNR signal was not acquired.');
assert(isfield(acq, 'secondPeakCodePhaseSamples'), ...
    'Acquisition result is missing the second-peak code phase.');
assert(isfield(acq, 'exclusionSamples'), ...
    'Acquisition result is missing the peak exclusion width.');
binSpacing = abs(cfg.acquisition.dopplerBinsHz(2) - ...
    cfg.acquisition.dopplerBinsHz(1));
assert(abs(acq.dopplerHz - cfg.signal.dopplerHz) <= binSpacing, ...
    'Acquisition Doppler error exceeds one bin.');
phaseError = abs(acq.codePhaseSamples - trueDelaySamples);
phaseError = min(phaseError, samplesPerMs - phaseError);
assert(phaseError <= 1, 'Acquisition code-phase error exceeds one sample.');

secondPeakDistance = abs(acq.secondPeakCodePhaseSamples - ...
    acq.codePhaseSamples);
secondPeakDistance = min(secondPeakDistance, ...
    acq.coherentSamples - secondPeakDistance);
assert(secondPeakDistance > acq.exclusionSamples, ...
    'Second peak lies inside the primary-peak exclusion interval.');

fprintf('PASS test_acquisition\n');
end
