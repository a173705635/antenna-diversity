function test_tracking()
%TEST_TRACKING Verify closed-loop DLL/FLL/PLL behavior after acquisition.

cfg = default_config();
fs = cfg.dfe.fsOutHz;
samplesPerMs = round(fs * 1e-3);
durationMs = 40;
n = 0:durationMs*samplesPerMs-1;
trueDelaySamples = 600;
trueDelayChips = trueDelaySamples * cfg.signal.codeRateHz / fs;
code = gnss.ca_code(cfg.signal.prn);
codePhase = mod(n * cfg.signal.codeRateHz / fs - trueDelayChips, 1023);
sampledCode = code(floor(codePhase) + 1);

rng(9, 'twister');
x = sampledCode .* exp(1j * (2*pi * cfg.signal.dopplerHz * n / fs + 0.4));
x = x + 0.15/sqrt(2) * (randn(size(x)) + 1j*randn(size(x)));

acq = receiver.acquire(x, fs, cfg.signal.prn, cfg.acquisition);
assert(acq.success, 'Tracking test acquisition failed.');
trk = receiver.track(x, fs, cfg.signal.prn, acq, cfg.tracking, ...
    cfg.signal.carrierHz, cfg.signal.codeRateHz);

assert(trk.success, 'Tracking did not produce any complete epochs.');
assert(numel(trk.prompt) >= 20, 'Tracking produced too few epochs.');
assert(all(isfinite(trk.carrierFrequencyHz)), ...
    'Carrier loop generated nonfinite frequency estimates.');
assert(all(isfinite(trk.codeRateHz)), ...
    'Code loop generated nonfinite rate estimates.');
assert(mean(abs(trk.prompt(end-9:end))) > 100, ...
    'Steady-state Prompt correlation is unexpectedly small.');
assert(abs(trk.carrierFrequencyHz(end) - cfg.signal.dopplerHz) <= ...
    cfg.tracking.maxFinalDopplerErrorHz, ...
    'Final Doppler estimate is outside tolerance.');
assert(sqrt(mean(trk.dllError(end-9:end).^2)) < 0.5, ...
    'Steady-state DLL error is unbounded.');
assert(sqrt(mean(trk.pllErrorRad(end-9:end).^2)) < 0.5, ...
    'Steady-state PLL error is unbounded.');

fprintf('PASS test_tracking\n');
end
