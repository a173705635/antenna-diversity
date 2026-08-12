function test_e5_tracking()
%TEST_E5_TRACKING Verify direct full-band joint tracking convergence.

cfg = e5_default_config();
cfg.signal.cn0DbHz = 55;
cfg.plot.enable = false;
[x, truth, knownData] = galileo.generate_e5_signal(cfg);
init.dopplerHz = truth.dopplerHz + ...
    cfg.initialization.dopplerErrorHz;
init.carrierPhaseRad = truth.initialCarrierPhaseRad + ...
    cfg.initialization.carrierPhaseErrorRad;
init.codePhaseChips = truth.initialCodePhaseChips + ...
    cfg.initialization.codePhaseErrorChips;

trk = galileo.track_e5_fullband(x, cfg.signal.fsHz, ...
    cfg, init, knownData);
assert(trk.success);
assert(numel(trk.prompt) >= 20);
assert(all(isfinite(trk.dopplerHz)));
assert(all(isfinite(trk.codePhaseChips)));
assert(mean(abs(trk.prompt(end-9:end))) > 0.25);
assert(abs(mean(trk.dopplerHz(end-9:end)) - truth.dopplerHz) < 20);

epochTimeSec = trk.epochStartSamples(end-9:end) / cfg.signal.fsHz;
trueCode = truth.initialCodePhaseChips + epochTimeSec * truth.codeRateHz;
codeError = wrap_code(trk.codePhaseChips(end-9:end) - ...
    trueCode, cfg.signal.primaryLengthChips);
assert(abs(mean(codeError)) < 0.03);
assert(sqrt(mean(trk.pllErrorRad(end-9:end).^2)) < 0.25);

assert_error(@() galileo.track_e5_fullband([], cfg.signal.fsHz, ...
    cfg, init, knownData), 'galileo:track_e5_fullband:InvalidInput');

sideCfg = e5_default_config();
sideCfg.signal.durationMs = 12;
sideCfg.signal.cn0DbHz = inf;
sideCfg.tracking.maxEpochs = 12;
sideCfg.tracking.sidePeakRatio = 1.01;
[sideX, sideTruth, sideData] = galileo.generate_e5_signal(sideCfg);
sideInit.dopplerHz = sideTruth.dopplerHz;
sideInit.carrierPhaseRad = sideTruth.initialCarrierPhaseRad;
sideInit.codePhaseChips = sideTruth.initialCodePhaseChips + ...
    sideCfg.tracking.sidePeakOffsetChips + 0.02;
sideTrk = galileo.track_e5_fullband(sideX, sideCfg.signal.fsHz, ...
    sideCfg, sideInit, sideData);
assert(any(sideTrk.sidePeakJumpChips ~= 0));
sideTimeSec = sideTrk.epochStartSamples(end) / sideCfg.signal.fsHz;
sideTrueCode = sideTruth.initialCodePhaseChips + ...
    sideTimeSec * sideTruth.codeRateHz;
sideCodeError = wrap_code(sideTrk.codePhaseChips(end) - ...
    sideTrueCode, sideCfg.signal.primaryLengthChips);
assert(abs(sideCodeError) < 0.05);

fprintf('PASS test_e5_tracking\n');
end

function value = wrap_code(value, period)
value = mod(value + period/2, period) - period/2;
end

function assert_error(action, expectedIdentifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier, expectedIdentifier));
    return;
end
error('test_e5_tracking:MissingError', ...
    'Expected error %s was not raised.', expectedIdentifier);
end
