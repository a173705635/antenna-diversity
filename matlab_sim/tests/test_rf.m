function test_rf()
%TEST_RF Verify spatial signal generation and the RF/ADC model.

cfg = default_config();
cfg.jammer.enable = false;
[x, truth] = gnss.generate_scenario(cfg);

expectedCodeRateHz = cfg.signal.codeRateHz * ...
    (1 + cfg.signal.dopplerHz / cfg.signal.carrierHz);
assert(isfield(truth, 'codeRateHz') && ...
    abs(truth.codeRateHz - expectedCodeRateHz) < 1e-9, ...
    'Scenario code Doppler is inconsistent with carrier Doppler.');

ratio = mean(x(2,:) ./ x(1,:));
expectedRatio = truth.desiredSteeringEffective(2) / ...
    truth.desiredSteeringEffective(1);
assert(abs(ratio - expectedRatio) < 1e-10, ...
    'Desired signal was not projected through the array correctly.');
assert(isequal(size(x), [cfg.array.numElements, ...
    round(cfg.signal.fsRfHz * cfg.signal.durationMs * 1e-3)]), ...
    'Scenario output dimensions are incorrect.');

stage = struct('name', 'OnlyStage', 'gainDb', 20, 'noiseFigureDb', 2);
m = rf.cascade_metrics(stage);
assert(abs(m.totalGainDb - 20) < 1e-12, 'Single-stage gain is incorrect.');
assert(abs(m.noiseFigureDb - 2) < 1e-12, ...
    'Single-stage noise figure is incorrect.');

[adc1, info1] = rf.process_frontend(x, cfg.rf, cfg.randomSeed);
[adc2, info2] = rf.process_frontend(x, cfg.rf, cfg.randomSeed);
assert(isequal(adc1, adc2), 'RF output is not repeatable for a fixed seed.');
assert(isequal(info1.saturationFraction, info2.saturationFraction), ...
    'RF metrics are not repeatable for a fixed seed.');
assert(isequal(size(adc1), size(x)), 'RF processing changed array dimensions.');
assert(all(real(adc1(:)) >= -512 & real(adc1(:)) <= 511), ...
    'ADC I codes exceeded the signed 10-bit range.');
assert(all(imag(adc1(:)) >= -512 & imag(adc1(:)) <= 511), ...
    'ADC Q codes exceeded the signed 10-bit range.');
assert(all(real(adc1(:)) == round(real(adc1(:)))) && ...
       all(imag(adc1(:)) == round(imag(adc1(:)))), ...
    'ADC output must contain integer-valued I/Q codes.');

fprintf('PASS test_rf\n');
end
