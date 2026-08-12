function test_e5_signal()
%TEST_E5_SIGNAL Verify deterministic full-band complex low-IF generation.

cfg = e5_default_config();
cfg.signal.durationMs = 2;
cfg.signal.cn0DbHz = inf;
[x1, truth1, data1] = galileo.generate_e5_signal(cfg);
[x2, ~, data2] = galileo.generate_e5_signal(cfg);

assert(numel(x1) == 2 * cfg.signal.samplesPerEpoch);
assert(isequal(x1, x2));
assert(isequal(data1, data2));
assert(abs(truth1.e5aCenterHz + 11.550e6) < 1e-9);
assert(abs(truth1.e5bCenterHz - 19.140e6) < 1e-9);

z = x1 .* conj(truth1.basebandReplica);
frequencyHz = angle(sum(z(2:end) .* conj(z(1:end-1)))) * ...
    cfg.signal.fsHz / (2*pi);
assert(abs(frequencyHz - ...
    (cfg.signal.ifHz + cfg.signal.dopplerHz)) < 1e-6);
assert(all(abs(abs(x1) - 1) < 1e-12));

nFft = numel(x1);
spectrumPower = abs(fftshift(fft(x1))).^2;
frequencyAxisHz = (-nFft/2:nFft/2-1) * cfg.signal.fsHz / nFft;
halfSidebandHz = 10.23e6;
e5aMask = abs(frequencyAxisHz - truth1.e5aCenterHz) <= ...
    halfSidebandHz;
e5bMask = abs(frequencyAxisHz - truth1.e5bCenterHz) <= ...
    halfSidebandHz;
e5aPowerRatio = sum(spectrumPower(e5aMask)) / sum(spectrumPower);
e5bPowerRatio = sum(spectrumPower(e5bMask)) / sum(spectrumPower);
assert(e5aPowerRatio > 0.30 && e5bPowerRatio > 0.30);
assert(abs(e5aPowerRatio - e5bPowerRatio) < 0.02);

cfg.signal.cn0DbHz = 48;
[xn1, tn1] = galileo.generate_e5_signal(cfg);
[xn2, tn2] = galileo.generate_e5_signal(cfg);
assert(isequal(xn1, xn2));
assert(tn1.noiseVariance > 0 && ...
    tn1.noiseVariance == tn2.noiseVariance);

fprintf('PASS test_e5_signal\n');
end
