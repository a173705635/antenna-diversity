function test_diversity()
%TEST_DIVERSITY Verify target normalization and spatial interference rejection.

cfg = default_config();
cfg.diversity.diagonalLoadingFactor = 1e-4;
a = arraysim.steering_vector(cfg.array.positionsM, ...
    cfg.array.desiredAngleDeg, cfg.signal.wavelengthM);
b = arraysim.steering_vector(cfg.array.positionsM, ...
    cfg.jammer.angleDeg, cfg.signal.wavelengthM);

rng(11, 'twister');
n = 20000;
s = (randn(1,n) + 1j*randn(1,n)) / sqrt(2);
j = (randn(1,n) + 1j*randn(1,n)) / sqrt(2);
X = a * s + 20 * b * j;

[~, singleInfo] = diversity.combine(X, a, 'single', cfg.diversity);
[~, egcInfo] = diversity.combine(X, a, 'egc', cfg.diversity);
[~, mvdrInfo] = diversity.combine(X, a, 'mvdr', cfg.diversity);

assert(abs(egcInfo.desiredResponse - 1) < 1e-10, ...
    'EGC target response is not unity.');
assert(abs(mvdrInfo.desiredResponse - 1) < 1e-10, ...
    'MVDR target response is not unity.');
singleJammerGain = abs(singleInfo.weights' * b);
mvdrJammerGain = abs(mvdrInfo.weights' * b);
assert(mvdrJammerGain < 0.1 * singleJammerGain, ...
    'MVDR did not suppress the deterministic interference direction.');

fprintf('PASS test_diversity\n');
end
