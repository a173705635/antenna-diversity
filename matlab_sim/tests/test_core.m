function test_core()
%TEST_CORE Verify the default configuration, C/A code, and array geometry.

cfg = default_config();

code = gnss.ca_code(1);
assert(isequal(size(code), [1, 1023]), 'C/A code must be a 1x1023 row vector.');
assert(all(code == 1 | code == -1), 'C/A code must contain only +/-1.');

r = ifft(abs(fft(code)).^2);
assert(abs(real(r(1)) - 1023) < 1e-9, 'C/A autocorrelation peak is incorrect.');
outOfPhaseValues = sort(unique(round(real(r(2:end)))).');
assert(isequal(outOfPhaseValues, [-65; -1; 63]), ...
    'C/A Gold-code autocorrelation values are incorrect.');

a = arraysim.steering_vector([0; cfg.array.spacingM], 30, ...
    cfg.signal.wavelengthM);
expected = exp(-1j * 2*pi * cfg.array.spacingM * sind(30) / ...
    cfg.signal.wavelengthM);
assert(abs(a(1) - 1) < 1e-12, 'Reference element response must equal one.');
assert(abs(a(2) - expected) < 1e-12, 'Array phase convention is incorrect.');

fprintf('PASS test_core\n');
end
