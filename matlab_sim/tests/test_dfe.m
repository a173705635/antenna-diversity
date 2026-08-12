function test_dfe()
%TEST_DFE Verify DDC, FIR delay handling, decimation, and channel coherence.

cfg = default_config();
cfg.dfe.outputBits = 0;
durationSec = 2e-3;
nSamples = round(cfg.dfe.fsInHz * durationSec);
n = 0:nSamples-1;
tone = 100 * exp(1j * 2*pi * cfg.dfe.ifHz * n / cfg.dfe.fsInHz);
x = [tone; tone * exp(1j * 0.37)];

[y, info] = dfe.process_channels(x, cfg.dfe);

expectedLength = floor((nSamples - info.groupDelayInputSamples - 1) / ...
    cfg.dfe.decimation) + 1;
assert(size(y, 1) == 2, 'DFE changed the channel count.');
assert(size(y, 2) == expectedLength, 'DFE output length is incorrect.');
assert(abs(info.fsOutHz - 4.092e6) < 1e-9, 'DFE output rate is incorrect.');

phaseRatio = mean(y(2,:) ./ y(1,:));
assert(abs(phaseRatio - exp(1j * 0.37)) < 1e-10, ...
    'DFE did not preserve the inter-channel phase.');

spectrum = abs(fftshift(fft(y(1,:))));
frequency = (-floor(size(y,2)/2):ceil(size(y,2)/2)-1) * ...
    info.fsOutHz / size(y,2);
[~, peakIndex] = max(spectrum);
assert(abs(frequency(peakIndex)) <= info.fsOutHz / size(y,2), ...
    'DDC did not move the IF tone to zero frequency.');

fprintf('PASS test_dfe\n');
end
