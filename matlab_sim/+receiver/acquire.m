function acq = acquire(x, fsHz, prn, cfg)
%ACQUIRE Perform noncoherent FFT parallel-code-phase acquisition.

if ~isvector(x) || isempty(x)
    error('receiver:acquire:InvalidInput', 'x must be a nonempty vector.');
end
if ~isscalar(fsHz) || fsHz <= 0 || ~isfinite(fsHz)
    error('receiver:acquire:InvalidSampleRate', ...
        'fsHz must be a positive finite scalar.');
end
required = {'dopplerBinsHz', 'coherentMs', 'noncoherentMs', ...
    'exclusionChips', 'threshold', 'codeRateHz'};
for k = 1:numel(required)
    if ~isfield(cfg, required{k})
        error('receiver:acquire:MissingConfig', ...
            'Missing acquisition configuration field: %s.', required{k});
    end
end

x = x(:).';
samplesPerMs = fsHz * 1e-3;
if abs(samplesPerMs - round(samplesPerMs)) > 1e-9
    error('receiver:acquire:NonintegerMillisecond', ...
        'The sample rate must produce an integer number of samples per millisecond.');
end
samplesPerMs = round(samplesPerMs);
coherentSamples = samplesPerMs * cfg.coherentMs;
requiredSamples = coherentSamples * cfg.noncoherentMs;
if numel(x) < requiredSamples
    error('receiver:acquire:InsufficientSamples', ...
        'Acquisition needs %d samples but received %d.', requiredSamples, numel(x));
end

code = gnss.ca_code(prn);
n = 0:coherentSamples-1;
codePhase = mod(n * cfg.codeRateHz / fsHz, 1023);
localCode = code(floor(codePhase) + 1);
localCodeSpectrumConjugate = conj(fft(localCode));
dopplerBins = cfg.dopplerBinsHz(:);
searchPower = zeros(numel(dopplerBins), coherentSamples);

for binIndex = 1:numel(dopplerBins)
    carrierWipeoff = exp(-1j * 2*pi * dopplerBins(binIndex) * n / fsHz);
    accumulatedPower = zeros(1, coherentSamples);
    for blockIndex = 1:cfg.noncoherentMs
        first = (blockIndex - 1) * coherentSamples + 1;
        block = x(first:first + coherentSamples - 1);
        correlation = ifft(fft(block .* carrierWipeoff) .* ...
            localCodeSpectrumConjugate);
        accumulatedPower = accumulatedPower + abs(correlation).^2;
    end
    searchPower(binIndex, :) = accumulatedPower;
end

[peak, linearIndex] = max(searchPower(:));
[bestBinIndex, bestCodeIndex] = ind2sub(size(searchPower), linearIndex);
bestRow = searchPower(bestBinIndex, :);
indices = 1:coherentSamples;
directDistance = abs(indices - bestCodeIndex);
circularDistance = min(directDistance, coherentSamples - directDistance);
exclusionSamples = max(1, round(cfg.exclusionChips * fsHz / cfg.codeRateHz));
bestRow(circularDistance <= exclusionSamples) = -inf;
[secondPeak, secondPeakIndex] = max(bestRow);
if ~isfinite(secondPeak) || secondPeak <= 0
    secondPeak = eps;
    secondPeakIndex = NaN;
end
metric = peak / secondPeak;

acq.success = metric >= cfg.threshold;
acq.prn = prn;
acq.dopplerHz = dopplerBins(bestBinIndex);
acq.codePhaseSamples = bestCodeIndex - 1;
acq.peak = peak;
acq.secondPeak = secondPeak;
acq.secondPeakCodePhaseSamples = secondPeakIndex - 1;
acq.exclusionSamples = exclusionSamples;
acq.metric = metric;
acq.dopplerBinsHz = dopplerBins;
acq.searchPower = searchPower;
acq.samplesPerMs = samplesPerMs;
acq.coherentSamples = coherentSamples;
end
