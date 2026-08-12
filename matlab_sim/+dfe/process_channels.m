function [xOut, info] = process_channels(adcCodes, cfg)
%PROCESS_CHANNELS Apply common-NCO DDC, low-pass filtering, and decimation.

if ~ismatrix(adcCodes) || isempty(adcCodes)
    error('dfe:process_channels:InvalidInput', ...
        'adcCodes must be a nonempty channels-by-samples matrix.');
end
required = {'fsInHz', 'ifHz', 'decimation', 'cutoffHz', 'numTaps', ...
    'outputBits', 'outputBinaryShift'};
for k = 1:numel(required)
    if ~isfield(cfg, required{k})
        error('dfe:process_channels:MissingConfig', ...
            'Missing DFE configuration field: %s.', required{k});
    end
end

[h, filterInfo] = dfe.design_decimator(cfg.fsInHz, cfg.decimation, ...
    cfg.cutoffHz, cfg.numTaps);
nSamples = size(adcCodes, 2);
n = 0:nSamples-1;
nco = exp(-1j * 2*pi * cfg.ifHz * n / cfg.fsInHz);
mixed = double(adcCodes) .* nco;

filtered = complex(zeros(size(mixed)));
for channel = 1:size(mixed, 1)
    filtered(channel, :) = filter(h, 1, mixed(channel, :));
end

firstValid = filterInfo.groupDelayInputSamples + 1;
if firstValid > nSamples
    error('dfe:process_channels:InsufficientSamples', ...
        'Input is shorter than the FIR group delay.');
end
aligned = filtered(:, firstValid:end);
xOut = aligned(:, 1:cfg.decimation:end);

if cfg.outputBits > 0
    if cfg.outputBits < 2 || cfg.outputBits ~= fix(cfg.outputBits) || ...
            cfg.outputBinaryShift < 0 || cfg.outputBinaryShift ~= fix(cfg.outputBinaryShift)
        error('dfe:process_channels:InvalidQuantizer', ...
            'Output bits and binary shift must be valid nonnegative integers.');
    end
    scale = 2^cfg.outputBinaryShift;
    minCode = -2^(cfg.outputBits - 1);
    maxCode = 2^(cfg.outputBits - 1) - 1;
    iCode = min(max(round(real(xOut) / scale), minCode), maxCode);
    qCode = min(max(round(imag(xOut) / scale), minCode), maxCode);
    xOut = complex(iCode, qCode);
else
    scale = 1;
end

info = filterInfo;
info.ifHz = cfg.ifHz;
info.outputBits = cfg.outputBits;
info.outputScale = scale;
info.outputLength = size(xOut, 2);
info.filterCoefficients = h;
end
