function [h, info] = design_decimator(fsInHz, decimation, cutoffHz, numTaps)
%DESIGN_DECIMATOR Design a toolbox-free windowed-sinc decimation filter.

if ~isscalar(fsInHz) || fsInHz <= 0 || ~isfinite(fsInHz)
    error('dfe:design_decimator:InvalidSampleRate', ...
        'fsInHz must be a positive finite scalar.');
end
if ~isscalar(decimation) || decimation < 1 || decimation ~= fix(decimation)
    error('dfe:design_decimator:InvalidDecimation', ...
        'decimation must be a positive integer.');
end
fsOutHz = fsInHz / decimation;
if ~isscalar(cutoffHz) || cutoffHz <= 0 || cutoffHz >= fsOutHz / 2
    error('dfe:design_decimator:InvalidCutoff', ...
        'cutoffHz must lie between zero and the output Nyquist frequency.');
end
if ~isscalar(numTaps) || numTaps < 3 || numTaps ~= fix(numTaps) || ...
        mod(numTaps, 2) == 0
    error('dfe:design_decimator:InvalidTapCount', ...
        'numTaps must be an odd integer of at least three.');
end

groupDelay = (numTaps - 1) / 2;
m = -groupDelay:groupDelay;
u = 2 * cutoffHz / fsInHz * m;
normalizedSinc = ones(size(u));
nonzero = u ~= 0;
normalizedSinc(nonzero) = sin(pi * u(nonzero)) ./ (pi * u(nonzero));
h = (2 * cutoffHz / fsInHz) * normalizedSinc;
window = 0.54 - 0.46 * cos(2*pi * (0:numTaps-1) / (numTaps-1));
h = h .* window;
h = h / sum(h);

info.fsInHz = fsInHz;
info.fsOutHz = fsOutHz;
info.decimation = decimation;
info.cutoffHz = cutoffHz;
info.numTaps = numTaps;
info.groupDelayInputSamples = groupDelay;
end
