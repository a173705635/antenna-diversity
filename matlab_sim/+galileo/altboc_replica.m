function [waveform, components] = altboc_replica(codePhaseChips, cfg, knownData)
%ALTBOC_REPLICA Generate the complete Galileo E5 AltBOC(15,10) replica.

if ~isvector(codePhaseChips) || isempty(codePhaseChips) || ...
        any(~isfinite(codePhaseChips))
    error('galileo:altboc_replica:InvalidCodePhase', ...
        'codePhaseChips must be a nonempty finite vector.');
end
requiredSignalFields = {'svid', 'primaryLengthChips', 'codeRateHz', ...
    'subcarrierRateHz'};
for k = 1:numel(requiredSignalFields)
    if ~isfield(cfg, 'signal') || ...
            ~isfield(cfg.signal, requiredSignalFields{k})
        error('galileo:altboc_replica:MissingConfig', ...
            'Missing signal configuration field: %s.', ...
            requiredSignalFields{k});
    end
end
if ~isstruct(knownData) || ~isfield(knownData, 'e5aI') || ...
        ~isfield(knownData, 'e5bI')
    error('galileo:altboc_replica:MissingKnownData', ...
        'knownData must contain e5aI and e5bI symbol vectors.');
end
if any(codePhaseChips < 0)
    error('galileo:altboc_replica:NegativeCodeTime', ...
        'Absolute code phase must be nonnegative.');
end

inputSize = size(codePhaseChips);
phase = double(codePhaseChips(:).');
dataE5a = validate_data(knownData.e5aI, 'e5aI');
dataE5b = validate_data(knownData.e5bI, 'e5bI');

persistent primary secondary
if isempty(primary)
    names = {'e5aI', 'e5aQ', 'e5bI', 'e5bQ'};
    for k = 1:numel(names)
        name = names{k};
        primary.(name) = galileo.e5_primary_code(name, cfg.signal.svid);
        secondary.(name) = galileo.e5_secondary_code(name, cfg.signal.svid);
    end
end

codeLength = cfg.signal.primaryLengthChips;
chip = floor(phase);
primaryIndex = mod(chip, codeLength) + 1;
primaryEpoch = floor(phase / codeLength);

dataIndexE5a = floor(primaryEpoch / 20) + 1;
dataIndexE5b = floor(primaryEpoch / 4) + 1;
if max(dataIndexE5a) > numel(dataE5a) || ...
        max(dataIndexE5b) > numel(dataE5b)
    error('galileo:altboc_replica:InsufficientKnownData', ...
        'Known data does not span the requested absolute code phase.');
end

e5aI = sample_component(primary.e5aI, secondary.e5aI, ...
    primaryIndex, primaryEpoch) .* dataE5a(dataIndexE5a);
e5aQ = sample_component(primary.e5aQ, secondary.e5aQ, ...
    primaryIndex, primaryEpoch);
e5bI = sample_component(primary.e5bI, secondary.e5bI, ...
    primaryIndex, primaryEpoch) .* dataE5b(dataIndexE5b);
e5bQ = sample_component(primary.e5bQ, secondary.e5bQ, ...
    primaryIndex, primaryEpoch);

subcarrierRatio = cfg.signal.subcarrierRateHz / cfg.signal.codeRateHz;
subcarrierSubperiod = floor(mod(phase * subcarrierRatio, 1) * 8);
column = 1 + 8*(e5aI > 0) + 4*(e5bI > 0) + ...
    2*(e5aQ > 0) + (e5bQ > 0);
phaseLut = [ ...
    5 4 4 3 6 3 1 2 6 5 7 2 7 8 8 1; ...
    5 4 8 3 2 3 1 2 6 5 7 6 7 4 8 1; ...
    1 4 8 7 2 3 1 2 6 5 7 6 3 4 8 5; ...
    1 8 8 7 2 3 1 6 2 5 7 6 3 4 4 5; ...
    1 8 8 7 2 7 5 6 2 1 3 6 3 4 4 5; ...
    1 8 4 7 6 7 5 6 2 1 3 2 3 8 4 5; ...
    5 8 4 3 6 7 5 6 2 1 3 2 7 8 4 1; ...
    5 4 4 3 6 7 5 2 6 1 3 2 7 8 8 1];
k = phaseLut(sub2ind(size(phaseLut), ...
    subcarrierSubperiod + 1, column));
waveform = reshape(exp(1j * pi/4 * k), inputSize);

if nargout > 1
    components.e5aI = reshape(e5aI, inputSize);
    components.e5aQ = reshape(e5aQ, inputSize);
    components.e5bI = reshape(e5bI, inputSize);
    components.e5bQ = reshape(e5bQ, inputSize);
    components.primaryEpoch = reshape(primaryEpoch, inputSize);
    components.subcarrierSubperiod = reshape(subcarrierSubperiod, inputSize);
end
end

function sampled = sample_component(primary, secondary, indices, epochs)
secondaryIndex = mod(epochs, numel(secondary)) + 1;
sampled = primary(indices) .* secondary(secondaryIndex);
end

function data = validate_data(data, fieldName)
if ~isvector(data) || isempty(data) || ...
        any(~isfinite(data)) || any(data ~= 1 & data ~= -1)
    error('galileo:altboc_replica:InvalidKnownData', ...
        'knownData.%s must be a nonempty +1/-1 vector.', fieldName);
end
data = double(data(:).');
end
