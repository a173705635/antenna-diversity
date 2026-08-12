function metrics = cascade_metrics(stages)
%CASCADE_METRICS Compute total gain and Friis cascaded noise figure.

if isempty(stages) || ~isstruct(stages) || ...
        ~all(isfield(stages, {'gainDb', 'noiseFigureDb'}))
    error('rf:cascade_metrics:InvalidStages', ...
        'stages must be a nonempty struct array with gainDb and noiseFigureDb.');
end

gainDb = [stages.gainDb];
noiseFigureDb = [stages.noiseFigureDb];
if any(~isfinite(gainDb)) || any(~isfinite(noiseFigureDb)) || ...
        any(noiseFigureDb < 0)
    error('rf:cascade_metrics:InvalidStageValue', ...
        'Stage gains must be finite and noise figures must be finite and nonnegative.');
end

gainLinear = 10.^(gainDb / 10);
noiseFactor = 10.^(noiseFigureDb / 10);
cascadeNoiseFactor = noiseFactor(1);
precedingGain = gainLinear(1);
for k = 2:numel(stages)
    cascadeNoiseFactor = cascadeNoiseFactor + ...
        (noiseFactor(k) - 1) / precedingGain;
    precedingGain = precedingGain * gainLinear(k);
end

metrics.totalGainLinear = prod(gainLinear);
metrics.totalGainDb = 10 * log10(metrics.totalGainLinear);
metrics.noiseFactor = cascadeNoiseFactor;
metrics.noiseFigureDb = 10 * log10(cascadeNoiseFactor);
metrics.stageGainDb = gainDb;
metrics.stageNoiseFigureDb = noiseFigureDb;
end
