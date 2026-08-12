function [x, truth, knownData] = generate_e5_signal(cfg)
%GENERATE_E5_SIGNAL Generate one complete Galileo E5 complex low-IF signal.

required = {'svid', 'carrierHz', 'ifHz', 'fsHz', 'codeRateHz', ...
    'subcarrierRateHz', 'durationMs', 'dopplerHz', ...
    'initialCodePhaseChips', 'initialCarrierPhaseRad', 'cn0DbHz'};
if ~isstruct(cfg) || ~isfield(cfg, 'signal')
    error('galileo:generate_e5_signal:MissingConfig', ...
        'cfg.signal is required.');
end
for k = 1:numel(required)
    if ~isfield(cfg.signal, required{k})
        error('galileo:generate_e5_signal:MissingConfig', ...
            'Missing signal configuration field: %s.', required{k});
    end
end

nSamplesExact = cfg.signal.durationMs * 1e-3 * cfg.signal.fsHz;
if ~isfinite(nSamplesExact) || nSamplesExact < 1 || ...
        abs(nSamplesExact - round(nSamplesExact)) > 1e-9
    error('galileo:generate_e5_signal:InvalidDuration', ...
        'Duration and sample rate must produce an integer sample count.');
end
if ~isfinite(cfg.signal.initialCodePhaseChips) || ...
        cfg.signal.initialCodePhaseChips < 0
    error('galileo:generate_e5_signal:InvalidCodePhase', ...
        'Initial absolute code phase must be finite and nonnegative.');
end
if ~(isinf(cfg.signal.cn0DbHz) || ...
        (isfinite(cfg.signal.cn0DbHz) && cfg.signal.cn0DbHz > 0))
    error('galileo:generate_e5_signal:InvalidCn0', ...
        'C/N0 must be positive and finite, or Inf for a noiseless signal.');
end

nSamples = round(nSamplesExact);
rng(cfg.randomSeed, 'twister');
dopplerScale = 1 + cfg.signal.dopplerHz / cfg.signal.carrierHz;
receivedCodeRateHz = cfg.signal.codeRateHz * dopplerScale;
n = 0:nSamples-1;
codePhase = cfg.signal.initialCodePhaseChips + ...
    n * receivedCodeRateHz / cfg.signal.fsHz;
lastPrimaryEpoch = floor(codePhase(end) / ...
    cfg.signal.primaryLengthChips);
nE5aSymbols = floor(lastPrimaryEpoch / 20) + 1;
nE5bSymbols = floor(lastPrimaryEpoch / 4) + 1;
knownData.e5aI = 1 - 2*randi([0, 1], 1, nE5aSymbols);
knownData.e5bI = 1 - 2*randi([0, 1], 1, nE5bSymbols);

basebandReplica = galileo.altboc_replica(codePhase, cfg, knownData);
carrierFrequencyHz = cfg.signal.ifHz + cfg.signal.dopplerHz;
carrier = exp(1j * (2*pi*carrierFrequencyHz*n/cfg.signal.fsHz + ...
    cfg.signal.initialCarrierPhaseRad));
noiseless = basebandReplica .* carrier;

if isinf(cfg.signal.cn0DbHz)
    noiseVariance = 0;
    noise = complex(zeros(1, nSamples));
else
    cn0Linear = 10^(cfg.signal.cn0DbHz / 10);
    noiseVariance = cfg.signal.fsHz / cn0Linear;
    noise = sqrt(noiseVariance/2) * ...
        (randn(1, nSamples) + 1j*randn(1, nSamples));
end
x = noiseless + noise;

truth.svid = cfg.signal.svid;
truth.dopplerHz = cfg.signal.dopplerHz;
truth.carrierFrequencyHz = carrierFrequencyHz;
truth.initialCarrierPhaseRad = cfg.signal.initialCarrierPhaseRad;
truth.initialCodePhaseChips = cfg.signal.initialCodePhaseChips;
truth.finalCodePhaseChips = codePhase(end);
truth.codeRateHz = receivedCodeRateHz;
truth.ifHz = cfg.signal.ifHz;
truth.e5aCenterHz = cfg.signal.ifHz - cfg.signal.subcarrierRateHz;
truth.e5bCenterHz = cfg.signal.ifHz + cfg.signal.subcarrierRateHz;
truth.fsHz = cfg.signal.fsHz;
truth.noiseVariance = noiseVariance;
truth.basebandReplica = basebandReplica;
end
