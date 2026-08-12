function trk = track_e5_fullband(x, fsHz, cfg, init, knownData)
%TRACK_E5_FULLBAND Track the complete E5 AltBOC waveform directly.

if ~isvector(x) || isempty(x) || any(~isfinite(x))
    error('galileo:track_e5_fullband:InvalidInput', ...
        'x must be a nonempty finite complex vector.');
end
if ~isscalar(fsHz) || ~isfinite(fsHz) || fsHz <= 0
    error('galileo:track_e5_fullband:InvalidSampleRate', ...
        'fsHz must be a positive finite scalar.');
end
requiredInit = {'dopplerHz', 'carrierPhaseRad', 'codePhaseChips'};
for k = 1:numel(requiredInit)
    if ~isstruct(init) || ~isfield(init, requiredInit{k}) || ...
            ~isscalar(init.(requiredInit{k})) || ...
            ~isfinite(init.(requiredInit{k}))
        error('galileo:track_e5_fullband:InvalidInitialization', ...
            'init.%s must be a finite scalar.', requiredInit{k});
    end
end
if init.codePhaseChips < 0
    error('galileo:track_e5_fullband:InvalidInitialization', ...
        'Initial absolute code phase must be nonnegative.');
end

x = x(:).';
epochSec = cfg.tracking.coherentMs * 1e-3;
samplesPerEpochExact = fsHz * epochSec;
if abs(samplesPerEpochExact - round(samplesPerEpochExact)) > 1e-9
    error('galileo:track_e5_fullband:NonintegerEpoch', ...
        'The coherent interval must contain an integer sample count.');
end
samplesPerEpoch = round(samplesPerEpochExact);
nEpochs = min(floor(numel(x) / samplesPerEpoch), ...
    cfg.tracking.maxEpochs);
if nEpochs < 1
    error('galileo:track_e5_fullband:InsufficientSamples', ...
        'At least one complete coherent epoch is required.');
end

sampleIndex = 0:samplesPerEpoch-1;
dopplerHz = init.dopplerHz;
carrierPhaseRad = wrap_phase(init.carrierPhaseRad);
codePhaseChips = init.codePhaseChips;
dllFrequencyCorrectionHz = 0;
codeRateHz = cfg.signal.codeRateHz * ...
    (1 + dopplerHz / cfg.signal.carrierHz);
previousPrompt = [];

early = complex(zeros(1, nEpochs));
prompt = complex(zeros(1, nEpochs));
late = complex(zeros(1, nEpochs));
dllError = zeros(1, nEpochs);
pllErrorRad = zeros(1, nEpochs);
fllErrorHz = zeros(1, nEpochs);
dopplerHistory = zeros(1, nEpochs);
carrierPhaseHistory = zeros(1, nEpochs);
codeRateHistory = zeros(1, nEpochs);
codePhaseHistory = zeros(1, nEpochs);
epochStartSamples = zeros(1, nEpochs);
promptMagnitude = zeros(1, nEpochs);
phaseLockMetric = zeros(1, nEpochs);
cn0DbHz = nan(1, nEpochs);
sidePeakMinus = complex(zeros(1, nEpochs));
sidePeakPlus = complex(zeros(1, nEpochs));
sidePeakJumpChips = zeros(1, nEpochs);
minusConfirmationCount = 0;
plusConfirmationCount = 0;

for epoch = 1:nEpochs
    blockFirst = (epoch - 1) * samplesPerEpoch + 1;
    block = x(blockFirst:blockFirst + samplesPerEpoch - 1);

    epochStartSamples(epoch) = blockFirst - 1;
    dopplerHistory(epoch) = dopplerHz;
    carrierPhaseHistory(epoch) = carrierPhaseRad;
    codeRateHistory(epoch) = codeRateHz;
    codePhaseHistory(epoch) = codePhaseChips;

    carrierPhase = carrierPhaseRad + 2*pi * ...
        (cfg.signal.ifHz + dopplerHz) * sampleIndex / fsHz;
    wiped = block .* exp(-1j * carrierPhase);

    phase = codePhaseChips + sampleIndex * codeRateHz / fsHz;
    halfSpacing = cfg.tracking.earlyLateSpacingChips / 2;
    earlyReplica = galileo.altboc_replica(phase + halfSpacing, ...
        cfg, knownData);
    promptReplica = galileo.altboc_replica(phase, cfg, knownData);
    lateReplica = galileo.altboc_replica(phase - halfSpacing, ...
        cfg, knownData);
    sideMinusReplica = galileo.altboc_replica(phase - ...
        cfg.tracking.sidePeakOffsetChips, cfg, knownData);
    sidePlusReplica = galileo.altboc_replica(phase + ...
        cfg.tracking.sidePeakOffsetChips, cfg, knownData);

    early(epoch) = mean(wiped .* conj(earlyReplica));
    prompt(epoch) = mean(wiped .* conj(promptReplica));
    late(epoch) = mean(wiped .* conj(lateReplica));
    sidePeakMinus(epoch) = mean(wiped .* conj(sideMinusReplica));
    sidePeakPlus(epoch) = mean(wiped .* conj(sidePlusReplica));
    promptMagnitude(epoch) = abs(prompt(epoch));

    earlyPower = abs(early(epoch))^2;
    latePower = abs(late(epoch))^2;
    dllError(epoch) = (earlyPower - latePower) / ...
        (earlyPower + latePower + eps);
    pllErrorRad(epoch) = angle(prompt(epoch));
    if ~isempty(previousPrompt)
        fllErrorHz(epoch) = angle(prompt(epoch) * conj(previousPrompt)) / ...
            (2*pi*epochSec);
    end

    fllCorrectionHz = 0;
    if epoch > 1 && epoch <= cfg.tracking.fllAssistEpochs
        fllCorrectionHz = cfg.tracking.fllGain * fllErrorHz(epoch);
    end
    dopplerHz = dopplerHz + fllCorrectionHz + ...
        cfg.tracking.pllKiHz * pllErrorRad(epoch);
    carrierPhaseRad = wrap_phase(carrierPhaseRad + 2*pi * ...
        (cfg.signal.ifHz + dopplerHz) * epochSec + ...
        cfg.tracking.pllKp * pllErrorRad(epoch));

    dllFrequencyCorrectionHz = dllFrequencyCorrectionHz + ...
        cfg.tracking.dllKiHz * dllError(epoch);
    codeRateHz = cfg.signal.codeRateHz * ...
        (1 + dopplerHz / cfg.signal.carrierHz) + ...
        dllFrequencyCorrectionHz;
    codePhaseChips = codePhaseChips + ...
        samplesPerEpoch * codeRateHz / fsHz + ...
        cfg.tracking.dllKpChips * dllError(epoch);

    minusIsStronger = abs(sidePeakMinus(epoch)) > ...
        cfg.tracking.sidePeakRatio * abs(prompt(epoch)) && ...
        abs(sidePeakMinus(epoch)) > abs(sidePeakPlus(epoch));
    plusIsStronger = abs(sidePeakPlus(epoch)) > ...
        cfg.tracking.sidePeakRatio * abs(prompt(epoch)) && ...
        abs(sidePeakPlus(epoch)) > abs(sidePeakMinus(epoch));
    if minusIsStronger
        minusConfirmationCount = minusConfirmationCount + 1;
        plusConfirmationCount = 0;
    elseif plusIsStronger
        plusConfirmationCount = plusConfirmationCount + 1;
        minusConfirmationCount = 0;
    else
        minusConfirmationCount = 0;
        plusConfirmationCount = 0;
    end
    if minusConfirmationCount >= cfg.tracking.sidePeakConfirmEpochs
        sidePeakJumpChips(epoch) = -cfg.tracking.sidePeakOffsetChips;
        codePhaseChips = codePhaseChips + sidePeakJumpChips(epoch);
        minusConfirmationCount = 0;
    elseif plusConfirmationCount >= cfg.tracking.sidePeakConfirmEpochs
        sidePeakJumpChips(epoch) = cfg.tracking.sidePeakOffsetChips;
        codePhaseChips = codePhaseChips + sidePeakJumpChips(epoch);
        plusConfirmationCount = 0;
    end
    previousPrompt = prompt(epoch);

    windowFirst = max(1, epoch - cfg.tracking.cn0WindowEpochs + 1);
    promptWindow = prompt(windowFirst:epoch);
    phaseLockMetric(epoch) = abs(sum(promptWindow)) / ...
        (sum(abs(promptWindow)) + eps);
    if numel(promptWindow) >= 2
        coherentPower = abs(mean(promptWindow))^2;
        totalPower = mean(abs(promptWindow).^2);
        noisePower = max(totalPower - coherentPower, eps);
        cn0DbHz(epoch) = 10*log10(max(coherentPower / ...
            (noisePower * epochSec), eps));
    end
end

trk.success = true;
trk.status = 'tracking';
trk.epochStartSamples = epochStartSamples;
trk.early = early;
trk.prompt = prompt;
trk.late = late;
trk.dllError = dllError;
trk.pllErrorRad = pllErrorRad;
trk.fllErrorHz = fllErrorHz;
trk.dopplerHz = dopplerHistory;
trk.carrierPhaseRad = carrierPhaseHistory;
trk.codeRateHz = codeRateHistory;
trk.codePhaseChips = codePhaseHistory;
trk.promptMagnitude = promptMagnitude;
trk.phaseLockMetric = phaseLockMetric;
trk.cn0DbHz = cn0DbHz;
trk.sidePeakMinus = sidePeakMinus;
trk.sidePeakPlus = sidePeakPlus;
trk.sidePeakJumpChips = sidePeakJumpChips;
end

function phase = wrap_phase(phase)
phase = mod(phase + pi, 2*pi) - pi;
end
