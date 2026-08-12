function trk = track(x, fsHz, prn, acq, cfg, carrierHz, nominalCodeRateHz)
%TRACK Run 1 ms Early/Prompt/Late DLL with FLL-assisted PLL tracking.

if ~isvector(x) || isempty(x)
    error('receiver:track:InvalidInput', 'x must be a nonempty vector.');
end
if ~isfield(acq, 'success') || ~acq.success
    trk.success = false;
    trk.status = 'acquisition_failed';
    trk.prompt = complex([]);
    return;
end
if fsHz <= 0 || carrierHz <= 0 || nominalCodeRateHz <= 0
    error('receiver:track:InvalidFrequency', ...
        'Sample, carrier, and code frequencies must be positive.');
end

x = x(:).';
samplesPerEpochExact = fsHz * cfg.coherentMs * 1e-3;
if abs(samplesPerEpochExact - round(samplesPerEpochExact)) > 1e-9
    error('receiver:track:NonintegerEpoch', ...
        'The coherent interval must contain an integer number of samples.');
end
samplesPerEpoch = round(samplesPerEpochExact);
acquisitionSamples = round(cfg.acquisitionMs * 1e-3 * fsHz);

firstEpoch = acq.codePhaseSamples + 1;
while firstEpoch <= acquisitionSamples
    firstEpoch = firstEpoch + round(fsHz * 1e-3);
end
availableEpochs = floor((numel(x) - firstEpoch + 1) / samplesPerEpoch);
nEpochs = min(availableEpochs, cfg.maxEpochs);
if nEpochs < 1
    error('receiver:track:InsufficientSamples', ...
        'No complete tracking epoch remains after acquisition.');
end

code = gnss.ca_code(prn);
sampleIndex = 0:samplesPerEpoch-1;
epochDurationSec = samplesPerEpoch / fsHz;
carrierFrequencyHz = acq.dopplerHz;
carrierPhaseRad = 0;
codePhaseChips = 0;
dllFrequencyCorrectionHz = 0;
codeRateHz = nominalCodeRateHz * (1 + carrierFrequencyHz / carrierHz);
previousPrompt = [];

early = complex(zeros(1, nEpochs));
prompt = complex(zeros(1, nEpochs));
late = complex(zeros(1, nEpochs));
dllError = zeros(1, nEpochs);
pllErrorRad = zeros(1, nEpochs);
fllErrorHz = zeros(1, nEpochs);
carrierFrequencyHistory = zeros(1, nEpochs);
codeRateHistory = zeros(1, nEpochs);
codePhaseHistory = zeros(1, nEpochs);
phaseLockMetric = zeros(1, nEpochs);
cn0DbHz = nan(1, nEpochs);
epochStartSamples = zeros(1, nEpochs);

for epoch = 1:nEpochs
    blockFirst = firstEpoch + (epoch - 1) * samplesPerEpoch;
    block = x(blockFirst:blockFirst + samplesPerEpoch - 1);
    epochStartSamples(epoch) = blockFirst - 1;

    carrierPhase = carrierPhaseRad + ...
        2*pi * carrierFrequencyHz * sampleIndex / fsHz;
    baseband = block .* exp(-1j * carrierPhase);

    promptPhase = codePhaseChips + sampleIndex * codeRateHz / fsHz;
    halfSpacing = cfg.earlyLateSpacingChips / 2;
    promptCode = sample_code(code, promptPhase);
    earlyCode = sample_code(code, promptPhase + halfSpacing);
    lateCode = sample_code(code, promptPhase - halfSpacing);

    early(epoch) = sum(baseband .* earlyCode);
    prompt(epoch) = sum(baseband .* promptCode);
    late(epoch) = sum(baseband .* lateCode);

    dllError(epoch) = (abs(early(epoch)) - abs(late(epoch))) / ...
        (abs(early(epoch)) + abs(late(epoch)) + eps);
    pllErrorRad(epoch) = atan2(imag(prompt(epoch)), real(prompt(epoch)));
    if ~isempty(previousPrompt)
        fllErrorHz(epoch) = angle(prompt(epoch) * conj(previousPrompt)) / ...
            (2*pi * epochDurationSec);
    end

    fllCorrection = 0;
    if epoch <= cfg.fllAssistEpochs && epoch > 1
        fllCorrection = cfg.fllGain * fllErrorHz(epoch);
    end
    carrierFrequencyHz = carrierFrequencyHz + fllCorrection + ...
        cfg.pllKiHz * pllErrorRad(epoch);
    carrierPhaseRad = wrap_phase(carrierPhaseRad + ...
        2*pi * carrierFrequencyHz * epochDurationSec + ...
        cfg.pllKp * pllErrorRad(epoch));

    dllFrequencyCorrectionHz = dllFrequencyCorrectionHz + ...
        cfg.dllKiHz * dllError(epoch);
    codeRateHz = nominalCodeRateHz * (1 + carrierFrequencyHz / carrierHz) + ...
        dllFrequencyCorrectionHz;
    codePhaseChips = mod(codePhaseChips + ...
        samplesPerEpoch * codeRateHz / fsHz + ...
        cfg.dllKpChips * dllError(epoch), 1023);

    carrierFrequencyHistory(epoch) = carrierFrequencyHz;
    codeRateHistory(epoch) = codeRateHz;
    codePhaseHistory(epoch) = codePhaseChips;
    previousPrompt = prompt(epoch);

    windowFirst = max(1, epoch - cfg.cn0WindowEpochs + 1);
    promptWindow = prompt(windowFirst:epoch);
    phaseLockMetric(epoch) = abs(sum(promptWindow)) / ...
        (sum(abs(promptWindow)) + eps);
    if numel(promptWindow) >= 2
        coherentPower = abs(mean(promptWindow))^2;
        totalPower = mean(abs(promptWindow).^2);
        noisePower = max(totalPower - coherentPower, eps);
        cn0DbHz(epoch) = 10*log10(max(coherentPower / ...
            (noisePower * epochDurationSec), eps));
    end
end

trk.success = true;
trk.status = 'tracking';
trk.firstEpochSample = firstEpoch - 1;
trk.epochStartSamples = epochStartSamples;
trk.early = early;
trk.prompt = prompt;
trk.late = late;
trk.dllError = dllError;
trk.pllErrorRad = pllErrorRad;
trk.fllErrorHz = fllErrorHz;
trk.carrierFrequencyHz = carrierFrequencyHistory;
trk.codeRateHz = codeRateHistory;
trk.codePhaseChips = codePhaseHistory;
trk.promptMagnitude = abs(prompt);
trk.phaseLockMetric = phaseLockMetric;
trk.cn0DbHz = cn0DbHz;
end

function sampled = sample_code(code, phaseChips)
indices = floor(mod(phaseChips, 1023)) + 1;
sampled = code(indices);
end

function phase = wrap_phase(phase)
phase = mod(phase + pi, 2*pi) - pi;
end
