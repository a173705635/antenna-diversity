function [xAntennaV, truth] = generate_scenario(cfg)
%GENERATE_SCENARIO Generate dual-antenna GPS and CW-interference voltages.

required = {'signal', 'array', 'jammer'};
for k = 1:numel(required)
    if ~isfield(cfg, required{k})
        error('gnss:generate_scenario:MissingConfig', ...
            'Missing configuration group: %s.', required{k});
    end
end

nAnt = cfg.array.numElements;
if numel(cfg.array.positionsM) ~= nAnt
    error('gnss:generate_scenario:ArraySizeMismatch', ...
        'array.positionsM must contain one position per element.');
end
nSamples = round(cfg.signal.fsRfHz * cfg.signal.durationMs * 1e-3);
if nSamples <= 0
    error('gnss:generate_scenario:InvalidDuration', ...
        'The signal duration must produce at least one sample.');
end

code = gnss.ca_code(cfg.signal.prn);
n = 0:nSamples-1;
t = n / cfg.signal.fsRfHz;
receivedCodeRateHz = cfg.signal.codeRateHz * ...
    (1 + cfg.signal.dopplerHz / cfg.signal.carrierHz);
codePhase = mod(n * receivedCodeRateHz / cfg.signal.fsRfHz - ...
    cfg.signal.codeDelayChips, 1023);
sampledCode = code(floor(codePhase) + 1);

signalPowerW = 10.^((cfg.signal.powerDbm - 30) / 10);
signalAmplitudeV = sqrt(signalPowerW * cfg.signal.resistanceOhm);
signalCarrier = exp(1j * (2*pi * ...
    (cfg.signal.ifHz + cfg.signal.dopplerHz) * t + ...
    cfg.signal.initialPhaseRad));
desiredWaveform = signalAmplitudeV * sampledCode .* signalCarrier;

desiredSteering = arraysim.steering_vector(cfg.array.positionsM, ...
    cfg.array.desiredAngleDeg, cfg.signal.wavelengthM);
channelResponse = 10.^(cfg.array.channelAmplitudeErrorDb(:) / 20) .* ...
    exp(1j * deg2rad(cfg.array.channelPhaseErrorDeg(:)));
if numel(channelResponse) ~= nAnt
    error('gnss:generate_scenario:ChannelErrorSizeMismatch', ...
        'Channel error vectors must contain one value per element.');
end
desiredSteeringEffective = channelResponse .* desiredSteering;
desiredComponent = desiredSteeringEffective * desiredWaveform;

if cfg.jammer.enable
    jammerPowerW = 10.^((cfg.jammer.powerDbm - 30) / 10);
    jammerAmplitudeV = sqrt(jammerPowerW * cfg.signal.resistanceOhm);
    jammerWaveform = jammerAmplitudeV * exp(1j * (2*pi * ...
        (cfg.signal.ifHz + cfg.jammer.offsetHz) * t + ...
        cfg.jammer.initialPhaseRad));
    jammerSteering = arraysim.steering_vector(cfg.array.positionsM, ...
        cfg.jammer.angleDeg, cfg.signal.wavelengthM);
    jammerSteeringEffective = channelResponse .* jammerSteering;
    jammerComponent = jammerSteeringEffective * jammerWaveform;
else
    jammerSteering = zeros(nAnt, 1);
    jammerSteeringEffective = zeros(nAnt, 1);
    jammerComponent = complex(zeros(nAnt, nSamples));
end

xAntennaV = desiredComponent + jammerComponent;

truth.timeSec = t;
truth.code = code;
truth.desiredWaveform = desiredWaveform;
truth.desiredComponent = desiredComponent;
truth.jammerComponent = jammerComponent;
truth.desiredSteeringNominal = desiredSteering;
truth.desiredSteeringEffective = desiredSteeringEffective;
truth.jammerSteeringNominal = jammerSteering;
truth.jammerSteeringEffective = jammerSteeringEffective;
truth.channelResponse = channelResponse;
truth.dopplerHz = cfg.signal.dopplerHz;
truth.codeRateHz = receivedCodeRateHz;
truth.codeDelayChips = cfg.signal.codeDelayChips;
truth.codeDelayRfSamples = cfg.signal.codeDelayChips * ...
    cfg.signal.fsRfHz / cfg.signal.codeRateHz;
truth.fsRfHz = cfg.signal.fsRfHz;
truth.ifHz = cfg.signal.ifHz;
end
