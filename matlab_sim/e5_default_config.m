function cfg = e5_default_config()
%E5_DEFAULT_CONFIG Return the standalone Galileo E5 tracking setup.

cfg.randomSeed = 81;

cfg.signal.svid = 1;
cfg.signal.carrierHz = 1191.795e6;
cfg.signal.loHz = 1188e6;
cfg.signal.ifHz = cfg.signal.carrierHz - cfg.signal.loHz;
cfg.signal.fsHz = 132e6;
cfg.signal.codeRateHz = 10.23e6;
cfg.signal.subcarrierRateHz = 15.345e6;
cfg.signal.primaryLengthChips = 10230;
cfg.signal.epochSec = 1e-3;
cfg.signal.samplesPerEpoch = round(cfg.signal.fsHz * cfg.signal.epochSec);
cfg.signal.durationMs = 30;
cfg.signal.dopplerHz = 850;
cfg.signal.initialCodePhaseChips = 1234.25;
cfg.signal.initialCarrierPhaseRad = 0.35;
cfg.signal.cn0DbHz = 52;

cfg.initialization.dopplerErrorHz = 200;
cfg.initialization.codePhaseErrorChips = 0.08;
cfg.initialization.carrierPhaseErrorRad = 0.20;

cfg.tracking.coherentMs = 1;
cfg.tracking.maxEpochs = 25;
cfg.tracking.earlyLateSpacingChips = 0.10;
cfg.tracking.fllAssistEpochs = 5;
cfg.tracking.fllGain = 0.45;
cfg.tracking.pllKp = 0.25;
cfg.tracking.pllKiHz = 8.0;
cfg.tracking.dllKpChips = 0.04;
cfg.tracking.dllKiHz = 0.20;
cfg.tracking.cn0WindowEpochs = 10;
cfg.tracking.sidePeakOffsetChips = 2/3;
cfg.tracking.sidePeakRatio = 1.10;
cfg.tracking.sidePeakConfirmEpochs = 2;

cfg.plot.enable = true;
cfg.plot.visible = 'on';

validate_config(cfg);
end

function validate_config(cfg)
frequencies = [cfg.signal.carrierHz, cfg.signal.loHz, cfg.signal.fsHz, ...
    cfg.signal.codeRateHz, cfg.signal.subcarrierRateHz];
if any(~isfinite(frequencies)) || any(frequencies <= 0)
    error('e5_default_config:InvalidFrequency', ...
        'Carrier, LO, sample, code, and subcarrier rates must be positive.');
end
if cfg.signal.svid ~= 1
    error('e5_default_config:UnsupportedSvid', ...
        'The standalone validation supports SVID 1 only.');
end
samplesPerEpochExact = cfg.signal.fsHz * cfg.signal.epochSec;
if abs(samplesPerEpochExact - round(samplesPerEpochExact)) > 1e-9
    error('e5_default_config:NonintegerEpoch', ...
        'The sample rate must produce an integer number of samples per epoch.');
end
if ~isscalar(cfg.signal.durationMs) || ~isfinite(cfg.signal.durationMs) || ...
        cfg.signal.durationMs <= 0
    error('e5_default_config:InvalidDuration', ...
        'signal.durationMs must be a positive finite scalar.');
end
if cfg.tracking.earlyLateSpacingChips <= 0 || ...
        cfg.tracking.earlyLateSpacingChips >= ...
        cfg.tracking.sidePeakOffsetChips
    error('e5_default_config:InvalidCorrelatorSpacing', ...
        'Early/late spacing must lie between zero and the side-peak offset.');
end
end
