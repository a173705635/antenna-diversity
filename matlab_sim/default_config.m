function cfg = default_config()
%DEFAULT_CONFIG Return the baseline dual-antenna GPS L1 C/A simulation setup.

c = 299792458;

cfg.randomSeed = 42;

cfg.signal.prn = 1;
cfg.signal.carrierHz = 1575.42e6;
cfg.signal.wavelengthM = c / cfg.signal.carrierHz;
cfg.signal.codeRateHz = 1.023e6;
cfg.signal.fsRfHz = 16.368e6;
cfg.signal.ifHz = 4.092e6;
cfg.signal.dopplerHz = 1500;
cfg.signal.codeDelayChips = 137.25;
cfg.signal.initialPhaseRad = 0.35;
cfg.signal.durationMs = 40;
cfg.signal.powerDbm = -128.5;
cfg.signal.resistanceOhm = 50;

cfg.array.numElements = 2;
cfg.array.spacingM = cfg.signal.wavelengthM / 2;
cfg.array.positionsM = (0:cfg.array.numElements-1).' * cfg.array.spacingM;
cfg.array.desiredAngleDeg = -10;
cfg.array.channelAmplitudeErrorDb = zeros(cfg.array.numElements, 1);
cfg.array.channelPhaseErrorDeg = zeros(cfg.array.numElements, 1);

cfg.jammer.enable = true;
cfg.jammer.powerDbm = -90;
cfg.jammer.offsetHz = 250e3;
cfg.jammer.angleDeg = 35;
cfg.jammer.initialPhaseRad = 0.8;

cfg.rf.temperatureK = 290;
cfg.rf.noiseBandwidthHz = 8.184e6;
cfg.rf.resistanceOhm = cfg.signal.resistanceOhm;
cfg.rf.stages = [ ...
    struct('name', 'LNA',   'gainDb', 28, 'noiseFigureDb', 1.2), ...
    struct('name', 'Mixer', 'gainDb', 8,  'noiseFigureDb', 8.0), ...
    struct('name', 'VGA',   'gainDb', 44, 'noiseFigureDb', 10.0)];
cfg.rf.adcBits = 10;
cfg.rf.adcVpp = 1.0;
cfg.rf.enableSoftLimiting = true;

cfg.dfe.fsInHz = cfg.signal.fsRfHz;
cfg.dfe.ifHz = cfg.signal.ifHz;
cfg.dfe.decimation = 4;
cfg.dfe.fsOutHz = cfg.dfe.fsInHz / cfg.dfe.decimation;
cfg.dfe.cutoffHz = 1.8e6;
cfg.dfe.numTaps = 129;
cfg.dfe.inputBits = cfg.rf.adcBits;
cfg.dfe.outputBits = 8;
cfg.dfe.outputBinaryShift = cfg.dfe.inputBits - cfg.dfe.outputBits;

cfg.diversity.modes = {'single', 'egc', 'mvdr'};
cfg.diversity.diagonalLoadingFactor = 1e-2;

cfg.acquisition.dopplerBinsHz = -5000:250:5000;
cfg.acquisition.coherentMs = 1;
cfg.acquisition.noncoherentMs = 4;
cfg.acquisition.exclusionChips = 1;
cfg.acquisition.threshold = 2.5;
cfg.acquisition.codeRateHz = cfg.signal.codeRateHz;

cfg.tracking.coherentMs = 1;
cfg.tracking.acquisitionMs = cfg.acquisition.coherentMs * ...
    cfg.acquisition.noncoherentMs;
cfg.tracking.earlyLateSpacingChips = 0.5;
cfg.tracking.fllAssistEpochs = 5;
cfg.tracking.fllGain = 0.5;
cfg.tracking.pllKp = 0.30;
cfg.tracking.pllKiHz = 4.0;
cfg.tracking.dllKpChips = 0.08;
cfg.tracking.dllKiHz = 0.5;
cfg.tracking.cn0WindowEpochs = 10;
cfg.tracking.maxEpochs = 30;
cfg.tracking.maxFinalDopplerErrorHz = 250;

cfg.plot.enable = true;
cfg.plot.visible = 'on';

samplesPerCode = cfg.dfe.fsOutHz * 1e-3;
samplesPerChip = cfg.dfe.fsOutHz / cfg.signal.codeRateHz;
if abs(samplesPerCode - round(samplesPerCode)) > 1e-12
    error('default_config:NonintegerCodePeriod', ...
        'DFE output must contain an integer number of samples per code period.');
end
if abs(samplesPerChip - round(samplesPerChip)) > 1e-12 || samplesPerChip < 2
    error('default_config:InvalidSamplesPerChip', ...
        'DFE output must contain at least two integer samples per chip.');
end
if mod(cfg.dfe.numTaps, 2) ~= 1
    error('default_config:EvenFilterLength', 'DFE FIR length must be odd.');
end
end
