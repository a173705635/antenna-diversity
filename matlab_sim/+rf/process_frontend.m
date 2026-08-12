function [adcCodes, info] = process_frontend(xAntennaV, cfg, randomSeed)
%PROCESS_FRONTEND Apply equivalent RF noise/gain, limiting, and ADC quantization.

if ~ismatrix(xAntennaV) || isempty(xAntennaV)
    error('rf:process_frontend:InvalidInput', ...
        'xAntennaV must be a nonempty antennas-by-samples matrix.');
end
if cfg.temperatureK < 0 || cfg.noiseBandwidthHz <= 0 || ...
        cfg.resistanceOhm <= 0 || cfg.adcBits < 2 || cfg.adcVpp <= 0
    error('rf:process_frontend:InvalidConfig', ...
        'RF temperature, bandwidth, resistance, ADC bits, and full scale are invalid.');
end

rng(randomSeed, 'twister');
cascade = rf.cascade_metrics(cfg.stages);
kBoltzmann = 1.380649e-23;
noisePowerInputW = kBoltzmann * cfg.temperatureK * ...
    cfg.noiseBandwidthHz * cascade.noiseFactor;
noiseRmsV = sqrt(noisePowerInputW * cfg.resistanceOhm);
noise = noiseRmsV / sqrt(2) * ...
    (randn(size(xAntennaV)) + 1j * randn(size(xAntennaV)));

voltageGain = sqrt(cascade.totalGainLinear);
adcInputBeforeLimit = (xAntennaV + noise) * voltageGain;
componentLimitV = cfg.adcVpp / 2;
saturated = abs(real(adcInputBeforeLimit)) > componentLimitV | ...
    abs(imag(adcInputBeforeLimit)) > componentLimitV;

if cfg.enableSoftLimiting
    adcInputV = componentLimitV * tanh(real(adcInputBeforeLimit) / componentLimitV) + ...
        1j * componentLimitV * tanh(imag(adcInputBeforeLimit) / componentLimitV);
else
    adcInputV = complex( ...
        min(max(real(adcInputBeforeLimit), -componentLimitV), componentLimitV), ...
        min(max(imag(adcInputBeforeLimit), -componentLimitV), componentLimitV));
end

lsbV = cfg.adcVpp / 2^cfg.adcBits;
minCode = -2^(cfg.adcBits - 1);
maxCode = 2^(cfg.adcBits - 1) - 1;
iCode = min(max(round(real(adcInputV) / lsbV), minCode), maxCode);
qCode = min(max(round(imag(adcInputV) / lsbV), minCode), maxCode);
adcCodes = complex(iCode, qCode);

info.totalGainDb = cascade.totalGainDb;
info.noiseFigureDb = cascade.noiseFigureDb;
info.inputNoisePowerW = noisePowerInputW;
info.inputNoiseRmsV = noiseRmsV;
info.adcLsbV = lsbV;
info.saturationFraction = mean(saturated, 2);
info.inputRmsV = sqrt(mean(abs(xAntennaV).^2, 2));
info.adcInputRmsV = sqrt(mean(abs(adcInputBeforeLimit).^2, 2));
info.adcOutputRmsCode = sqrt(mean(abs(adcCodes).^2, 2));
info.adcBits = cfg.adcBits;
info.adcVpp = cfg.adcVpp;
end
