function results = run_e5_fullband_tracking(cfg)
%RUN_E5_FULLBAND_TRACKING Run standalone full-band Galileo E5 tracking.

if nargin < 1 || isempty(cfg)
    cfg = e5_default_config();
end

[x, truth, knownData] = galileo.generate_e5_signal(cfg);
init.dopplerHz = truth.dopplerHz + ...
    cfg.initialization.dopplerErrorHz;
init.carrierPhaseRad = truth.initialCarrierPhaseRad + ...
    cfg.initialization.carrierPhaseErrorRad;
init.codePhaseChips = truth.initialCodePhaseChips + ...
    cfg.initialization.codePhaseErrorChips;

tracking = galileo.track_e5_fullband(x, cfg.signal.fsHz, ...
    cfg, init, knownData);
summary = build_summary(tracking, truth, cfg);

results.config = cfg;
results.truth = truth;
results.knownData = knownData;
results.input = x;
results.initialization = init;
results.tracking = tracking;
results.summary = summary;

fprintf(['Galileo E5 full-band tracking: Doppler error %.3f Hz, ' ...
    'code error %.6f chip, PLL RMS %.4f rad\n'], ...
    summary.finalDopplerErrorHz, ...
    summary.finalCodePhaseErrorChips, summary.pllRmsRad);

if cfg.plot.enable
    plot_results(results);
end
end

function summary = build_summary(tracking, truth, cfg)
window = max(1, numel(tracking.prompt)-9):numel(tracking.prompt);
epochTimeSec = tracking.epochStartSamples(window) / cfg.signal.fsHz;
trueCodePhase = truth.initialCodePhaseChips + ...
    epochTimeSec * truth.codeRateHz;
codeError = wrap_code(tracking.codePhaseChips(window) - ...
    trueCodePhase, cfg.signal.primaryLengthChips);

summary.finalDopplerErrorHz = mean(tracking.dopplerHz(window)) - ...
    truth.dopplerHz;
summary.finalCodePhaseErrorChips = mean(codeError);
summary.pllRmsRad = sqrt(mean(tracking.pllErrorRad(window).^2));
summary.dllRms = sqrt(mean(tracking.dllError(window).^2));
summary.meanPromptMagnitude = mean(abs(tracking.prompt(window)));
summary.sidePeakJumpCount = nnz(tracking.sidePeakJumpChips);
end

function value = wrap_code(value, period)
value = mod(value + period/2, period) - period/2;
end

function plot_results(results)
cfg = results.config;
x = results.input;
tracking = results.tracking;
nFft = min(cfg.signal.samplesPerEpoch, numel(x));
spectrum = fftshift(fft(x(1:nFft)));
frequencyMHz = ((-floor(nFft/2)):(ceil(nFft/2)-1)) * ...
    cfg.signal.fsHz / nFft / 1e6;
spectrumDb = 20*log10(abs(spectrum) / max(abs(spectrum)) + eps);

figure('Name', 'Galileo E5 Full-Band Spectrum', ...
    'Visible', cfg.plot.visible);
plot(frequencyMHz, spectrumDb, 'LineWidth', 1);
hold on;
xline(results.truth.e5aCenterHz/1e6, '--', 'E5a');
xline(results.truth.e5bCenterHz/1e6, '--', 'E5b');
xline(cfg.signal.ifHz/1e6, ':', 'E5 center IF');
hold off;
xlabel('Frequency relative to digital DC (MHz)');
ylabel('Normalized magnitude (dB)');
title('Complete AltBOC complex low-IF spectrum');
grid on;
xlim([-cfg.signal.fsHz, cfg.signal.fsHz] / 2e6);

epoch = 1:numel(tracking.prompt);
figure('Name', 'Galileo E5 Full-Band Tracking', ...
    'Visible', cfg.plot.visible);
tiledlayout(2, 2);
nexttile;
plot(epoch, abs(tracking.prompt), 'LineWidth', 1);
xlabel('Epoch (ms)');
ylabel('|Prompt|');
grid on;
nexttile;
plot(epoch, tracking.dllError, 'LineWidth', 1);
xlabel('Epoch (ms)');
ylabel('DLL error');
grid on;
nexttile;
plot(epoch, tracking.pllErrorRad, 'LineWidth', 1);
xlabel('Epoch (ms)');
ylabel('PLL error (rad)');
grid on;
nexttile;
plot(epoch, tracking.dopplerHz, 'LineWidth', 1);
hold on;
yline(results.truth.dopplerHz, '--', 'Truth');
hold off;
xlabel('Epoch (ms)');
ylabel('Doppler (Hz)');
grid on;
end
