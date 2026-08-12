function plot_results(results)
%PLOT_RESULTS Plot spectra, array responses, acquisition, and tracking states.

cfg = results.config;
visibility = cfg.plot.visible;

figure('Name', 'RF and DFE chain', 'Visible', visibility);
subplot(3,1,1);
plot_spectrum(results.sceneInputV(1,:), cfg.signal.fsRfHz, 'Antenna 1 input');
subplot(3,1,2);
plot_spectrum(results.adcCodes(1,:), cfg.signal.fsRfHz, 'ADC output');
subplot(3,1,3);
plot_spectrum(results.dfeChannels(1,:), results.dfe.info.fsOutHz, 'DFE output');

angles = -90:0.25:90;
figure('Name', 'Diversity array response', 'Visible', visibility);
hold on;
modeNames = fieldnames(results.branches);
for k = 1:numel(modeNames)
    w = results.branches.(modeNames{k}).combining.weights;
    response = zeros(size(angles));
    for q = 1:numel(angles)
        a = arraysim.steering_vector(cfg.array.positionsM, angles(q), ...
            cfg.signal.wavelengthM);
        response(q) = abs(w' * a);
    end
    responseDb = 20*log10(max(response, 1e-6));
    plot(angles, responseDb, 'DisplayName', upper(modeNames{k}));
end
xline(cfg.array.desiredAngleDeg, 'g--', 'Desired');
if cfg.jammer.enable
    xline(cfg.jammer.angleDeg, 'r--', 'Jammer');
end
grid on;
xlabel('Angle (deg)');
ylabel('Response (dB)');
ylim([-80, 10]);
legend('Location', 'best');
title('Nominal array response');

analysis.plot_acquisition(results);

figure('Name', 'Tracking comparison', 'Visible', visibility);
subplot(2,1,1);
hold on;
for k = 1:numel(modeNames)
    trk = results.branches.(modeNames{k}).tracking;
    if isfield(trk, 'success') && trk.success
        plot(trk.carrierFrequencyHz, 'DisplayName', upper(modeNames{k}));
    end
end
yline(results.truth.dopplerHz, 'k--', 'Truth');
grid on;
ylabel('Doppler (Hz)');
legend('Location', 'best');
title('Carrier tracking');
subplot(2,1,2);
hold on;
for k = 1:numel(modeNames)
    trk = results.branches.(modeNames{k}).tracking;
    if isfield(trk, 'success') && trk.success
        plot(abs(trk.prompt), 'DisplayName', upper(modeNames{k}));
    end
end
grid on;
xlabel('Tracking epoch (ms)');
ylabel('|Prompt|');
legend('Location', 'best');
title('Prompt correlation magnitude');
end

function plot_spectrum(x, fsHz, plotTitle)
x = x(:).';
nFft = min(65536, numel(x));
x = x(1:nFft);
window = 0.54 - 0.46*cos(2*pi*(0:nFft-1)/max(1,nFft-1));
spectrum = fftshift(fft(x .* window));
frequencyMHz = (-floor(nFft/2):ceil(nFft/2)-1) * fsHz / nFft / 1e6;
plot(frequencyMHz, 20*log10(max(abs(spectrum), eps)));
grid on;
xlabel('Frequency (MHz)');
ylabel('Magnitude (dB)');
title(plotTitle);
end
