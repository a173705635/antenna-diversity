function test_acquisition_plot()
%TEST_ACQUISITION_PLOT Verify acquisition search-map figure generation.

cfg = default_config();
cfg.plot.visible = 'off';
fs = cfg.dfe.fsOutHz;
samplesPerMs = round(fs * 1e-3);
n = 0:cfg.acquisition.noncoherentMs*samplesPerMs-1;
delaySamples = 600;
delayChips = delaySamples * cfg.signal.codeRateHz / fs;
code = gnss.ca_code(cfg.signal.prn);
phase = mod(n * cfg.signal.codeRateHz / fs - delayChips, 1023);
x = code(floor(phase) + 1) .* ...
    exp(1j * 2*pi * cfg.signal.dopplerHz * n / fs);

acq = receiver.acquire(x, fs, cfg.signal.prn, cfg.acquisition);
results.config = cfg;
results.branches.mvdr.acquisition = acq;

fig = analysis.plot_acquisition(results);
cleanup = onCleanup(@() close_if_valid(fig));
assert(isgraphics(fig, 'figure'), 'Plot function did not return a figure.');
assert(numel(findall(fig, 'Type', 'image')) >= 1, ...
    'Acquisition figure does not contain a two-dimensional heatmap.');
assert(numel(findall(fig, 'Type', 'line')) >= 3, ...
    'Acquisition figure does not contain the slice and peak markers.');

axesHandles = findall(fig, 'Type', 'axes');
sliceAxesFound = 0;
for k = 1:numel(axesHandles)
    if strcmp(axesHandles(k).YLabel.String, 'Normalized power (dB)')
        sliceAxesFound = sliceAxesFound + 1;
        titleText = axesHandles(k).Title.String;
        assert(iscell(titleText) && numel(titleText) == 2, ...
            'Acquisition detail title must use two lines to avoid clipping.');
    end
end
assert(sliceAxesFound == 1, 'Expected one acquisition detail axis.');

fprintf('PASS test_acquisition_plot\n');
end

function close_if_valid(fig)
if isgraphics(fig)
    close(fig);
end
end
