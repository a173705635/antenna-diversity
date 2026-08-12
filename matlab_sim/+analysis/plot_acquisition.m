function fig = plot_acquisition(results)
%PLOT_ACQUISITION Plot 2-D search maps and best-Doppler code slices.

if ~isfield(results, 'config') || ~isfield(results.config, 'plot') || ...
        ~isfield(results.config.plot, 'visible')
    error('analysis:plot_acquisition:MissingPlotConfig', ...
        'results.config.plot.visible is required.');
end
if ~isfield(results, 'branches') || isempty(fieldnames(results.branches))
    error('analysis:plot_acquisition:MissingBranches', ...
        'results.branches must contain at least one acquisition result.');
end

modeNames = fieldnames(results.branches);
fig = figure('Name', 'Acquisition search results', ...
    'Visible', results.config.plot.visible, 'Color', 'w');
layout = tiledlayout(fig, numel(modeNames), 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:numel(modeNames)
    modeName = modeNames{k};
    if ~isfield(results.branches.(modeName), 'acquisition')
        error('analysis:plot_acquisition:MissingAcquisition', ...
            'Branch %s has no acquisition result.', modeName);
    end
    acq = results.branches.(modeName).acquisition;
    validate_acquisition(acq, modeName);

    peakScale = max([acq.peak; acq.searchPower(:); eps]);
    searchDb = 10*log10(max(acq.searchPower / peakScale, eps));
    searchDb = max(searchDb, -40);
    codePhaseChips = (0:size(acq.searchPower, 2)-1) / ...
        acq.samplesPerMs * 1023;
    peakChip = acq.codePhaseSamples / acq.samplesPerMs * 1023;

    mapAxes = nexttile(layout);
    imagesc(mapAxes, codePhaseChips, acq.dopplerBinsHz, searchDb);
    axis(mapAxes, 'xy');
    hold(mapAxes, 'on');
    plot(mapAxes, peakChip, acq.dopplerHz, 'rx', ...
        'MarkerSize', 10, 'LineWidth', 2);
    clim(mapAxes, [-40, 0]);
    colorbar(mapAxes);
    xlabel(mapAxes, 'Code phase (chips)');
    ylabel(mapAxes, 'Doppler (Hz)');
    title(mapAxes, sprintf('%s search map', upper(modeName)));

    [~, bestBin] = min(abs(acq.dopplerBinsHz - acq.dopplerHz));
    sliceDb = 10*log10(max(acq.searchPower(bestBin, :) / peakScale, eps));
    sliceDb = max(sliceDb, -40);
    sliceAxes = nexttile(layout);
    plot(sliceAxes, codePhaseChips, sliceDb, 'b-', 'LineWidth', 1);
    hold(sliceAxes, 'on');
    plot(sliceAxes, peakChip, 0, 'rx', 'MarkerSize', 10, ...
        'LineWidth', 2, 'DisplayName', 'Primary peak');
    if isfinite(acq.secondPeakCodePhaseSamples)
        secondPeakChip = acq.secondPeakCodePhaseSamples / ...
            acq.samplesPerMs * 1023;
        secondPeakDb = 10*log10(max(acq.secondPeak / peakScale, eps));
        plot(sliceAxes, secondPeakChip, secondPeakDb, 'o', ...
            'Color', [0.95, 0.45, 0.05], 'MarkerSize', 7, ...
            'LineWidth', 1.5, 'DisplayName', 'Second peak');
    end
    grid(sliceAxes, 'on');
    ylim(sliceAxes, [-40, 3]);
    xlabel(sliceAxes, 'Code phase (chips)');
    ylabel(sliceAxes, 'Normalized power (dB)');
    if acq.success
        statusText = 'PASS';
    else
        statusText = 'FAIL';
    end
    title(sliceAxes, { ...
        sprintf('%s: %s, ratio=%.2f', upper(modeName), statusText, acq.metric), ...
        sprintf('Doppler=%.0f Hz, code=%.2f chips', acq.dopplerHz, peakChip)});
end

title(layout, 'FFT acquisition: Doppler and code-phase search');
end

function validate_acquisition(acq, modeName)
required = {'searchPower', 'dopplerBinsHz', 'samplesPerMs', 'peak', ...
    'secondPeak', 'codePhaseSamples', 'secondPeakCodePhaseSamples', ...
    'dopplerHz', 'metric', 'success'};
for k = 1:numel(required)
    if ~isfield(acq, required{k})
        error('analysis:plot_acquisition:MissingField', ...
            'Acquisition branch %s is missing field %s.', modeName, required{k});
    end
end
if size(acq.searchPower, 1) ~= numel(acq.dopplerBinsHz) || ...
        isempty(acq.searchPower) || acq.samplesPerMs <= 0
    error('analysis:plot_acquisition:DimensionMismatch', ...
        'Acquisition search dimensions are inconsistent for branch %s.', modeName);
end
end
