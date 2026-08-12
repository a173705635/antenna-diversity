function metrics = compute_metrics(results, truth)
%COMPUTE_METRICS Build a compact comparison of all diversity branches.

modeNames = fieldnames(results.branches);
nModes = numel(modeNames);
acquired = false(nModes, 1);
acquisitionMetric = nan(nModes, 1);
estimatedDopplerHz = nan(nModes, 1);
jammerDirectionGain = nan(nModes, 1);
trackingLocked = false(nModes, 1);
finalDopplerHz = nan(nModes, 1);
pllRmsRad = nan(nModes, 1);
dllRms = nan(nModes, 1);
meanPromptMagnitude = nan(nModes, 1);
finalCn0DbHz = nan(nModes, 1);

for k = 1:nModes
    branch = results.branches.(modeNames{k});
    acquired(k) = branch.acquisition.success;
    acquisitionMetric(k) = branch.acquisition.metric;
    estimatedDopplerHz(k) = branch.acquisition.dopplerHz;
    if norm(truth.jammerSteeringNominal) > 0
        jammerDirectionGain(k) = abs(branch.combining.weights' * ...
            truth.jammerSteeringNominal);
    end
    trackingLocked(k) = isfield(branch.tracking, 'success') && ...
        branch.tracking.success;
    if trackingLocked(k)
        trk = branch.tracking;
        finalDopplerHz(k) = trk.carrierFrequencyHz(end);
        window = max(1, numel(trk.prompt)-9):numel(trk.prompt);
        pllRmsRad(k) = sqrt(mean(trk.pllErrorRad(window).^2));
        dllRms(k) = sqrt(mean(trk.dllError(window).^2));
        meanPromptMagnitude(k) = mean(abs(trk.prompt(window)));
        finiteCn0 = trk.cn0DbHz(isfinite(trk.cn0DbHz));
        if ~isempty(finiteCn0)
            finalCn0DbHz(k) = finiteCn0(end);
        end
    end
end

metrics.trueDopplerHz = truth.dopplerHz;
metrics.summaryTable = table(string(modeNames), acquired, acquisitionMetric, ...
    estimatedDopplerHz, jammerDirectionGain, trackingLocked, finalDopplerHz, ...
    pllRmsRad, dllRms, meanPromptMagnitude, finalCn0DbHz, ...
    'VariableNames', {'Mode', 'Acquired', 'AcquisitionMetric', ...
    'AcquiredDopplerHz', 'JammerDirectionGain', 'Tracking', ...
    'FinalDopplerHz', 'PllRmsRad', 'DllRms', 'MeanPromptMagnitude', ...
    'FinalCn0DbHz'});
end
