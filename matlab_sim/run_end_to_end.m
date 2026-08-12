function results = run_end_to_end(cfg)
%RUN_END_TO_END Simulate GPS L1 C/A from antenna RF input through tracking.

if nargin < 1 || isempty(cfg)
    cfg = default_config();
end

[sceneInputV, truth] = gnss.generate_scenario(cfg);
[adcCodes, rfInfo] = rf.process_frontend(sceneInputV, cfg.rf, cfg.randomSeed);
[dfeChannels, dfeInfo] = dfe.process_channels(adcCodes, cfg.dfe);

desiredSteering = truth.desiredSteeringNominal;
branches = struct();
for k = 1:numel(cfg.diversity.modes)
    mode = cfg.diversity.modes{k};
    [combined, combiningInfo] = diversity.combine(dfeChannels, ...
        desiredSteering, mode, cfg.diversity);
    acq = receiver.acquire(combined, dfeInfo.fsOutHz, ...
        cfg.signal.prn, cfg.acquisition);
    if acq.success
        trk = receiver.track(combined, dfeInfo.fsOutHz, cfg.signal.prn, ...
            acq, cfg.tracking, cfg.signal.carrierHz, cfg.signal.codeRateHz);
    else
        trk.success = false;
        trk.status = 'acquisition_failed';
        trk.prompt = complex([]);
    end
    branch.combined = combined;
    branch.combining = combiningInfo;
    branch.acquisition = acq;
    branch.tracking = trk;
    branches.(lower(mode)) = branch;
end

results.config = cfg;
results.truth = truth;
results.sceneInputV = sceneInputV;
results.adcCodes = adcCodes;
results.rf.info = rfInfo;
results.dfeChannels = dfeChannels;
results.dfe.info = dfeInfo;
results.branches = branches;
results.metrics = analysis.compute_metrics(results, truth);

disp(results.metrics.summaryTable);
if cfg.plot.enable
    analysis.plot_results(results);
end
end
