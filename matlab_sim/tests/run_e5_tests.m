%RUN_E5_TESTS Execute all standalone Galileo E5 full-band tests.

test_e5_codes();
test_e5_altboc();
test_e5_signal();
test_e5_tracking();

cfg = e5_default_config();
cfg.plot.enable = false;
cfg.signal.cn0DbHz = 55;
results = run_e5_fullband_tracking(cfg);
assert(results.tracking.success);
assert(isfield(results, 'summary'));
assert(abs(results.summary.finalDopplerErrorHz) < 20);

fprintf('PASS all Galileo E5 full-band tests\n');
