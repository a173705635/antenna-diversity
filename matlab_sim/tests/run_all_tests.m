%RUN_ALL_TESTS Execute all RF-to-tracking regression tests.

test_core();
test_rf();
test_dfe();
test_diversity();
test_acquisition();
test_acquisition_plot();
test_tracking();
test_end_to_end();
fprintf('PASS all MATLAB simulation tests\n');
