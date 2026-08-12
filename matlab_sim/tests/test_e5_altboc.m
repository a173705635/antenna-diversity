function test_e5_altboc()
%TEST_E5_ALTBOC Verify the complete ICD 8-PSK AltBOC replica.

cfg = e5_default_config();
knownData.e5aI = [1, -1, 1];
knownData.e5bI = [1, -1, -1, 1, 1, -1, 1, 1];
phase = cfg.signal.initialCodePhaseChips + ...
    (0:cfg.signal.samplesPerEpoch-1) * cfg.signal.codeRateHz / ...
    cfg.signal.fsHz;

[s, components] = galileo.altboc_replica(phase, cfg, knownData);
assert(isequal(size(s), size(phase)));
assert(max(abs(abs(s) - 1)) < 1e-12);
assert(abs(mean(abs(s).^2) - 1) < 1e-12);
assert(all(components.subcarrierSubperiod >= 0 & ...
    components.subcarrierSubperiod <= 7));
assert(all(components.e5aI == 1 | components.e5aI == -1));
assert(all(components.e5aQ == 1 | components.e5aQ == -1));
assert(all(components.e5bI == 1 | components.e5bI == -1));
assert(all(components.e5bQ == 1 | components.e5bQ == -1));

phaseStates = mod(round(angle(s) / (pi/4)), 8);
assert(numel(unique(phaseStates)) == 8);

assert_error(@() galileo.altboc_replica([], cfg, knownData), ...
    'galileo:altboc_replica:InvalidCodePhase');

fprintf('PASS test_e5_altboc\n');
end

function assert_error(action, expectedIdentifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier, expectedIdentifier));
    return;
end
error('test_e5_altboc:MissingError', ...
    'Expected error %s was not raised.', expectedIdentifier);
end
