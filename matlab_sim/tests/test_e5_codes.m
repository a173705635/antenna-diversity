function test_e5_codes()
%TEST_E5_CODES Verify SVID 1 E5 codes against independent ICD literals.

cfg = e5_default_config();
assert(cfg.signal.fsHz == 132e6);
assert(abs(cfg.signal.ifHz - 3.795e6) < 1e-9);
assert(cfg.signal.samplesPerEpoch == 132000);

components = {'e5aI', 'e5aQ', 'e5bI', 'e5bQ'};
initialHex = {'3CEA9D', '515537', 'C5BEA1', 'E49AF0'};
secondaryHex = { ...
    '842E9', ...
    '83F6F69D8F6E15411FB8C9B1C', ...
    'E', ...
    'CFF914EE3C6126A49FD5E5C94'};
secondaryLengths = [20, 100, 4, 100];

for k = 1:numel(components)
    primary = galileo.e5_primary_code(components{k}, 1);
    assert(isequal(size(primary), [1, 10230]));
    assert(all(primary == 1 | primary == -1));
    expectedPrimary = hex_to_signal(initialHex{k}, 24);
    assert(isequal(primary(1:24), expectedPrimary));

    secondary = galileo.e5_secondary_code(components{k}, 1);
    assert(numel(secondary) == secondaryLengths(k));
    assert(isequal(secondary, ...
        hex_to_signal(secondaryHex{k}, secondaryLengths(k))));
end

assert_error(@() galileo.e5_primary_code('bad', 1), ...
    'galileo:e5_primary_code:UnknownComponent');
assert_error(@() galileo.e5_primary_code('e5aI', 2), ...
    'galileo:e5_primary_code:UnsupportedSvid');

fprintf('PASS test_e5_codes\n');
end

function signal = hex_to_signal(hexText, lengthBits)
values = hex2dec(cellstr(hexText.'));
bits = reshape((dec2bin(values, 4).' - '0'), 1, []);
signal = 1 - 2 * bits(1:lengthBits);
end

function assert_error(action, expectedIdentifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier, expectedIdentifier));
    return;
end
error('test_e5_codes:MissingError', ...
    'Expected error %s was not raised.', expectedIdentifier);
end
