function code = e5_primary_code(component, svid)
%E5_PRIMARY_CODE Generate one SVID 1 Galileo E5 primary code.

if ~isscalar(svid) || ~isfinite(svid) || svid ~= 1
    error('galileo:e5_primary_code:UnsupportedSvid', ...
        'Only Galileo SVID 1 is supported by this validation.');
end

switch char(component)
    case 'e5aI'
        register1TapsOctal = '40503';
        register2TapsOctal = '50661';
        register2StartOctal = '30305';
    case 'e5aQ'
        register1TapsOctal = '40503';
        register2TapsOctal = '50661';
        register2StartOctal = '25652';
    case 'e5bI'
        register1TapsOctal = '64021';
        register2TapsOctal = '51445';
        register2StartOctal = '07220';
    case 'e5bQ'
        register1TapsOctal = '64021';
        register2TapsOctal = '43143';
        register2StartOctal = '03331';
    otherwise
        error('galileo:e5_primary_code:UnknownComponent', ...
            'Unknown Galileo E5 component: %s.', char(component));
end

registerLength = 14;
codeLength = 10230;
register1 = ones(1, registerLength);
register2 = bitget(base2dec(register2StartOctal, 8), 1:registerLength);
register1Taps = bitget(base2dec(register1TapsOctal, 8), ...
    2:registerLength+1);
register2Taps = bitget(base2dec(register2TapsOctal, 8), ...
    2:registerLength+1);
logicCode = false(1, codeLength);

for chip = 1:codeLength
    logicCode(chip) = xor(register1(registerLength), ...
        register2(registerLength));
    feedback1 = mod(sum(register1 .* register1Taps), 2);
    feedback2 = mod(sum(register2 .* register2Taps), 2);
    register1 = [feedback1, register1(1:registerLength-1)];
    register2 = [feedback2, register2(1:registerLength-1)];
end

code = 1 - 2 * double(logicCode);
end
