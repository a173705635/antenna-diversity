function code = e5_secondary_code(component, svid)
%E5_SECONDARY_CODE Return the ICD secondary code assigned to SVID 1.

if ~isscalar(svid) || ~isfinite(svid) || svid ~= 1
    error('galileo:e5_secondary_code:UnsupportedSvid', ...
        'Only Galileo SVID 1 is supported by this validation.');
end

switch char(component)
    case 'e5aI'
        hexText = '842E9';
        codeLength = 20;
    case 'e5aQ'
        hexText = '83F6F69D8F6E15411FB8C9B1C';
        codeLength = 100;
    case 'e5bI'
        hexText = 'E';
        codeLength = 4;
    case 'e5bQ'
        hexText = 'CFF914EE3C6126A49FD5E5C94';
        codeLength = 100;
    otherwise
        error('galileo:e5_secondary_code:UnknownComponent', ...
            'Unknown Galileo E5 component: %s.', char(component));
end

values = hex2dec(cellstr(hexText.'));
bits = reshape((dec2bin(values, 4).' - '0'), 1, []);
code = 1 - 2 * bits(1:codeLength);
end
