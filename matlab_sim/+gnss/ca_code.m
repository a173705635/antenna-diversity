function code = ca_code(prn)
%CA_CODE Generate a GPS L1 C/A PRN sequence as +/-1 chips.

taps = [ ...
    2 6; 3 7; 4 8; 5 9; 1 9; 2 10; 1 8; 2 9; ...
    3 10; 2 3; 3 4; 5 6; 6 7; 7 8; 8 9; 9 10; ...
    1 4; 2 5; 3 6; 4 7; 5 8; 6 9; 1 3; 4 6; ...
    5 7; 6 8; 7 9; 8 10; 1 6; 2 7; 3 8; 4 9];

if ~isscalar(prn) || prn ~= fix(prn) || prn < 1 || prn > size(taps, 1)
    error('gnss:ca_code:InvalidPrn', 'PRN must be an integer from 1 through 32.');
end

g1 = ones(1, 10);
g2 = ones(1, 10);
code = zeros(1, 1023);
tap1 = taps(prn, 1);
tap2 = taps(prn, 2);

for k = 1:1023
    bit = xor(g1(10), xor(g2(tap1), g2(tap2)));
    code(k) = 1 - 2 * double(bit);

    g1Feedback = xor(g1(3), g1(10));
    g2Feedback = xor(g2(2), xor(g2(3), xor(g2(6), ...
        xor(g2(8), xor(g2(9), g2(10))))));
    g1 = [g1Feedback, g1(1:9)];
    g2 = [g2Feedback, g2(1:9)];
end
end
