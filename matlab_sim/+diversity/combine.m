function [y, info] = combine(X, desiredSteering, mode, cfg)
%COMBINE Apply single-channel, equal-gain, or loaded-MVDR combining.

if ~ismatrix(X) || isempty(X)
    error('diversity:combine:InvalidInput', ...
        'X must be a nonempty antennas-by-samples matrix.');
end
a = desiredSteering(:);
nAnt = size(X, 1);
if numel(a) ~= nAnt
    error('diversity:combine:DimensionMismatch', ...
        'desiredSteering must contain one value per antenna.');
end
if norm(a) == 0
    error('diversity:combine:ZeroSteering', ...
        'desiredSteering must be nonzero.');
end

R = (double(X) * double(X)') / size(X, 2);
loading = cfg.diagonalLoadingFactor * real(trace(R)) / nAnt;
if loading <= 0 || ~isfinite(loading)
    loading = eps(max(1, norm(R, 'fro')));
end
Rloaded = R + loading * eye(nAnt);

switch lower(char(mode))
    case 'single'
        if abs(a(1)) < eps
            error('diversity:combine:ZeroReferenceResponse', ...
                'The first antenna has zero desired response.');
        end
        w = complex(zeros(nAnt, 1));
        w(1) = 1 / conj(a(1));
    case 'egc'
        w = a / (a' * a);
    case 'mvdr'
        z = Rloaded \ a;
        denominator = a' * z;
        if abs(denominator) < eps
            error('diversity:combine:SingularConstraint', ...
                'Loaded covariance cannot support the desired constraint.');
        end
        w = z / denominator;
    otherwise
        error('diversity:combine:UnknownMode', ...
            'Unknown diversity mode: %s.', char(mode));
end

y = w' * X;
info.mode = lower(char(mode));
info.weights = w;
info.desiredResponse = w' * a;
info.weightNorm = norm(w);
info.conditionNumber = cond(Rloaded);
info.diagonalLoading = loading;
info.sampleCovariance = R;
info.loadedCovariance = Rloaded;
end
