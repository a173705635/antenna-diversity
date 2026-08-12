function a = steering_vector(positionsM, angleDeg, wavelengthM)
%STEERING_VECTOR Return the ULA response for broadside-referenced angle.

if ~isvector(positionsM) || isempty(positionsM)
    error('arraysim:steering_vector:InvalidPositions', ...
        'positionsM must be a nonempty vector.');
end
if ~isscalar(angleDeg) || ~isfinite(angleDeg)
    error('arraysim:steering_vector:InvalidAngle', ...
        'angleDeg must be a finite scalar.');
end
if ~isscalar(wavelengthM) || ~isfinite(wavelengthM) || wavelengthM <= 0
    error('arraysim:steering_vector:InvalidWavelength', ...
        'wavelengthM must be a positive finite scalar.');
end

a = exp(-1j * 2*pi * positionsM(:) * sind(angleDeg) / wavelengthM);
end
