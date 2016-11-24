function [normalizedVector,vectorMin,vectorMax] = normalizeRealVector(vector)
% Normalizez the values in the vector to the interval [0,1]
% vector - an array of numeric values
% normalizedVector - the array with the normalized values
% vectorMin - the minimum value of the original vector
% vectorMax - the maximum value of the original vector
    vectorMin = min(vector);
    vectorMax = max(vector);
    range = vectorMax - vectorMin;
    normalizedVector = (vector - vectorMin)./(range);
end
