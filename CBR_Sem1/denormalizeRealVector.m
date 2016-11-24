function [ vector ] = denormalizeRealVector( normalizedVector,vectorMin,vectorMax )
%DENORMALIZEREALVECTOR recovers the original data from a normalized vector,
%given the minimum and maximum of the original vector
%   normalizedVector - the array with the normalized values
%   vectorMin - the minimum value of the original vector
%   vectorMax - the maximum value of the original vector
%   vector - the original array of numeric values

    range = vectorMax - vectorMin;
    vector = vectorMin + normalizedVector.*range;


end

