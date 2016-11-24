function [ numericalVector ] = nominalToNumeric( nominalVector, nominalValues )
%NOMINALTONUMERIC changes the input vector with nominal values to a vector
%with numerical values using the cell array given - the cellarray should
%contain all possible nominal values. Then every nominal value in the
%vector is changed to the index of its cell array
%   nominalVector - the vector containig nominal values
%   nominalValues - the cellarray containg every possible nominal value
%                   once
%   numericalVector - the numerical equivalent of the nominal vector

numericalVector = zeros(size(nominalVector));

for i = 1:length(nominalValues)
    for j = 1:length(nominalVector)
        if strcmp(nominalVector(j),nominalValues(i))
            numericalVector(j) = i;
        end
    end
end

end

