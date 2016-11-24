function [ nominalVector ] = numericToNominal( numericalVector, nominalValues )
%NUMERICTONOMINAL recreates a nominal vector from a numerical vector and a
%vactor with all the possible nominal values. The numerical vector's values
%should be the to the nominalValues vector
%   numericalVector - a vector with numerical values (indices)
%   nominalValues - all the possible nominal values
%   nominalVector - the recreated vector with nominal values

for i = 1:length(nominalValues)
    for j = 1:length(numericalVector)
        if i == numericalVector(j)
            nominalVector(j) = nominalValues(i);
        end
    end
end

end

