function [ originalStruct ] = matrixToStruct( dataMatrix, recoveryStruct, classVector )
%MATRIXTOSTRUCT recovers the data from a arff dataset reconstructing the
%original struct. It may change the class associated with the data points
%if a class vector is given
%   dataMatrix - the matrix that contains the normalized values of the data
%   recoveryStruct - a struct that is used to recreate the original data.
%                    This struct should be the one parser_arff_file or
%                    normalizeStruct functions return when applied to the
%                    original struct that you want to recreate
%   classVector - an optional parameter. If the original struct had
%                 asosiated classes then the recovery of the original
%                 struct will contain the classes without the add of a
%                 class vector. A class vector should be given only if the
%                 associated classes were changed
%   originalStruct - recoveres the original data, eventually with different
%                    classes if the class vector is given

originalStruct = recoveryStruct;
fields = fieldnames(recoveryStruct);

for i=1:size(dataMatrix,2)
    minValue = originalStruct.(fields{i}).values(1);
    maxValue = originalStruct.(fields{i}).values(2);
    originalStruct.(fields{i}).values = denormalizeRealVector(dataMatrix(:,i),minValue,maxValue);
    
    if iscell(originalStruct.(fields{i}).kind)
        originalStruct.(fields{i}).values = numericToNominal(originalStruct.(fields{i}).values, originalStruct.(fields{i}).kind);
    else
        originalStruct.(fields{i}).values = originalStruct.(fields{i}).values';
    end
    
    
end



if nargin > 2
    originalStruct.Class.values = classVector;
end

end
