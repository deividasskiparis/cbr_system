function [data,classVector,recoveryStruct]=parser_arff_file(filename)
%This function reads an arff dataset and transforms it into a matrix that
%has on every column a feature, the last column the class and every row an
%instance. All numerical features have been normalized between [0,1] and
%all nominal features have been transformed to numerical and also normalized
%between [0,1], only the class feature has not been normalized.
%   filename - the filename of the .arff file that contains the data
%   matrix - the resulted matrix as explained in the description
%   recoveryStruct - a struct that is used to recover the data from the
%                    matrix, using matrixToStruct function

rawArffStruct = arffparser('read',filename);
[normalizedArffStruct, classVector, recoveryStruct] = normalizeStructMOD(rawArffStruct);

[data] = structToMatrix(normalizedArffStruct);

end

