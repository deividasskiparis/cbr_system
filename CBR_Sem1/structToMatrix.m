function [matrixTrain,matrixTest,logTrain] = structToMatrix(struct)
% Transforms a struct with the data into a matrix. The matrix is split
% into 'Test and 'Train' matrices if:
% a) 'TestTrain' field exists.
% b) Both output variables are specified.
% Otherwise, one full matrix is returned
% 
% struct - a struct that has multiple structs, each of them having 2 fields
%          - kind and values. If kind is 'numeric' then the values should
%          be an array of real numbers.
% 
% matrixTest and matrixTrain - matrices that contain all the arrays from
%                              the struct.
    matrixTest = [];
    matrixTrain=[];
    
    fields = fieldnames(struct);
    sizeX=size(struct.(fields{1}).values,2);
    
    logTrain= true([1 sizeX]);
    logTest=false([1 sizeX]);
            
    maxV=max(strcmp('TestTrain',fields));
    if maxV>0
        % TestTrain column exists
        if nargout>1

            % Get logical matrix for Test and for Train
            logTrain=strcmp(struct.TestTrain.values,'Train');
            logTest=strcmp(struct.TestTrain.values,'Test');
        end
        struct=rmfield(struct,'TestTrain');
        fields = fieldnames(struct);
    end
    
    
    
    
    for i = 1:numel(fields)
        if ~(strcmp('Class',fields{i}) || strcmp('class',fields{i}))
            matrixTrain= [matrixTrain;struct.(fields{i}).values(logTrain)];
            matrixTest = [matrixTest;struct.(fields{i}).values(logTest)];
            
        end
    end
    matrixTrain = matrixTrain';
    matrixTest=matrixTest';
    
end