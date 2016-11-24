function [ normalizedStruct, classVector, recoveryStruct ] = normalizeStructMOD(inputStruct)
% Normalizes all the numeric arrays in the given struct.
% inputStruct - a struct that has multiple structs, each of them having 2
%               fields - kind and values.
% 
% normalizedStruct - a struct with the same construction where the arrays
%                    of real numbers are normalized within range [0,1].
% 
% classVector - returns a cell containing the class labels for the
%               data points
% 
% recoveryStruct - a struct that recovers the transformed data, contains
%                  minimum and maximum data of the original struct
% 
% General idea for the normalization of a struct is as depicted in the
% decision tree below:
% 
%                 ___|'Class' or 'TrainTest' feature?|___
%                NO                                     YES 
%                |                                       |
%    |Type of 'kind' of field?|                *Return field as is*
%     |                     \  
%   CHAR            DOUBLE or STRUCT
%       \                           \
%   |What is the value?|              \ (Dealing with nominal values)
%       |               \               \
%   'numeric'          OTHERS             \
%       |                 |                 \
%  *Normalize*        *Do nothing*          *Split all nominal values into
%                                           *a separate feature, with
%                                           *values 1, where true and 0
%                                           *where false

% Initialize a recovery struct
recoveryStruct = inputStruct;

% Loop through all the fields in the struct
fields = fieldnames(inputStruct);
for i = 1:numel(fields)
    
    % Check if this is a 'Class' or 'Test-Train' attribute
    if strcmp('Class',fields{i}) || strcmp('class',fields{i})|| strcmp('CLASS',fields{i})
        classVector = inputStruct.(fields{i}).values';
    elseif ~isempty(regexp(fields{i},'[Tt]est.*[Tt]rain', 'once')) ...
            || ~isempty(regexp(fields{i},'[Tt]rain.*[Tt]est', 'once'))
        
        % Do nothing except rename the field to get consistency
        inputStruct.('TestTrain')=inputStruct.(fields{i});
        inputStruct=rmfield(inputStruct,fields{i});
    else
        if isa(inputStruct.(fields{i}).kind,'char')
            if strcmp(inputStruct.(fields{i}).kind,'numeric')
                % !Dealing with numerical values!
                
                % Normalize numerical attribute
                [inputStruct.(fields{i}).values, minValue, maxValue] = normalizeRealVector(inputStruct.(fields{i}).values);
                % Build recovery struct
                recoveryStruct.(fields{i}).values = [minValue,maxValue];
            end

        else
            % !Dealing with nominal values!
            
            % For every different nominal element create a feature in the struct    
            nominals=inputStruct.(fields{i}).kind;
            for n=1:length(nominals)
                
                % Make a field name for a new attribute
                if isa(nominals,'double')
                    % Nominal values are exressed in digits
                    newFieldName=num2str(nominals(n));
                    vals=double(inputStruct.(fields{i}).values==newFieldName);
                else
                    % Nominal values are exressed in strings
                    newFieldName=nominals{n};
                    vals=double(strcmp(inputStruct.(fields{i}).values,newFieldName));
                end
                
                % Create new attribute
                newFieldNameFull=[fields{i},newFieldName];
                inputStruct.(newFieldNameFull)=struct('kind','numeric','values',vals);

            end
            
            % Delete original field
            inputStruct=rmfield(inputStruct,fields{i});
            
        end
    end
end

normalizedStruct = inputStruct;
end
