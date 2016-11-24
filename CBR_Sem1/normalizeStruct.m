function [ normalizedStruct, classVector, recoveryStruct ] = normalizeStruct(struct)
% Normalizes all the numeric arrays in the given struct.
% struct - a struct that has multiple structs, each of them having 2 fields
%          - kind and values. If kind is 'numeric' then the values should
%          be an array of real numbers.
% normalizedStruct - a struct with the same construction where the arays of
%                    real numbers are normalized (in [0,1]).
% classVector - returns a vector containing the class values for the
%               data points
% recoveryStruct - a struct that recovers the transformed data, contains
%                  for every numerical type fields a 'numerical' value of
%                  kind and minValue, maxValue for minMax and for every
%                  nominal only the kind changes where the string changes
%                  to the original cells TODO
    recoveryStruct = struct;
    fields = fieldnames(struct);
    for i = 1:numel(fields)
        if strcmp(struct.(fields{i}).kind,'numeric')
            [struct.(fields{i}).values, minValue, maxValue] = normalizeRealVector(struct.(fields{i}).values);
            recoveryStruct.(fields{i}).values = [minValue,maxValue];
        end
        if iscell(struct.(fields{i}).kind) && ~(strcmp('Class',fields{i}) || strcmp('class',fields{i}) )
            struct.(fields{i}).values = nominalToNumeric(struct.(fields{i}).values, struct.(fields{i}).kind);
            [struct.(fields{i}).values, minValue, maxValue] = normalizeRealVector(struct.(fields{i}).values);
            recoveryStruct.(fields{i}).values = [minValue,maxValue];
        end
        
        if strcmp('Class',fields{i}) || strcmp('class',fields{i})
            classVector = struct.(fields{i}).values';
        end
    end
    normalizedStruct = struct;
end
