function [ updatedCaseMemory, kIndices ] = oblivionByGoodness( caseMemory, kIndices )
%OBLIVIONBYGOODNESS Summary of this function goes here
%   Detailed explanation goes here

updatedCaseMemory = caseMemory;

% Sometimes, if the case base has really few cases and the k is large the
% retrieved indeces of the k retrieved cases can be duplicated i.e.
% retrieve 2 times the same case. This can break the oblivion process so
% that case it is skipped.
if length(unique(kIndices)) ~= length(kIndices), return;end
    
for k = 1:length(kIndices)
    if caseMemory.goodness(kIndices(k)) < caseMemory.initialGoodness(kIndices(k))
        % Delete from case memory
        updatedCaseMemory.currentCB.CBx(kIndices(k),:) = [];
        updatedCaseMemory.currentCB.CBy(kIndices(k)) = [];
        updatedCaseMemory.goodness(kIndices(k)) = [];
        updatedCaseMemory.initialGoodness(kIndices(k)) = [];
        
        % Update k indices
        kIndices(kIndices>=kIndices(k)) = kIndices(kIndices>=kIndices(k)) - 1;
        kIndices(kIndices==0) = 1;
    end
end

end

