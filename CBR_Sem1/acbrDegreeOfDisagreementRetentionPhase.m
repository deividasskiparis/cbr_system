function [caseMemory,kInd] = acbrDegreeOfDisagreementRetentionPhase(newCaseX, newCaseY, kInd, caseMemory,threshold)
% acbrAlwaysRetentionPhase - retaining algorithm for ACBR
% This algorithm adds new case to the memory if it is above a 'threshold'
% value. By default threshold=0.3
%
% newCaseX (Cnew) - current case base under test
% newCaseY (CSol) - expected label for this test case
% kInd - the indeces of the retrieved k cases over wich the case is
%        retained. Default: 0.3
% threshold - the degree of disagreement 
% caseMemory - struct containing the case base as well as initial and
%              current goodness values
% *******************************************************************
% field1 = 'currentCB';
% field2 = 'goodness';
% field3 = 'initialGoodness';
% caseMemory = struct(field1,value1,field2,value2,field3,value3);
% 
% currentCB is a struct containing the case base
% field4 = 'CBx' %representing the datamatrix with the datapoints
% field5 = 'CBy' %representing the labels (classes) of the datapoints in
%                 CBx
% value1 = struct(field4,value4,field5,value5)
% *******************************************************************

% setting default value for threshold
if nargin < 5
    threshold = 0.3;
end

% The number of classes
numberOfClasses = length(unique(caseMemory.currentCB.CBy));

% Get the majority class from K 
% This has already been done in the reuse phase, in our case, and the
% result should be the same as newCaseY
majorityClassType=resolveKNNVotes(caseMemory.currentCB.CBy(kInd));

% Get the subset of set K which matches the majority class type
mk = caseMemory.currentCB.CBx(kInd,:);
if isnumeric(majorityClassType)
    mk = mk(caseMemory.currentCB.CBy(kInd)==majorityClassType);
else
    mk = mk(strcmp(caseMemory.currentCB.CBy(kInd),majorityClassType));
end

% Calculate degree of disagreement
d = (length(kInd) - size(mk,1))/((numberOfClasses -1)*size(mk,1));

if d >= threshold
   % Add Cnew and its goodness to the case memory
    caseMemory.currentCB.CBx = [caseMemory.currentCB.CBx;newCaseX];
    caseMemory.currentCB.CBy = [caseMemory.currentCB.CBy;newCaseY];
    initCNewGoodness = max(caseMemory.goodness(kInd));
    caseMemory.goodness = [caseMemory.goodness;initCNewGoodness];
    caseMemory.initialGoodness = [caseMemory.initialGoodness;initCNewGoodness];
end

end

