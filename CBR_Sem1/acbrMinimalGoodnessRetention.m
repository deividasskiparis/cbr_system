function [ CM ] = acbrMinimalGoodnessRetention( cNew, cLabel, kInd, CM )
%ACBRMINIMALGOODNESSRETENTION is a retention algorithm based on the
%goodness of the retrieved cases and the current case
%   cNew - the test case
%   cLabel - the reused label for the new test case
%   kInd - the indeces of the retrieved k most simmilar cases
%   CM -  case memory, a struct containing the case base as well as initial
%         and current goodness values
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

% Get the majority class from K 
% This has already been done in the reuse phase, in our case, and the
% result should be the same as newCaseY
if iscell(CM.currentCB.CBy)
    nominalValues = unique(CM.currentCB.CBy(kInd));
    numrv = nominalToNumeric(CM.currentCB.CBy(kInd),nominalValues);
    majorityClassType = mode(numrv);
    majorityClassType = nominalValues(majorityClassType);
else
    majorityClassType = mode(CM.currentCB.CBy(kInd));
end

% Get the subset of set kInd which matches the majority class type
if isnumeric(majorityClassType)
    mk = kInd(CM.currentCB.CBy(kInd)==majorityClassType);
else
    mk = kInd(strcmp(CM.currentCB.CBy(kInd),majorityClassType));
end

% the maximum goodness of mk
g = max(CM.goodness(mk));
if iscell(CM.currentCB.CBy)
    majorityClassGoodness = CM.goodness(strcmp(CM.currentCB.CBy,majorityClassType));
else
    majorityClassGoodness = CM.goodness(CM.currentCB.CBy == majorityClassType);
end
% maximum goodness of majority class
gmax = max(majorityClassGoodness);
% minimum goodness of majority class
gmin = min(majorityClassGoodness);

% computing the threshold
threshold = (gmax + gmin)/2;

% adding cNew to the case base if the condition is met
if (g < threshold)
    CM.currentCB.CBx = [CM.currentCB.CBx;cNew];
    CM.currentCB.CBy = [CM.currentCB.CBy;cLabel];
    initCNewGoodness = max(CM.goodness(kInd));
    CM.goodness = [CM.goodness;initCNewGoodness];
    CM.initialGoodness = [CM.initialGoodness;initCNewGoodness];
end


end

