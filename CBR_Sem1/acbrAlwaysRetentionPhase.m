function [caseMemory] = acbrAlwaysRetentionPhase(newCaseX, newCaseY, caseMemory)
% acbrAlwaysRetentionPhase - retaining algorithm for ACBR
% This algorithm does nothing except adding the new case to the memory
% newCaseX (Cnew) - current case base under test
% newCaseY (CSol) - expected label for this test case
% caseMemory - struct containing the case base as well as initial and
%              current goodness values
% *******************************************************************
% field1 = 'currentCaseBase';
% field2 = 'currentGoodness';
% field3 = 'initialGoodness';
% caseMemory = struct(field1,value1,field2,value2,field3,value3);
% 
% currentCB is a struct containing the case base
% field4 = 'CBx' %representing the datamatrix with the datapoints
% field5 = 'CBy' %representing the labels (classes) of the datapoints in
%                 CBx
% value1 = struct(field4,value4,field5,value5)
% *******************************************************************

caseMemory.currentCB.CBx = [caseMemory.currentCB.CBx;newCaseX];
caseMemory.currentCB.CBy = [caseMemory.currentCB.CBy;newCaseY];
caseMemory.goodness = [caseMemory.goodness;0.5];
caseMemory.initialGoodness = [caseMemory.initialGoodness;0.5];

end

