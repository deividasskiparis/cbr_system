function [idx,retrievedCBx,retrievedCBy]=acbrRetrievalPhase(newCaseX,caseMemory,k)

CBx=caseMemory.currentCB.CBx;
CBy=caseMemory.currentCB.CBy;

% Find k nearest neighbours
idx=findKNN(CBx,newCaseX,k);

% Retrieve the cases from the case base
retrievedCBx=CBx(idx,:);
retrievedCBy=CBy(idx);

end