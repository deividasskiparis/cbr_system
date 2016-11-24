function [confMat,prct]=testMaintCaseBase(origCBx,origCBy,maintCBx,maintCBy)
% This function is used to test how good is the maintained case base. For
% every point in origCBx, it uses kNN algorithm to find its nearest
% prototype. It then asigns the label of the protytpe to the point. The
% function returns a confusion matrix of actual labels in origCBy to
% asigned labels.

% Inputs:
%   origCBx - original case base data points
%   origCBy - original case base labels
%   maintCBx - case base data points returned by a maintenance algorithm
%   maintCBy - labels for the maintained case base

% Outputs:
%   confMat - returns a confusion matrix of actual labels
%   prct - percentage of correctly identified points

% Initialize a matrix for asigned labels
if isnumeric(origCBy)
    asignedY=zeros(size(origCBy));
else
    asignedY=cell(size(origCBy));
end

% Return one nearest neighbour indeces
Yidx=knnsearch(maintCBx,origCBx,'K',1);

% Asign a value for each index from maintCBy
asignedY=maintCBy(Yidx,:);

% Calculate confusion matrix
confMat=confusionmat(origCBy,asignedY);

% Calculate the percentage
trueVal=sum(diag(confMat));
allVal=sum(sum(confMat));
prct=trueVal/allVal


end