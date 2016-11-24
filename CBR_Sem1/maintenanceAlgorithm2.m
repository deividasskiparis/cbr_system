function [reducedCBx,reducedCBy]=maintenanceAlgorithm2(Xmatrix,Ymatrix,k)
% This function uses Edited k Nearest Neighbour EKNN algorithm to calculate
% the reduced case base. reducedCBx consists of points, which have been
% correctly classified using kNN method, where k is specified on input.
% Inputs:
%   Xmatrix - the original case-base
%   Ymatrix - the labels of the original case-base
%   k - integer specifying no of nearest neighbours.
% Outputs:
%   reducedCBx - Reduced case base
%   reducedCBy - Labels of the reduced case base

% Set default for k, if not specified
if nargin==2,k=3;end

% Check if k is greater than 0
if k<1
    k=3;
    disp(['"k" must be greater than 1. Searching with default k=',num2str(k)]);
end

% Search for k nearest neighbours for each point
Yidx=knnsearch(Xmatrix,Xmatrix,'K',k+1); % k+1, because the first k is always itself

% Remove the first column, as 1st k is always itself
Yidx=Yidx(:,2:end);

% Resolve KNN votes
Yasigned=resolveKNNVotes(Ymatrix(Yidx));

% Calculate a logical matrix, which is true, when asigned and true labels
% match
if isnumeric(Ymatrix)
    logicalM=Ymatrix==Yasigned;
else
    logicalM=strcmp(Ymatrix,Yasigned);
end

% Calculate the reduced case base
reducedCBx=Xmatrix(logicalM,:);
reducedCBy=Ymatrix(logicalM,:);

% Plot the data
plotMaintData(Xmatrix,Ymatrix,reducedCBx,reducedCBy);

end