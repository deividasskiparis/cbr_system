function [idx,retrievedCBx,retrievedCBy,W]=weightedACBRRetrievalPhase(newCaseX,CM,k,method,W)
% This function performs weighted retrieval according to 'method'
% using weights W provided. If W is not provided, it is calculated using
% the specified method.
% method = 'relieff' - Uses Relieff algorithm to return improtance of
%                      attributes in CBx and uses weighted KNN to 
%                      calculate nearest neighbours
%        = 'seqfs' - Uses Sequential feature selection method to reduce
%                    dimensionality of CBx

CBx = CM.currentCB.CBx;
CBy = CM.currentCB.CBy;

if nargin<5, W=NaN; end

% Check if weights have been provided and are valid
if size(CBx,2)~=size(W,2) || ~isrow(W)
    % Weights need calculating
    if strcmp(method,'relieff')
        fprintf('-----Performing Relieff feature evaluation-----\n');
        % Get weights using Relief-F
        [~,W]=relieff(CBx,CBy,10,'method','classification');
        fprintf('-----Relieff: Done-----\n');
    elseif strcmp(method,'seqfs')

        [xFoldStruct2]=xFoldData(CBx,CBy,10);
        [modelXtrain,modelYtrain,modelXtest,modelYtest]=xFoldTester(xFoldStruct2,randi([1 10]));

        opts = statset('display','iter');
        f=@(modelXtrain, modelYtrain, modelXtest, modelYtest)...
            (sum(~strcmp(modelYtest, classify(modelXtest, modelXtrain, modelYtrain,'diaglinear'))));
        W=sequentialfs(f,modelXtrain,modelYtrain,'options',opts,'direction','backward');
        W=double(W); 
    else
        return
    end
end

if strcmp(method,'relieff')
    % Find weighted k nearest neighbours
    idx=findKNN(CBx,newCaseX,k,W);

    % Retrieve the cases from the case base
    retrievedCBx=CBx(idx,:);
    retrievedCBy=CBy(idx);

    
elseif strcmp(method,'seqfs')


    % Columns numbers to consider
    col=(1:1:size(CBx,2)).*W;
    col=col(col>0);

    % Data to consider
    CBx=CBx(:,col);
    newCaseX=newCaseX(:,col);

    % Find k nearest neighbours
    idx=findKNN(CBx,newCaseX,k);

    % Retrieve the cases from the case base
    retrievedCBx=CBx(idx,:);
    retrievedCBy=CBy(idx);

    
else
    return
end
end