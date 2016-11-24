function [ caseBaseDatamatrix, caseBaseLabels, errorCount, CM, W] = acbrAlgorithm(...
    initialCaseBaseDataMatrix, InitialCaseBaseLabels,testCasesDataMatrix,k,...
    retentionAlgorithm,alpha,validationLabels,threshold,W)
%ACBRALGORITHM applies the adaptive case-base reasoning algorithm on some
%test cases, returning the resulted case base with the labels and the
%number of errors if validation data is provided
%   initialCaseBaseDataMatrix - The initial data matix: each row is a
%                               sample point, each column a feature
%   InitialCaseBaseLabels - the labels for these initial data points, a
%                           vector or cell array
%   testCasesDataMatrix - a dtat matrix containing the points to be
%                         labeled by the algorithm
%   k - the number of neighbours to be considered for the KNN retrieval
%       algorithm
%   retentionAlgorithm - a string meaning the desired retention algorithm 
%                        to be used, possibilities are: "always",
%                        "DD","never","MG"
%   alpha - learning rate
%   validationLabels - the labels that the test case points should have
%   threshold - the threshold for degree of disagreement retenrion
%               algorithm
%   W - weights for weighted retrieval. If not provided, they are
%   calculated
%   caseBaseDatamatrix - the resulting data matrix after applying the
%                        algorithm
%   caseBaseLabels - the labels of the points in the resulting data matrix
%   errorCount - if validation data is provided, returns the number of
%                cases where the test case point label given by the
%                algorithm is different than the one in the validation
%                labels (i.e. is erronous)

% Dealing with optional parameters
if nargin < 7
    if strcmp(retentionAlgorithm,'DD')
        threshold = 0.2;
    end
end

validation = true;
if nargin < 6
    validation = false;
end

errorCount = NaN;
if validation
    errorCount = 0;
end

if nargin < 9
    W=0;
end


% ------- Constructing the case memory struct -------------
% the case memory contains:
%   - 'currentCB' that represents the current case base and is a struct
%     itself that contains the datapoints in the case base ('CBx') and the
%     labels ('CBy') of these points
%   - 'goodness' that reprezents the current goodness that is associated
%     with the datapoints
%   - 'initialGoodness' that reprezents the initial goodness of the
%     datapoints, that is 0.5
initialGoodnes = ones(size(InitialCaseBaseLabels))./2;

goodness = initialGoodnes;

if iscell(InitialCaseBaseLabels)
    CM = struct('currentCB',struct('CBx',initialCaseBaseDataMatrix,'CBy',...
    {InitialCaseBaseLabels}),'goodness',goodness,'initialGoodness',initialGoodnes);
else
    CM = struct('currentCB',struct('CBx',initialCaseBaseDataMatrix,'CBy',...
    InitialCaseBaseLabels),'goodness',goodness,'initialGoodness',initialGoodnes);
end

% Applying the ACBR for every point in the test case
for i = 1:size(testCasesDataMatrix,1)
    % The new test case
    cNew = testCasesDataMatrix(i,:);
    
    % ------- RETRIEVAL -----
    % using KNN in order to retrieve the cases from the case base
    % that are going to be reused
    [kInd,~,retrievedCBy] = acbrRetrievalPhase(cNew,CM,k);
    
    % ------- WEIGHTED RETRIEVAL --------
    % Uncomment to use weighted retrieval
    
    % Use Relieff method
    % [kInd,~,retrievedCBy,W]=weightedACBRRetrievalPhase...
    %       (cNew,CM,k,'relieff',W);

    % Use Sequential Feature Selection
    % [kInd,~,retrievedCBy,W]=weightedACBRRetrievalPhase...
    %   (cNew,CM,k,'seqfs',W);

    
    
    % ------- REUSE ---------
    % using voting for reusal of the cases, each case votes with a value of
    % 1 for its label the label with most votes wins and will be reused for
    % the test case
    cLabel = acbrReusePhase(retrievedCBy);
    
    
    % ------- REVISE --------
    % If validation data is provided the number of errors is counted
    if validation
        errorCount = errorCount + acbrRevisionPhase(validationLabels(i),cLabel);
    end
    
    % ------ REVIEW --------
    % Updating Goodness
    updatedKGoodness = updateGoodness(cLabel,retrievedCBy,CM.goodness(kInd),alpha);
    CM.goodness(kInd) = updatedKGoodness;
    
    % Forgetting Strategy - Oblivion by goodness
    [CM,kInd] = oblivionByGoodness(CM,kInd);
    
    
    % ------- RETENTION ---------
    % 4 retention algorithms possible, 2 trivial, 2 nontrivial:
    % 'never' - never adding the test case in case memory
    % 'always' - always adding the test case to the case memory
    % 'DD' - degree of disagreement retention strategy
    % 'MG' - Minimum goodness retention strategy
    if strcmp(retentionAlgorithm,'always')
        CM = acbrAlwaysRetentionPhase(cNew,cLabel,CM);
    elseif strcmp(retentionAlgorithm,'never')
        CM = acbrNoRetentionPhase(cNew,cLabel,CM);
    elseif strcmp(retentionAlgorithm,'DD')
        CM = acbrDegreeOfDisagreementRetentionPhase(cNew,cLabel,kInd,CM,threshold);
    elseif strcmp(retentionAlgorithm,'MG')
        CM = acbrMinimalGoodnessRetention( cNew,cLabel, kInd, CM );
    end
    
end

caseBaseDatamatrix = CM.currentCB.CBx;
caseBaseLabels = CM.currentCB.CBy;

end

