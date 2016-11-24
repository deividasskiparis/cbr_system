
% Main script that will run ACBR algorithm.
clear;

%% Dataset and Retention Algorithm
% The desired dataset to be used - we used vowel.arff and vehicle.arff
dataset = 'vowel.arff';

% Define which Retention Algorithm to run
retentionAlgorithms = {'never','always','MG','DD'};
algorithm = retentionAlgorithms{4};

%% Initializations
% Variables
alpha=0.2;
K=[3,5,7];
tresh = 0.3;

% output initialization
accuracies = zeros(size(K));
reducRatio = zeros(size(K));
runTime = zeros(size(K));

%% Data preprocessing
% Get the normalized data from arff file
[data,labels,recoveryStruct] = parser_arff_file(dataset);

% Split the dataset into 10 random subsets for 10-fold cross validation
[xFoldStruct]=xFoldData(data,labels,10);
fprintf(strcat(dataset,' , ',algorithm,'\n'));

%% ACBR Testing
for i = 1:10  
    % 10-fold cross validation
    [dataTrain, classTrain, dataTest,classTest] = xFoldTester(xFoldStruct,i);
    fprintf(strcat('Fold: ',int2str(i),'\n'));
    
    origSize=size(dataTrain,1);
    
    % weighted retrieval weights
    W=0;
    
    % Perform Case base maintenance
%     [dataTrain,classTrain]=maintenanceAlgorithm1(dataTrain,classTrain);
%     [dataTrain,classTrain]=maintenanceAlgorithm2(dataTrain,classTrain);
    
    % For every k...
    for k = 1:length(K);
        
        
        tic;
        % ***** RUN ACBR *****
        [caseBaseDatamatrix,caseBaseLabels,errorCount,CM,W] = acbrAlgorithm(...
            dataTrain,classTrain,dataTest,K(k),algorithm,alpha,classTest,tresh,W);
        runTime(i,k)=toc/length(classTest);
        % Calculate accuracy and case-base reduction ratio
        accuracies(i,k) = 1-errorCount/length(classTest);
        reducRatio(i,k)=size(CM.currentCB.CBx,1)/origSize;

    end
end

%% Results Processing
% Averaging accuracies
meanAcc=mean(accuracies,1);
meanRR=mean(reducRatio,1);
meanRuntime=mean(runTime,1);
% Generate a table of results
results=table(K',{dataset;dataset;dataset},{algorithm;algorithm;algorithm},...
    meanAcc',meanRR',meanRuntime',...
    'VariableNames',{'K' 'DB' 'RET' 'ACC' 'RedRatio' 'RunTime'});
disp(results);
	