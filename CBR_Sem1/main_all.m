
% Main script that will run ACBR algorithm for all algorithm for 20 times
% and will make an average of the accuracies for each algorith
clear;
%Initialize dataset and variables
dataset = 'vowel.arff';
alpha=0.2;
K=[3,5,7];
tresh = 0.2;
accuracies = zeros(size(K));
reducRatio = zeros(size(K));
runTime = zeros(size(K));

% Define which Retention Algorithm to run
retentionAlgorithms = {'never','always','MG','DD'};
algorithm = retentionAlgorithms{4};

% Get the normalized data from arff file
[data,labels,recoveryStruct] = parser_arff_file(dataset);
a = struct('never',0,'always',0,'MG',0,'DD',0);
nberas = 20;
for era=1:nberas
    % Split the dataset into 10 random subsets for 10-fold cross validation
    [xFoldStruct]=xFoldData(data,labels,10);
    fprintf(strcat(dataset,' , ',algorithm,'\n'));

    r = struct;
    for alg = retentionAlgorithms
        algorithm = char(alg);
        for i = 1:10  
            % 10-fold cross validation
            [dataTrain, classTrain, dataTest,classTest] = xFoldTester(xFoldStruct,i);
            fprintf(strcat('Era: ',int2str(era),'Alg: ',algorithm,' Fold: ',int2str(i),'\n'));

            origSize=size(dataTrain,1);

            % Reset weights for weighted retrieval if used.
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

        % Averaging accuracies
        meanAcc=mean(accuracies,1);
        meanRR=mean(reducRatio,1);
        meanRuntime=mean(runTime,1);
        
        a.(algorithm) = a.(algorithm) + meanAcc(1);
        
        % Generate a table of results
        results=table(K',{dataset;dataset;dataset},{algorithm;algorithm;algorithm},...
            meanAcc',meanRR',meanRuntime',...
            'VariableNames',{'K' 'DB' 'RET' 'ACC' 'RedRatio' 'RunTime'});
        r.(algorithm) = results;
    end
end
a.never = a.never/nberas;
a.always = a.always/nberas;
a.MG = a.MG/nberas;
a.DD = a.DD/nberas;


	