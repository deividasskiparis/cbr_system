% Maintenance algorith testing

clear

filename='vowel.arff';
[data,class]=parser_arff_file(filename);
[xFoldStruct]=xFoldData(data,class,10);

prct=[];
reductionRatio=[];
prctKNN=[];

for p=1:10
    [dataTrain,labelsTrain,dataTest,labelsTest]=xFoldTester(xFoldStruct,p);
    
%     [reducedCBx,reducedCBy]=maintenanceAlgorithm1(dataTrain,labelsTrain);
    [reducedCBx,reducedCBy]=maintenanceAlgorithm2(dataTrain,labelsTrain,3);
    idx=findKNN(reducedCBx,dataTest,1);
    predictedLabels=reducedCBy(idx);
    C=confusionmat(labelsTest,predictedLabels);
    
    % Get kNN results
    idx2=findKNN(data,dataTest,1+1);
    idx2(:,1)=[];
    predLabelsKNN=resolveKNNVotes(class(idx2));
    C2=confusionmat(labelsTest,predLabelsKNN);

    % Calculate performance:
    % Reduction ratio
    reductionRatio=[reductionRatio;round(size(reducedCBx,1)/size(dataTrain,1),3)]
    avgRR=mean(reductionRatio)
    % Efficiency
    prct=[prct;round(sum(diag(C))/sum(C(:)),3)]
    avgprct=mean(prct)
    % kNN efficiency
    prctKNN=[prctKNN;round(sum(diag(C2))/sum(C2(:)),3)]
    avgKNN=mean(prctKNN)

end
