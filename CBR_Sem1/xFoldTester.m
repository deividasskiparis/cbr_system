function [dataTrain,labelsTrain,dataTest,labelsTest]=xFoldTester(xFoldStruct,n)
% From xFoldStruct, this function returns n'th data partition as a test 
% matrix and other data fields as a training matrix.
% Use with For loop for automatic x-fold testing

fieldN=fieldnames(xFoldStruct);
x=size(fieldN,1);

dataTest=xFoldStruct.(fieldN{n}).data;
labelsTest=xFoldStruct.(fieldN{n}).labels;
dataTrain=[];
labelsTrain=[];
for i=1:x
    if i~=n
        dataTrain=[dataTrain;xFoldStruct.(fieldN{i}).data];
        labelsTrain=[labelsTrain;xFoldStruct.(fieldN{i}).labels];
    end
end
end