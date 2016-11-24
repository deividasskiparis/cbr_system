function [xFoldStruct]=xFoldData(data,labels,x)
% This function splits the data and corresponding labels into x equal 
% samples for x-fold testing.
% Output - struct with x-fold data

% Check if data sizes and types is correct
if size(data,1)~=size(labels,1)
    throw(MException('MatrixSize:Mismatch','Number of instances in label and data matrices do not match'))
end
% if mod(nargout,2)>0
%     throw(MException('Output:Wrong','Output is a ''data,label'' pair. Number of output arguments must be even'))
% end

xFoldStruct=struct;

splitSize=fix(size(data,1)/x);
for i=1:x
    
    if i==x
        sampleData=data;
        sampleLabels=labels;
    else
        [sampleData,idx]=datasample(data,splitSize,1,'Replace',false);
        sampleLabels=labels(idx);
        data(idx,:)=[];
        labels(idx)=[];
    end
    xFoldStruct.(['data',num2str(i)]).('data')=sampleData;
    xFoldStruct.(['data',num2str(i)]).('labels')=sampleLabels;
end
end