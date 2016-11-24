function [reducedCBx,reducedCBy]=maintenanceAlgorithm1(Xmatrix,Ymatrix)
% This function uses condensed nearest neighbour rule to calculate the
% reduced case base. The points in the reduced case base for convenience
% are called prototypes. Prototypes are minimum point required to correctly
% identify all the points in the original Xmatrix
% Inputs:
%   Xmatrix - the original case-base
%   Ymatrix - the labels of the original case-base
% Outputs:
%   reducedCBx - Reduced case base
%   reducedCBy - Labels of the reduced case base

% Make working copies of the original case base and its labels
Xclone=Xmatrix;
Yclone=Ymatrix;

% Initialize the reduced case base
reducedCBx=zeros(1,size(Xmatrix,2));

% Asign values for the initial prototype. The first prototype is random
[reducedCBx(1,:),Yidx]=datasample(Xmatrix,1,1);
reducedCBy=Yclone(Yidx);

% Remove 1st prototype from the matrices
Yclone(Yidx)=[];
Xclone(Yidx,:)=[];

converged=false;

while ~converged
    
%     Reset the flag to true. If no new additions will be made, the
%     'while' loop will be able to terminate
        converged=true;
        
%     Loop through every remaining point in Xclone
    for i=1:size(Xclone,1)
        
%         Find the current points nearest prototype in reducedCBx        
        [Yidx,~]=findKNN(reducedCBx,Xclone(i,:),1);

%         Get the labels of the point and its nearest prototype
        Ypttp=reducedCBy(Yidx);
        Ypnt=Yclone(i);

%         Compare the labels
        if isnumeric(Ypttp)
            if Ypttp~=Ypnt
                noMatch=true;
            else
                noMatch=false;
            end
        else
            if ~strcmp(Ypttp,Ypnt)
                noMatch=true;
            else
                noMatch=false;
            end
        end
        
        if noMatch
            
%             If the labels don't match, make the current point a new
%             prototype and add it to XX and YY matrices.
            reducedCBx=[reducedCBx;Xclone(i,:)];
            reducedCBy=[reducedCBy;Yclone(i)];

%             Remove the new prototype from the original matrices
            Yclone(i)=[];
            Xclone(i,:)=[];

%             New addition has been made. The loop has to be restarted.
            converged=false;
            break
        end
    end
end

% Plot the data if the input has 2 dimensions
if size(Xmatrix,2)==2
    plotMaintData(Xmatrix,Ymatrix,reducedCBx,reducedCBy)
end
end