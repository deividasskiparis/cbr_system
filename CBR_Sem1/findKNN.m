function [idx,D]=findKNN(X,Y,k,W)
% This function uses Euclidean distance to return k nearest neighbours for
% every Y in X. If W is provided, wegihted distance is returned.

% X and Y horizontal (feature) dimensions should match
if size(X,2)~=size(Y,2),return;end

if nargin<4
    W=NaN;
elseif size(W)~=size(X(1,:))
    W=NaN;
end

% Normalize the weights if provided
if ~isnan(W),W=(W-min(W))/(max(W)-min(W));end

% Initialize working matrix Euclidian distance
euclDist=zeros(size(X,1),size(Y,1));

for n=1:size(Y,1) % For every instance in Y
    % Calculate the elemental difference
    difference=X-repmat(Y(n,:),[size(X,1) 1]);
    
    % Caculate the square root of sum of differences^2
    if isnan(W)
        euclDist(:,n)=sqrt(sum(difference.^2,2));
    else
        euclDist(:,n)=sqrt(sum((difference.^2).*repmat(W,[size(X,1) 1]),2));
    end
end
euclDist=euclDist';

% Initialize index (idx) and distance (D) matrices
idx=zeros(size(Y,1),k);
D=idx;

for m=1:k % For every k
    % Find the nearest point, i.e. minimum distance
    [~,indices]=min(euclDist,[],2);
    
    % Update index matrix
    idx(:,m)=indices;
    
    % Convert matrix indices to linear indices
    reference=sub2ind(size(euclDist),(1:size(indices,1))',indices);
    
    % Return the distances, corresponding to the indices
    D(:,m)=euclDist(reference);
    
    % Remove the instances from the data
    euclDist(reference)=NaN;
end

end