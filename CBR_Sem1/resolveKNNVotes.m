function decisionLabel=resolveKNNVotes(votes)
% This function is used to resolve the winner from a set of votes in
% 'votes'. Votes are categorical labels returned by KNN. The function uses
% scoring technique to take into account that first columns represent
% closest neighbours from KNN. Therefore in the case on conflict, priority
% is given to group of labels, which are closest to first column

if iscolumn(votes)
    votes=votes';
end

J=size(votes,2);
categrs=categorical(votes);
choices=categories(categrs);

scores=countcats(categrs,2);

for n=1:J % Loop through every column 
    % Asign a score to a label, which depends of which column it is in.
    scores=scores+countcats(categrs(:,n),2)*(J-n)*0.001;
end
[~,winnerIDs]=max(scores,[],2);
decisionLabel=choices(winnerIDs);


end