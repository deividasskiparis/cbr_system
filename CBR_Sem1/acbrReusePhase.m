function [reusedLabel]=acbrReusePhase(retrievedKCBlabels)
% Voting - return the label with the maximum occurences

reusedLabel=resolveKNNVotes(retrievedKCBlabels);

end