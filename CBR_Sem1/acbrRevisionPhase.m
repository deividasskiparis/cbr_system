function errorCount=acbrRevisionPhase(trueLabel,predLabel)
% This function is a simple label comparison to evaluate if the reused
% label(s) is correct label(s).

if iscell(trueLabel)
    check=~strcmp(trueLabel,predLabel);
else
    check=trueLabel~=trueLabel;
end
errorCount = sum(check(:));
end