function [ updatedKGoodness ] = updateGoodness( newCaseClass, kCasesClasses, kCurrentGoodness, alpha)
%UPDATEGOODNESS updates the goodness measure given the k Cases retrieved
%classes and the new cases's class and the current goodness of the k cases
%   newCaseClass - the found class for the new case
%   kCasesClasses - the classes of the k retrieved cases
%   kCurrentGoodness - the goodness of the k cases
%   alpha - the learning rate
%   updatedKGoodness - the updated goodness of the k cases

updatedKGoodness = kCurrentGoodness;
for k = 1:length(kCasesClasses)
    r = 0;
    if isnumeric(newCaseClass)
        if newCaseClass == kCasesClasses(k)
            r = 1;
        end
    else
        if strcmp(newCaseClass, kCasesClasses(k))
            r = 1;
        end
    end
    t1 = r - kCurrentGoodness(k);
    t2 = alpha*t1;
    updatedKGoodness(k) = kCurrentGoodness(k) + t2;
    
end

end

