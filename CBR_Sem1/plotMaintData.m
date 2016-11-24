function plotMaintData(Xmatrix,Ymatrix,maintXmatrix,maintYmatrix)
% The function plots original data and data returned by maintenance
% algorithm, given the data is 2 dimensional.

if nargin<3
    maintXmatrix=NaN;
    maintYmatrix=NaN;
end

if size(Xmatrix,2)==2
    unc=unique(Ymatrix);
    figure
    hold on
    legendStr=cell(0,0);
    for n=1:size(unc,1)
        colour=rand(1,3);
        if isnumeric(unc(n))
            plot(Xmatrix(Ymatrix==unc(n),1),Xmatrix(Ymatrix==unc(n),2),...
                '.','MarkerFaceColor',colour,'MarkerEdgeColor',colour)
            if ~isnan(maintXmatrix)
                plot(maintXmatrix(maintYmatrix==unc(n),1),maintXmatrix(maintYmatrix==unc(n),2),...
                    's','MarkerFaceColor',colour,'MarkerEdgeColor',colour)
            end
        else
            plot(Xmatrix(strcmp(Ymatrix,unc{n}),1),Xmatrix(strcmp(Ymatrix,unc{n}),2),...
                '.','MarkerFaceColor',colour,'MarkerEdgeColor',colour)
            if ~isnan(maintXmatrix)
                plot(maintXmatrix(strcmp(maintYmatrix,unc{n}),1),maintXmatrix(strcmp(maintYmatrix,unc{n}),2),...
                    's','MarkerFaceColor',colour,'MarkerEdgeColor',colour)
            end
        end
        if isnumeric(unc(n))
            newLegStr=num2str(unc(n));
            if ~isnan(maintXmatrix)
                legendStr=[legendStr;newLegStr;newLegStr];
            else
                legendStr=[legendStr;newLegStr];
            end
        else
            newLegStr=unc(n);
            if ~isnan(maintXmatrix)
                legendStr=[legendStr;newLegStr;newLegStr];
            else
                legendStr=[legendStr;newLegStr];
            end
        end
    end
        legend(legendStr);
        hold off
end
end