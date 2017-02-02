function cMap = mriMakeCmap(numColors,outName,noBlack,noSimple,noShuffle)    
    if ~exist('distinguishable_colors','file')
        codeFolder = '/Users/kohler/code';
        if exist([codeFolder,'/matlab/others/plotTools'],'dir')
            addpath(genpath([codeFolder,'/matlab/others/plotTools']));
        else
            error('distinguishable_colors.m not on path, please add');
        end
    else
    end
    if noBlack
        bgColor = [1 1 1; 0 0 0; .5 .5 .5];
    else
        bgColor = [1 1 1];
    end
    if noSimple
        bgColor = [bgColor; eye(3); [1 1 0] ];
    else
    end
    cMap = distinguishable_colors(numColors,bgColor);
    if ~noShuffle
        oddIdx = logical(mod(1:numColors,2));
        cMap = [cMap(~oddIdx,:); cMap(oddIdx,:)];
        %cMap = cMap(randperm(numColors),:);
    else
    end
    cMap = [[1,1,1];cMap];
    cMap(:,4) = 0:numColors;
    dlmwrite(outName,cMap,'delimiter','\t','precision',3)
end
