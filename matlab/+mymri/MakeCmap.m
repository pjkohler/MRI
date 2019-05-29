function cmap = MakeCmap(in_colors,out_name,noBlack,noSimple,noShuffle)
    if numel(in_colors) == 1
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
        cmap = distinguishable_colors(in_colors,bgColor);
        if ~noShuffle
            oddIdx = logical(mod(1:in_colors,2));
            cmap = [cmap(~oddIdx,:); cmap(oddIdx,:)];
            %cMap = cMap(randperm(in_colors),:);
        else
        end
    else
        cmap = in_colors;
    end
    cmap = [[1,1,1];cmap];
    cmap(:,4) = 0:(length(cmap)-1);
    dlmwrite(out_name,cmap,'delimiter','\t','precision',3)
    save([out_name,'.mat'],"cmap")
end
