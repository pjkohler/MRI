function [data,summary] = mriRoiFFT(DATAfiles,ROIfile,ROIlabel,nCycles,roiSelection,nHarm)
    %% SORT OUT ROI NAMES
    if exist(ROIlabel{1},'file')
        tempNames = lbl2list(ROIlabel{1});
    else
        tempNames = ROIlabel; % ROIlabels can either be a label file or a list of ROI
    end
    if nargin < 4
        nCycles = 10;
    else
    end
    if nargin < 5
        roiSelection = {'V1','V2','V3','V3A','V4','LOC','LOCv','MT'};
    else
    end
    if nargin < 6
        nHarm = 5;
    else
    end
    %% get data
    DV = {'d-','v-'};
    LR = {'-R','-L'};
    for z=1:length(DATAfiles)
        tempRoiData = mriRoiExtract(DATAfiles{z},ROIfile);
        roiCounter = 0;
        for r=1:length(roiSelection)
            if ~isempty(strfind(roiSelection{r},'_'))
                prefix = lower(roiSelection{r}(1:(strfind(roiSelection{r},'_'))));
            else
                prefix = '';
            end
            roiIdx = cell2mat(cellfun(@(x) ~isempty(strfind(lower(x),lower(roiSelection{r}))), tempNames,'uni',false));
            
            % sort out ambiguous ROI names

            switch lower(roiSelection{r})
               case [prefix,'v3']
                   nullIdx = logical(cell2mat(cellfun(@(x) ~isempty(strfind(lower(x),'v3a')),tempNames,'uni',false)) + ...
                             cell2mat(cellfun(@(x) ~isempty(strfind(lower(x),'v3b')),tempNames,'uni',false))); % take out the ones that are V3a or V3b
               case [prefix,'v3ab']
                   roiIdx = cell2mat(cellfun(@(x) ~isempty(strfind(lower(x),'v3a')),tempNames,'uni',false)); % include V3a and V3ab
                   nullIdx = false(1,length(tempNames));
               otherwise
                   nullIdx = false(1,length(tempNames));
            end            
            roiIdx = roiIdx.*~nullIdx;
            summary(r) = sum(roiIdx);
            
            roiIdx = find(roiIdx);
            
            if ~isempty(cell2mat(strfind({[prefix,'V1'],[prefix,'V2'],[prefix,'V3']},roiSelection{r})))
                numROIs = 7;
                if z==length(DATAfiles)
                    roiNames(roiCounter+(1:4))=cellfun(@(x) [roiSelection{r},x], {'d-R','d-L','v-R','v-L'},'uni',0);
                    roiNames(roiCounter+(5:6))=cellfun(@(x) [roiSelection{r},x], {'-R','-L'},'uni',0);
                    roiNames(roiCounter+7)=roiSelection(r);
                else
                end
            else
                numROIs = 3;
                if z==length(DATAfiles)
                    roiNames(roiCounter+(1:2))=cellfun(@(x) [roiSelection{r},x], {'-R','-L'},'uni',0);
                    roiNames(roiCounter+3)=roiSelection(r);
                else
                end
            end
            
            if ~isempty(roiIdx) && sum(length(tempRoiData)>=roiIdx)==length(roiIdx) % check if ROI even exists in file
                for subR= 1:length(roiIdx)
                    dvIdx = find(cell2mat(cellfun(@(x) ~isempty(strfind(lower(tempNames{roiIdx(subR)}),x)),lower(DV),'uni',0)));        
                    if isempty(dvIdx)
                        dvIdx = 1;
                    end
                    lrIdx = find(cell2mat(cellfun(@(x) ~isempty(strfind(lower(tempNames{roiIdx(subR)}),x)),lower(LR),'uni',0)));
                    if isempty(lrIdx)
                        lrIdx = 1;
                    end
                    subIdx = lrIdx+((dvIdx-1).*2);
                    if ~isempty(tempRoiData{roiIdx(subR)}) && length(tempRoiData)>=roiIdx(subR)
                        roiData{roiCounter+subIdx}(:,:,z) = tempRoiData{roiIdx(subR)};
                    else
                        roiData{roiCounter+subIdx} = [];
                    end
                end
                if numROIs==3;
                    if sum(arrayfun(@(x) isempty(roiData{roiCounter+x}),1:2))==0
                        roiData{roiCounter+3}(:,:,z) = [roiData{roiCounter+1}(:,:,z);roiData{roiCounter+2}(:,:,z)]; %BOTH
                    else
                        roiData{roiCounter+3} = [];
                    end
                elseif numROIs==7
                   tempVal = find(cell2mat(arrayfun(@(x) ~isempty(roiData{x}), roiCounter+(1:4),'uni',false)));
                    blTemp = []; lhTemp = []; rhTemp = [];
                    for t=1:length(tempVal)
                        blTemp = [blTemp;roiData{roiCounter+tempVal(t)}(:,:,z)];
                        if mod(tempVal(t),2) % if odd
                            rhTemp = [lhTemp;roiData{roiCounter+tempVal(t)}(:,:,z)]; % RIGHT
                        else
                            lhTemp = [lhTemp;roiData{roiCounter+tempVal(t)}(:,:,z)]; % LEFT
                        end
                    end
                    roiData{roiCounter+7}(:,:,z) = blTemp; clear blTemp;
                    roiData{roiCounter+5}(:,:,z) = rhTemp; clear rhTemp;
                    roiData{roiCounter+6}(:,:,z) = lhTemp; clear lhTemp;
                        
                    %roiData{roiCounter+7}(:,:,z) = [roiData{roiCounter+1}(:,:,z);roiData{roiCounter+2}(:,:,z);...
                    %                                     roiData{roiCounter+3}(:,:,z);roiData{roiCounter+4}(:,:,z)]; % BOTH
                    %roiData{roiCounter+5}(:,:,z) = [roiData{roiCounter+1}(:,:,z);roiData{roiCounter+3}(:,:,z)]; % RIGHT
                    %roiData{roiCounter+6}(:,:,z) = [roiData{roiCounter+2}(:,:,z);roiData{roiCounter+4}(:,:,z)]; % LEFT
                else
                end
            else
                roiData(roiCounter+(1:numROIs))={NaN};
            end
            roiCounter = roiCounter+numROIs;
        end
    end
    
    % place time in first dimension, to match new mriFFT function
    roiData = cellfun(@(x) permute(x,[2,1,3]),roiData,'uni',false);
    
    emptyCells = cell2mat(cellfun(@(x) isempty(x),roiData,'uni',0));
    roiData(emptyCells)={NaN};
    nanCells = cell2mat(cellfun(@(x) isnan(x(1)),roiData,'uni',0));
    templateIdx = find(~nanCells,1);
    if ~isempty(templateIdx)
        template_data = mriFFT(roiData{templateIdx},nCycles,nHarm,roiNames{templateIdx});
    else
        template_data = orderfields(struct('name',[],'zScore',NaN(1,nHarm),'SNR', NaN(1,nHarm),'angle',NaN, ...
                            'harmonics',NaN,'rawData',NaN,'meanCycle',NaN,...
                            'realSignal',NaN(1,nHarm),'realNoise',NaN(nHarm,4),'imagSignal',NaN(1,nHarm),'imagNoise',NaN(nHarm,4)));
    end
    for r = 1:length(roiNames)
        if ~isnan(roiData{r}(:,:,1))
            [data(r)] = mriFFT(roiData{r},nCycles,nHarm,roiNames{r});
        else
            % fill in ROI with NaNs
            fNames = fieldnames(template_data);
            for f = 1:length(fNames)
                if strcmp(fNames{f},'name')
                    data(r).(fNames{f}) = roiNames{r};
                else
                    data(r).(fNames{f}) =  single(NaN(size(template_data.(fNames{f}))));
                end
            end
        end
    end
end
    