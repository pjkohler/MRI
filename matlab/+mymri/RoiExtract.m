function roiData = mriRoiExtract(DATAfile,ROIfile,oldStyle)
    if nargin < 3
        oldStyle = false;
    else
    end
    if ~iscell(ROIfile)
        ROIfile = {ROIfile};
    else
    end
    if ~iscell(DATAfile)
        DATAfile = {DATAfile};
    else
    end
    if length(DATAfile)>1
        error('Only single datafiles allowed');
    else
    end
    if ~oldStyle
        if ~isempty(strfind(DATAfile{1},'nii'))
            tmp = NIfTI.Read(DATAfile{1});
            allData = squeeze(tmp.data);
            dim4 = size(allData,4);
        elseif ~isempty(strfind(DATAfile{1},'orig.BRIK'))
            [err, allData,info ] = BrikLoad(DATAfile{1});
            dim4 = size(allData,4);
        else
            error('DATA in unknown format')
        end
        for r=1:length(ROIfile)
            if ~isempty(strfind(ROIfile,'nii'));
                tmp = NIfTI.Read(ROIfile{r});
                roiMat = tmp.data;
            elseif ~isempty(strfind(ROIfile,'orig.BRIK'))
                [err, roiMat,info ] = BrikLoad(ROIfile);
            else
                error('ROI in unknown format')
            end    
            roiMat = repmat(roiMat,[1,1,1,dim4]); 
            roiMax = max(unique(roiMat));
            for roiIdx = 1:roiMax
                if ~isempty(find(roiMat==roiIdx, 1))
                    tmpData = allData(roiMat==roiIdx);
                    tmpData = reshape(tmpData,length(tmpData)/dim4,dim4);
                    roiData(roiIdx)={tmpData};
                else
                end
            end
        end
    else
        dataIdx = 0;
        for r=1:length(ROIfile)
            system(['3dBrickStat -max -slow ',ROIfile{r},'>maxout']);
            roiMax = round(load('maxout'));
            system('rm maxout');
            for roiIdx = 1:roiMax
                dataIdx = dataIdx+1;
                status=system(['3dmaskdump -noijk -quiet -cmask ''-a ',ROIfile{r},' -expr ispositive(0.1-abs(a-',num2str(roiIdx),'))'' ',DATAfile{1},'>tempout']);
                if status == 0
                    roiData(dataIdx)={load('tempout')};
                else
                end
                system('rm tempout');
            end
        end    
    end
    if ~exist('roiData','var')
        roiData = NaN;
    else
    end
end

