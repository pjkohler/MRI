function [outData,dataDims,surfData] = mriReadBrainData(inputFiles,surfData)
    if nargin < 2
        if strcmp(inputFiles{1}(end-10:end),'niml.dset') % assume surface data if niml
            surfData = true;
        else
            surfData = false;
        end
    else
    end
    for z=1:length(inputFiles)
        if ~surfData   
            tmp = NIfTI.Read(inputFiles{z});
            tmp.data = squeeze(tmp.data);
            if z==1
                dataDims = size(tmp.data);
                outData = nan([dataDims,length(inputFiles)]); %pre-allocate data
            else
                if sum(dataDims ~= size(tmp.data))>0; % check that data have the same dimensions
                    error('input file %.0d is different from first 1',z); 
                else
                end
            end
            if ndims(tmp.data) == 3
                outData(:,:,:,z) = tmp.data;
            else
                outData(:,:,:,:,z) = tmp.data;
            end
        else
            tmp = afni_niml_readsimple(inputFiles{z});
            if z==1
                dataDims = size(tmp.data);
                outData = nan([dataDims,length(inputFiles)]); %pre-allocate data
            else
                if sum(dataDims ~= size(tmp.data))>0; % check that data have the same dimensions
                    error('input file %.0d is different from first 1',z); 
                else
                end
            end
            if ndims(tmp.data) == 1
                outData(:,z) = tmp.data;
            else
                outData(:,:,z) = tmp.data;
            end
        end
    end
end
