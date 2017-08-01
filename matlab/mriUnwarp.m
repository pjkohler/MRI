function mriUnwarp(dataDir)
    addpath(genpath('/Users/kohler/code/git/gardner/gru'));
    if ~iscell(dataDir)
        tempDir = {dataDir}; % wrap it
        clear dataDir;
        dataDir = tempDir;
    else
    end
    for d = 1:length(dataDir)
        if exist(dataDir{d},'dir')
            unwarp.EPIfiles = subfiles([dataDir{d},'/run*nii*']);
            unwarp.calfiles = subfiles([dataDir{d},'/*unwarp*nii*']);
            unwarp.EPIfiles = unwarp.EPIfiles(~ismember(unwarp.EPIfiles,unwarp.calfiles));
            % get rid of anatomy files
            anat = [subfiles([dataDir{d},'/*T1*']);...
                    subfiles([dataDir{d},'/*T2*']);...
                    subfiles([dataDir{d},'/*inplane*'])];
            for a = 1:length(anat)
                if ischar(anat{a})
                    unwarp.EPIfiles = unwarp.EPIfiles(~ismember(unwarp.EPIfiles,anat{a}));
                else
                end
            end            
            fsl_pe0pe1(dataDir{d},unwarp);
        else
            warning([dataDir{d}, ' does not exist']);
        end
        clear unwarp;
    end
end
