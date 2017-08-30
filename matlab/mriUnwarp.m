function unwarp = mriUnwarp(dataDir,doUnwarp)
    %% FSL Combine EPIfiles and calfiles calibration scans (mux8 acquisitions)
    % Based on code from Bob Dougherty (CNI)
    % written by Dan Birman (2015-05), dbirman@stanford.edu
    % updated by pjkohler (2017-08), pjkohler@stanford.edu
    %
    % unwarp = fsl_pe0pe1(dataDir,doUnwarp)
    %
    %   dataDir = string or cell of strings, indicating one or more
    %             directory in which to do unwarping
    %
    %   doUnwarp = logical indicating whether to actually run unwarping
    %              [true]/false
    %   
    %
    % CODE FROM BOB:
    %
    % #!/bin/bash
    % 
    % # topup for rs data mux8 hcp resting state data:
    % 
    % # To compute the echo train length, run:
    % # fslhd rs_EPIfiles.nii.gz | grep desc
    % # and compute acq[0]*ec/1000
    % echo '0 1 0 0.05720' > acq_params.txt
    % echo '0 -1 0 0.05720' >> acq_params.txt
    % fslroi rs_EPIfiles.nii.gz bu 1 1
    % fslroi rs_calfiles.nii.gz bd 1 1
    % fslmerge -t bud bu bd
    % topup --imain=bud --datain=acq_param.txt --config=b02b0.cnf --out=rs_topup
    % applytopup --imain=rs_EPIfiles --inindex=1 --method=jac --datain=acq_param.txt --topup=rs_topup --out=rs0_unwarped
    % applytopup --imain=rs_calfiles --inindex=2 --method=jac --datain=acq_param.txt --topup=rs_topup --out=rs1_unwarped
    %
    % FSL Requests that we include the following text in any manuscripts that
    % use this function:
    % Brief summary text: "Data was collected with reversed phase-encode blips, resulting in pairs of images with distortions going in opposite directions. From these pairs the susceptibility-induced off-resonance field was estimated using a method similar to that described in [Andersson 2003] as implemented in FSL [Smith 2004] and the two images were combined into a single corrected one."
    % 
    % [Andersson 2003] J.L.R. Andersson, S. Skare, J. Ashburner How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. NeuroImage, 20(2):870-888, 2003.
    % 
    % [Smith 2004] S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J. Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak, D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano, J.M. Brady, and P.M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-219, 2004. 

    if nargin < 2
        doUnwarp = true;
    else
    end
    
    %% CHECK FOR FSL INSTALL
    [s,r] = system('fslroi');
    if s==127
        error('FSL may not be properly installed. Check your PATH');
    end
    
    %% CHECK FOR FILES AND DO UNWARPING
    if ~iscell(dataDir)
        tempDir = {dataDir}; % wrap it
        clear dataDir;
        dataDir = tempDir;
    else
    end
    for d = 1:length(dataDir)
        if exist(dataDir{d},'dir')
            unwarp{d}.EPIfiles = subfiles([dataDir{d},'/run*nii*']);
            unwarp{d}.calfiles = subfiles([dataDir{d},'/*unwarp*nii*']);
            unwarp{d}.EPIfiles = unwarp{d}.EPIfiles(~ismember(unwarp{d}.EPIfiles,unwarp{d}.calfiles));
            % get rid of anatomy files
            anat = [subfiles([dataDir{d},'/*T1*']);...
                    subfiles([dataDir{d},'/*T2*']);...
                    subfiles([dataDir{d},'/*inplane*'])];
            for a = 1:length(anat)
                if ischar(anat{a})
                    unwarp{d}.EPIfiles = unwarp{d}.EPIfiles(~ismember(unwarp{d}.EPIfiles,anat{a}));
                else
                end
            end  
            % check if we have multiple calfiles scans
            if length(unwarp{d}.calfiles) > 1
                disp('(fsl_pe0pe1) Multiple calfiles scans found, using last scan.\nYou can implement different functionality...');
                scanchoice = 0;
                while ~scanchoice
                    in = input('(fsl_pe0pe1) Use first or last scan? [f/l]','s');
                    if strcmp(in,'f')
                        unwarp{d}.calfiles = unwarp{d}.calfiles(1); scanchoice=1;
                    elseif strcmp(in,'l')
                        unwarp{d}.calfiles = unwarp{d}.calfiles(end); scanchoice=1;
                    else
                        in = input('(fsl_pe0pe1) Incorrect input. Use first or last scan? [f/l]','s');
                    end
                end
            end
            if doUnwarp
                fsl_pe0pe1(dataDir{d},unwarp{d});
            else
            end
        else
            warning([dataDir{d}, ' does not exist']);
        end
    end
end

function unwarp = fsl_pe0pe1(folder,unwarp)
    fprintf('(fsl_pe0pe1) Unwarping in %s\n',folder);
    tic
    
    % create temp folder
    tfolder = fullfile(folder,'temp');
    mkdir(tfolder);
    files = dir(folder);
    
    % make acq_params file
    acqFile = fullfile(folder,'acq_params.txt');
    system(sprintf('echo ''0 1 0 1'' > %s',acqFile));
    system(sprintf('echo ''0 -1 0 1'' >> %s',acqFile));


    %% CONVERT TO NII AND MAKE BACKUPS
    unwarp.EPIfiles = cellfun(@(x) hlpr_unzip(x,folder), unwarp.EPIfiles,'uni',false);
    
    str = '';
    str = strcat(str,'\n','**********************************************************');
    for i = 1:length(unwarp.EPIfiles)
        str = strcat(str,'\n',sprintf('Unwarping %s',unwarp.EPIfiles{i}));
    end
    str = strcat(str,'\n',sprintf('Using calibration file %s',unwarp.calfiles{1}));
    str = strcat(str,'\n','**********************************************************');
    str = sprintf(str);

    % If we get here we are unwarping!
    disp(str);
    poolH = gcp;
   
    %% RUN FSLROI

    % calfiles
    roi1files = {};
    roi1files{1} = hlpr_fslroi(unwarp.calfiles{1},1,1,1,1,tfolder,folder);
    % EPIfiles
    roi0files = {};
    disp('Calculating ROIs ...');
    drop = [];
    for i = 1:length(unwarp.EPIfiles)
        if ~isempty(strfind(unwarp.EPIfiles{i},'CAL'))
            disp('(fsl_pe0pe1) !!! For some reason you included a CAL file in with your EPIs. Ignoring...');
            drop = [drop i];
        else
            roi0files{i} = hlpr_fslroi(unwarp.EPIfiles{i},i,0,1,1,tfolder,folder);
        end
    end
    roi0files(drop) = [];
    if length(roi0files) < length(unwarp.EPIfiles)-length(drop)
        disp('(fsl_pe0pe1) Bug... check it out...');
        keyboard
    end

    %% RUN FLSMERGE
    mergefiles = {};
    parfor i = 1:length(roi0files)
        mergefiles{i} = hlpr_fslmerge(roi0files{i},roi1files{1},i,tfolder);
    end
    if length(mergefiles) ~= length(roi0files)
        disp('(fsl_pe0pe1) Returned file list too short...');
    end

    %% COMPUTE TOP-UP
    tufiles = {};
    parfor i = 1:length(mergefiles)
        tufiles{i} = hlpr_topup(mergefiles{i},i,tfolder,folder);    
    end

    if length(tufiles) ~= length(roi0files)
        disp('(fsl_pe0pe1) Returned file list too short...');
    end

    %% APPLY TOP-UP
    disp('applying top-up ...')
    finalfiles = {};
    parfor i = 1:length(tufiles)
        finalfiles{i} = hlpr_applytopup(tufiles{i},unwarp.EPIfiles{i},tfolder,folder);
    end

    if length(finalfiles) ~= length(roi0files)
        disp('(fsl_pe0pe1) Returned file list too short...');
    end

    %% RENAME AND CONVERT BACK TO .NII.GZ

    for i = 1:length(unwarp.EPIfiles)
        file = unwarp.EPIfiles{i};
        fileLoc = fullfile(folder,strcat(file,'.gz'));
        % uw_ file
        uw_fileLoc = fullfile(folder,strcat('uw_',file,'.gz'));
        % rename uw_ file
        s = movefile(uw_fileLoc,fileLoc,'f');
        if s == 0
            fprintf('File rename seems to have failed... check result for %s',fileLoc);
            keyboard
        end
        % remove input file (backup already made in unwarp_orig)
        system(sprintf('rm %s',fullfile(folder,file)));
        unwarp.EPIfiles{i} = strcat(file,'.gz');
    end

    %% CHECK THAT FILES EXIST
    for i = 1:length(unwarp.EPIfiles)
        file = unwarp.EPIfiles{i};
        fileLoc = fullfile(folder,file);
        if ~exist(fileLoc,'file')
            disp(sprintf('File %s did not unwarp!! All temp files remain: you can re-build the unwarp by hand.\n[dbcont] to continue.',fileLoc));
            keyboard
        end
    end

    %% DISPLAY RESULT
    delete(poolH);
    T = toc;
    fprintf('(fsl_pe0pe1) Unwarping completed successfully for %s\n',folder);
    fprintf('(fsl_pe0pe1) Elapsed time %04.2f s\n',T);
end


    
function outName = hlpr_unzip(fileName,folder)
    % make folder for orig files 
    if ~isdir(fullfile(folder,'unwarp_orig'))
        mkdir(fullfile(folder,'unwarp_orig'));
    end
    system(sprintf('cp %s/%s %s/unwarp_orig/%s',folder,fileName,folder,fileName));
    
    tempSplit = strsplit(fileName,'.');
    if strcmp(cat(2,tempSplit{end-1},'.',tempSplit{end}),'nii.gz')
        disp('.nii.gz file, converting ...');
        s = system(sprintf('gunzip %s/%s',folder,fileName));
        if s ~= 0 
            error('conversion error on file %s, aborting',fileName)
        else
        end
        outName = fileName(1:end-3);
    elseif strcmp(tempSplit{end},'nii')
        disp('.nii file, not converting');
        outName = fileName;
    else
        error('unknown suffix on file %s',fileName)
    end
end


function outfile = hlpr_applytopup(tu,orig,tfolder,folder)
    % applytopup --imain=rs_EPIfiles --inindex=1 --method=jac --datain=acq_param.txt --topup=rs_topup --out=rs0_unwarped
    prefix = orig(1:end-4);
    outfile = sprintf('uw_%s',prefix);
    outfull = fullfile(folder,outfile);
    if exist(outfull,'file')
        disp(sprintf('(applytopup) Unwarping has already been run: File %s exists, skipping.',outfile));
        return
    end
    disp(sprintf('(apply) apply for: %s',outfile));
    tu = fullfile(tfolder,tu);
    acqFile = fullfile(folder,'acq_params.txt');
    command = sprintf('applytopup --imain=%s --inindex=1 --method=jac --datain=%s --topup=%s --out=%s',fullfile(folder,prefix),acqFile,tu,outfull);
    hlpr_fsl(command,true);
end

function outfile = hlpr_topup(merge,pos,tfolder,folder)
    outfile = sprintf('topup_%02.0f%s',pos);
    outfull = fullfile(tfolder,outfile);
    if exist(outfull,'file')
        disp(sprintf('(topup) Topup already calculated: File %s exists, skipping.',outfile));
        return
    end
    disp(sprintf('(topup) topup for: %s',outfile));
    merge = fullfile(tfolder,merge);
    acqFile = fullfile(folder,'acq_params.txt');
    command = sprintf('topup --imain=%s --datain=%s --config=b02b0.cnf --out=%s',merge,acqFile,outfull);
    hlpr_fsl(command);
end

function outfile = hlpr_fslmerge(scan0,scan1,pos,folder)
    outfile = sprintf('merge_%02.0f',pos);
    outfull = fullfile(folder,outfile);
    if exist(outfull,'file')
        disp(sprintf('(merge) Merge already calculated: File %s exists, skipping.',outfile));
        return
    end
    disp(sprintf('(merge) Merge for: %s',outfile));
    scan0 = fullfile(folder,scan0);
    scan1 = fullfile(folder,scan1);
    command = sprintf('fslmerge -t %s %s %s',outfull,scan0,scan1);
    hlpr_fsl(command);
end

function outfile = hlpr_fslroi(scan,pos,type,n1,n2,tfolder,folder)
    outfile = sprintf('pe%i_%02.0f%s',type,pos);
    outfull = fullfile(tfolder,outfile);
    if exist(outfull,'file')
        disp(sprintf('(roi) ROI already calculated: File %s exists, skipping.',outfile));
        return
    end
    disp(sprintf('(roi) ROI for: %s',outfile));
    scan = fullfile(folder,scan);
    command = sprintf('fslroi %s %s %i %i',scan,outfull,n1,n2);
    hlpr_fsl(command);
end

function success = hlpr_fsl(command,compress)
    if nargin < 2
        compress = false;
    else
    end
    if compress
        success = system(sprintf('export FSLOUTPUTTYPE=NIFTI_GZ; %s', command));
    else
        success = system(sprintf('export FSLOUTPUTTYPE=NIFTI; %s', command));
    end
end