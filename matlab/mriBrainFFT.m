function newHdr = mriBrainFFT(inputFiles,outputFile,nCycles)
    if nargin < 3
        nCycles = 10;
    else
    end
    %% LOAD DATA FILE(s) and average
    [inData,inSize,surfData] = mriNiftiRead(inputFiles);
    if surfData
        inData = permute(inData,[1,4,5,2,3]);
        % for surfaces, time is the second dimension and runs are the third
        % place time in dimension 4 and runs in dimension 5, just like volume data
    else
    end
    meanData = mean(inData,5); % average over runs
    clear inData; clear inSize;% free up memory

    %% COMPUTE FFT    
    outStrct = mriFFT(meanData,nCycles);
    
    %% ORGANIZE DATA
    if ~surfData
        voxSignal(:,:,:,1) = outStrct.zScore;
        voxSignal(:,:,:,2) = outStrct.norciaSNR;
        voxSignal(:,:,:,3) = outStrct.amplitude;
        voxSignal(:,:,:,4) = outStrct.phase;
        voxSignal(:,:,:,5) = outStrct.realSignal;
        voxSignal(:,:,:,6) = outStrct.imagSignal;

        %% PRINT NIFTI
        newNii = origNii;
        newNii = rmfield(newNii,{'data','ext'});
        newNii.data = voxSignal;
        newNii.hdr.dim(5) = size(voxSignal,4); % assign the number of t-dimensions
        NIfTI.Write(newNii,outputFile);
    else
        surfSignal(:,1) = squeeze(outStrct.zScore);
        surfSignal(:,2) = squeeze(outStrct.norciaSNR);
        surfSignal(:,3) = squeeze(outStrct.amplitude);
        surfSignal(:,4) = squeeze(outStrct.phase);
        surfSignal(:,5) = squeeze(outStrct.realSignal);
        surfSignal(:,6) = squeeze(outStrct.imagSignal);
        
        newLabels(1) = {'Z-SCORE'};
        newLabels(2) = {'SNR'};
        newLabels(3) = {'AMP'};
        newLabels(4) = {'PHASE'};
        newLabels(5) = {'REAL PART'};
        newLabels(6) = {'IMAG PART'};

        newStats = repmat({'none'},1,length(newLabels));

        %% PRINT NIFTI
        newStrct = struct();
        newStrct.data = surfSignal;
        newStrct.labels = newLabels;
        newStrct.stats = newStats;
        afni_niml_writesimple(outputFile,newStrct); 
        clear surfSignal;
    end
end


