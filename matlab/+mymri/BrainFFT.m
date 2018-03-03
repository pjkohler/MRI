function newHdr = mriBrainFFT(inputFiles,outputFile,nCycles)
    if nargin < 3
        nCycles = 10;
    else
    end
    %% LOAD DATA FILE(s) and average
    [inData,inStrct,surfData] = mriReadBrainData(inputFiles);
    if surfData
        inData = permute(inData,[1,4,5,2,3]);
        % for surfaces, time is the second dimension and runs are the third
        % place time in dimension 4 and runs in dimension 5, just like volume data
    else
    end
    meanData = mean(inData,5); % average over runs
    clear inData; clear inSize;% free up memory

    %% COMPUTE FFT    
    dataStrct = mriFFT(meanData,nCycles);
    
    %% ORGANIZE DATA
    if ~surfData
        voxSignal(:,:,:,1) = dataStrct.zScore;
        voxSignal(:,:,:,2) = dataStrct.SNR;
        voxSignal(:,:,:,3) = dataStrct.amplitude;
        voxSignal(:,:,:,4) = dataStrct.phase;
        voxSignal(:,:,:,5) = dataStrct.realSignal;
        voxSignal(:,:,:,6) = dataStrct.imagSignal;

        %% PRINT NIFTI
        outStrct = inStrct{1};
        outStrct = rmfield(outStrct,'ext');
        outStrct.data = voxSignal;
        outStrct.hdr.dim(5) = size(voxSignal,4); % assign the number of t-dimensions
        NIfTI.Write(outStrct,outputFile);
    else
        surfSignal(:,1) = squeeze(dataStrct.zScore);
        surfSignal(:,2) = squeeze(dataStrct.SNR);
        surfSignal(:,3) = squeeze(dataStrct.amplitude);
        surfSignal(:,4) = squeeze(dataStrct.phase);
        surfSignal(:,5) = squeeze(dataStrct.realSignal);
        surfSignal(:,6) = squeeze(dataStrct.imagSignal);
        
        newLabels(1) = {'Z-SCORE'};
        newLabels(2) = {'SNR'};
        newLabels(3) = {'AMP'};
        newLabels(4) = {'PHASE'};
        newLabels(5) = {'REAL PART'};
        newLabels(6) = {'IMAG PART'};

        newStats = repmat({'none'},1,length(newLabels));

        %% PRINT NIFTI
        outStrct = inStrct{1}; % not sure if this will cause problems
        outStrct.data = surfSignal;
        outStrct.labels = newLabels;
        outStrct.stats = newStats;
        afni_niml_writesimple(outputFile,outStrct); 
        clear surfSignal;
    end
end


