function outStrct = mriFFT(inData,nCycles,nHarm,roiName)
    % Description:	DO FFT on fMRI data
    % 
    % Syntax:	outPutStruct = fMRI_FFT(fmriData)
    % In:
    % 	inData   - 3-D matrix of data. First dimension is time, second
    %                                  dimension is voxels, and third dimension is runs
    %   nCycles  - number of cycles per run (default: 10)
    %   nHarm    - number of harmonics to compute SNR/z-score for (default: 5)
    %   noRunAve - don't average over runs
    %
    %   Out:
    % 	outStrct	- a struct with FFT data
    
    if nargin < 2
        nCycles = 10;
    else
    end
    if nargin < 3
        nHarm = 5;
    else
    end
    if nargin < 4
        roiName = false;
    else
    end
    
    if ndims(inData) > 3 % whole brain data
        inData = nanmean(inData,5); % average over runs
        % fourth dimension is the time dimension
        % temporalily shift this to the first dimension
        mean_tSeries = permute(inData,[4,1,2,3]); 
    else % roi data
        % first dimension is the time dimension
        if ndims(inData) == 3
            inData = nanmean(inData,3); % average over runs
        else
        end
        if ndims(inData) > 1
            % average over voxels, then fft
            mean_tSeries = mean(inData,2);
        else
        end
    end
    nT = size(inData,1); 
    maxCycles = round(nT/2);
    
    % scale amplitude by length
    % and multiply by 2 to get single-sided spectrum
    % (spectrum is symmetric around DC - positive and negative)
    absFFT = 2*abs(fft(mean_tSeries,[],1)) ./ nT;    
    y = absFFT(2:maxCycles+1,:,:,:); % omit DC 
    
    % compute phase
    p = angle(fft(mean_tSeries,[],1));
    p = p(2:maxCycles+1,:,:,:);
    y_angle = p(nCycles,:,:,:);
    
    % get real and imaginary component
    y_complex = 2*fft(mean_tSeries,[],1) ./ nT;
    y_complex = y_complex(2:maxCycles+1,:,:,:);

    %Calculate Z-score
    % Compute the mean and std of the non-signal amplitudes.  Then compute the
    % z-score of the signal amplitude w.r.t these other terms.  This appears as
    % the Z-score in the plot.  This measures how many standard deviations the
    % observed amplitude differs from the distribution of other amplitudes.
    
    if ndims(inData) > 3 % whole brain data, only do first harmonic
        nHarm = 1;
    else
        if nCycles*nHarm > (size(y,1)-2)
            error(['too many harmonics included (',num2str(nHarm),') no room in spectrum']);
        else
        end
    end
    for c = 1:nHarm % the first nHarm harmonics of stimulus frequency
        lst1 = true([1, maxCycles]);
        lst1(nCycles*c) = 0;
        zScore(c,:,:,:) = (y(nCycles*c,:,:,:) - mean(y(lst1,:,:,:),1)) / std(y(lst1,:,:,:),0,1); % all harmonics

        % norcia-style Signal-to-Noise Ratio. 
        % Signal divided by mean of 4 side bands
        lst2 = true([1, maxCycles]);
        lst2([nCycles*c-1,nCycles*c-2,nCycles*c+1,nCycles*c+2])=true;
        norciaSNR(c,:,:,:) = y(nCycles*c,:,:,:)/mean(y(lst2,:,:,:));
        y_raw_amp(c,:,:,:) = y(nCycles*c,:,:,:);

        y_real_signal(c,:,:,:) = real(y_complex(nCycles*c,:,:,:));
        y_real_noise(c,:,:,:,:) = real(y_complex(lst2));
        y_imag_signal(c,:,:,:) = imag(y_complex(nCycles*c,:,:,:));
        y_imag_noise(c,:,:,:,:) = imag(y_complex(lst2,:,:,:));
    end
    
    if ndims(inData) <= 3 % only compute mean cycle if ROI data
        % compute meanCycle
        meanCycle = mean(reshape(mean(inData,2),size(inData,1)/nCycles,nCycles),2);
        meanCycle = meanCycle-mean(meanCycle(1:3));
    else
        meanCycle = [];
        y = permute(y,[2,3,4,1]);
        zScore = permute(zScore,[2,3,4,1]);
        norciaSNR = permute(norciaSNR,[2,3,4,1]);
        y_angle = permute(y_angle,[2,3,4,1]);
        y_real_signal = permute(y_real_signal,[2,3,4,1]);
        y_imag_signal = permute(y_imag_signal,[2,3,4,1]);
        y_raw_amp = permute(y_raw_amp,[2,3,4,1]);
    end
        
    %% add to data struct
    outStrct.harmonics = y;
    outStrct.meanCycle = meanCycle;
    outStrct.zScore = zScore;
    outStrct.SNR = norciaSNR;
    outStrct.phase = y_angle;
    outStrct.amplitude = y_raw_amp;
    outStrct.rawData = inData;
    outStrct.realSignal = y_real_signal;
    outStrct.imagSignal = y_imag_signal;
    outStrct.realNoise = y_real_noise;
    outStrct.imagNoise = y_imag_noise;
    if roiName
        outStrct.name = roiName;
    else
    end
    outStrct = orderfields(outStrct);
end

