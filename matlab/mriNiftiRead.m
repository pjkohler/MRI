function data = pkNiftiRead(filePath)
    for z=1:length(filePath)
        tmp = NIfTI.Read(filePath{z});
        if ndims(tmp.data) == 3
            data(:,:,:,z) = tmp.data;
        else
            data(:,:,:,:,z) = tmp.data;
        end
    end
end
