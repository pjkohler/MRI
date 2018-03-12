import os, subprocess, sys, glob, shutil
from os.path import expanduser

## HELPER FUNCTIONS
def fs_dir_check(fs_dir,subject):
    # check if subjects' SUMA directory exists
    if os.path.isdir("{0}/{1}/SUMA".format(fs_dir,subject)):
        # no suffix needed
        suffix=""
    else:
        # suffix needed
        suffix="_fs4"
        if not os.path.isdir("{0}/{1}{2}".format(fs_dir,subject,suffix)):
            sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
            .format(fs_dir,subject,suffix))
    return suffix

def copy_suma_files(suma_dir,tmp_dir):
    for file in glob.glob(suma_dir+"/*h.smoothwm.asc"):
        shutil.copy(file,tmp_dir)
    for file in glob.glob(suma_dir+"/*h.pial.asc"):
        shutil.copy(file,tmp_dir)
    for file in glob.glob("{0}/{1}{2}{3}*.spec".format(suma_dir,spec_prefix,subject,suffix)):
        shutil.copy(file,tmp_dir)
    # for some reason, 3dVol2Surf requires these files, so copy them as well
    for file in glob.glob(suma_dir+"/*aparc.*.annot.niml.dset"):
        shutil.copy(file,tmp_dir)

def get_name_suffix(cur_file,surface=False):
    if surface:
        if ".niml.dset" in cur_file:
            file_name = cur_file[:-10]
        elif ".niml.roi" in cur_file:
            file_name = cur_file[:-9]
            # convert to niml.dset
            subprocess.call("ROI2dataset -prefix {0}.niml.dset -input {0}.niml.roi"
                .format(file_name), shell=True)
        else:
            sys.exit("ERROR! Unknown input dataset: '{0}'."
                .format(cur_file))
    else:
        if ".nii.gz" in cur_file:
            file_name = cur_file[:-7]
            suffix = ".nii.gz"
        elif ".nii" in cur_file:
            file_name = cur_file[:-4]
            suffix = ".nii"
        elif "+orig" in cur_file:
            file_name = cur_file.rpartition("+")            
            file_name = file_name[0]
            suffix = "".join(file_name[1:])
    
    return file_name, suffix

## MAIN FUNCTIONS

def Suma(subject, hemi='both', open_vol=False, surf_vol='standard', std141=False, fs_dir=None): 
    """
    Wrapper function for easy opening of SUMA viewer.
    Supports opening suma surfaces both in native and std141 space.
    Supports opening a volume file in afni, linked to the surfaces,
    via the --openvol and --surfvol options. If surfvol is given,  
    openvol will be assumed. Note that when a volume file is given,
    script will change to its directory.                           
    
    Author: pjkohler, Stanford University, 2016
    """

    if fs_dir is None:
        fs_dir = os.environ["SUBJECTS_DIR"]

    suffix = fs_dir_check(fs_dir,subject)
    
    suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir,subject,suffix)
    if std141:
        spec_file="{0}/std.141.{1}{2}_{3}.spec".format(suma_dir,subject,suffix,hemi)
    else:
        spec_file="{0}/{1}{2}_{3}.spec".format(suma_dir,subject,suffix,hemi)

    if surf_vol is "standard":
        surf_vol = "{0}/{1}{2}/SUMA/{1}{2}_SurfVol+orig".format(fs_dir,subject,suffix)
    else:
        # if surfvol was assigned, assume user wants to open volume
        open_vol = True

    if open_vol:        
        vol_dir = '/'.join(surf_vol.split('/')[0:-1])
        vol_file = surf_vol.split('/')[-1]        
        if vol_dir: # if volume directory is not empty
             os.chdir(vol_dir)
        subprocess.call("afni -niml & SUMA -spec {0} -sv {1} &".format(spec_file,vol_file), shell=True)
    else:
        subprocess.call("SUMA -spec {0} &".format(spec_file), shell=True)

def Pre(in_files, ref_file='last', tr_dur=0, pre_tr=0, total_tr=0, slice_time_file=None, pad_ap=0, pad_is=0, diff_mat=False, keep_temp=False):
    """
    Function for first stage of preprocessing
    Slice-time correction and deobliqueing, in that order.
    Also supports data with different number of slices,
    and padding of the matrices, via flags 
    --diffmat, --pad_ap and --pad_is. 

    Author: pjkohler, Stanford University, 2016"""

    # assign remaining defaults
    if ref_file in "last":
        ref_file = in_files[-1] # use last as reference
    if tr_dur is 0:
        # TR not given, so compute
        tr_dur = subprocess.check_output("3dinfo -tr -short {0}".format(ref), shell=True)
        tr_dur = tr_dur.rstrip("\n")
    if total_tr is 0:
        # include all TRs, so get max subbrick value
        total_tr = subprocess.check_output("3dinfo -nvi -short {0}".format(ref), shell=True)
        total_tr = total_tr.rstrip("\n")
    else:
        # subject has given total number of TRs to include, add preTR to that
        total_tr = eval("pre_tr + total_tr")

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    os.chdir(tmp_dir)
    
    name_list = []

    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)
        
        # crop and move files
        cur_total_tr = subprocess.check_output("3dinfo -nvi -short {1}/{0}{2}"
            .format(file_name,cur_dir,suffix), shell=True)
        cur_total_tr =  cur_total_tr.rstrip("\n")

        subprocess.call("3dTcat -prefix {0}+orig {1}/{0}{2}''[{3}..{4}]''"
            .format(file_name,cur_dir,suffix,pre_tr,min(cur_total_tr, total_tr)), shell=True)
        
        # slice timing correction
        if slice_time_file is None:
            subprocess.call("3dTshift -quintic -prefix {0}.ts+orig -TR {1}s -tzero 0 -tpattern alt+z {0}+orig".format(file_name,tr_dur), shell=True)
        else:
            subprocess.call("3dTshift -quintic -prefix {0}/{1}.ts+orig -TR {2}s -tzero 0 -tpattern @{3} {0}/{1}+orig"
                .format(tmp_dir,file_name,tr_dur,slice_time_file), shell=True)
            
        # deoblique
        subprocess.call("3dWarp -deoblique -prefix {0}.ts.do+orig {0}.ts+orig".format(file_name), shell=True)
        
        # pad 
        if pad_ap is not 0 or pad_is is not 0:        
            subprocess.call("3dZeropad -A {1} -P {1} -I {2} -S {2} -prefix {0}.ts.do.pad+orig {0}.ts.do+orig".format(file_name,pad_ap,pad_is), shell=True)
            subprocess.call("rm {0}.ts.do+orig*".format(file_name), shell=True)
            subprocess.call("3dRename {0}.ts.do.pad+orig {0}.ts.do+orig".format(file_name), shell=True)

        if diff_mat:
            # add file_names to list, move later
            namelist.append(file_name)
            if cur_file == ref_file:
                ref_name = file_name
        else:
            subprocess.call("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.nii.gz {0}.ts.do+orig".format(file_name,cur_dir), shell=True)
    
    if diff_mat:
        # take care of different matrices, and move
        for file_name in namelist:
                subprocess.call("@Align_Centers -base {1}.ts.do+orig -dset {0}.ts.do+orig".format(file_name,ref_name), shell=True)
                subprocess.call("3dresample -master {1}.ts.do+orig -prefix {0}.ts.do.rs+orig -inset {0}.ts.do_shft+orig".format(file_name,ref_name), shell=True)
                subprocess.call("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.rs.nii.gz {0}.ts.do.rs+orig".format(file_name,cur_dir), shell=True)   

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)

def Volreg(in_files, ref_file='last', slow=False, keep_temp=False):
    """
    Function for second stage of preprocessing: Volume registation.
    Typically run following mriPre.py
    Use option --slow for difficult cases
    Author: pjkohler, Stanford University, 2016
    """ 
    # assign remaining defaults
    if ref_file in "last":
        ref_file = in_files[-1] # use last as reference

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    os.chdir(tmp_dir)
    
    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)
        
        # move files
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(file_name,cur_dir,suffix), shell=True)
        
        # do volume registration
        if slow:
            subprocess.call("3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -heptic -twopass -maxite 50 {0}+orig"
                .format(file_name,ref,cur_dir), shell=True)
        else:
            subprocess.call("3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -Fourier {0}+orig"
                .format(file_name,ref,cur_dir), shell=True)
    
    os.chdir(cur_dir)    
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)

def Scale(in_files, no_dt=False, keep_temp=False):
    """
    Function for third stage of preprocessing: Scaling and Detrending.
    Typically run following mriPre.py and mriVolreg.py 
    Detrending currently requires motion registration parameters
    as .1D files: motparam.xxx.1D 

    Author: pjkohler, Stanford University, 2016
    """
    
    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    os.chdir(tmp_dir)
    
    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)
        
        # move files
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(file_name,cur_dir,suffix), shell=True)
        # compute mean
        subprocess.call("3dTstat -prefix mean_{0}+orig {0}+orig".format(file_name), shell=True)
        # scale
        if no_dt:
            # save scaled data in data folder directly
            subprocess.call("3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {1}/{0}.sc.nii.gz"
                .format(file_name,cur_dir), shell=True)
        else:
            # scale, then detrend and store in data folder
            subprocess.call("3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {0}.sc+orig"
                .format(file_name), shell=True)
            subprocess.call("3dDetrend -prefix {1}/{0}.sc.dt.nii.gz -polort 2 -vector {1}/motparam.{0}.1D {0}.sc+orig"
                .format(file_name,cur_dir), shell=True)
                
    os.chdir(curdir)    
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir)

def Vol2Surf(subject, in_files, map_func='ave', wm_mod=0.0, gm_mod=0.0, prefix=None, index='voxels', steps=10, mask=None, fs_dir=None, surf_vol='standard', std141=False, keep_temp=False):
    """
    Function for converting from volume to surface space.  
    Supports suma surfaces both in native and std141 space.
    Surfave volume can be given using the --surf_vol argument.
    Various other options from 3dVol2Surf are implemented, 
    sometimes with names that are more meaningful (to me).
    'data' option for mask still needs to be implemented. 

    Author: pjkohler, Stanford University, 2016
    """
    # get current directory    
    cur_dir = os.getcwd()
    # make temporary, local folder
    tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))   
    
    if fs_dir is None:
        fs_dir = os.environ["SUBJECTS_DIR"]
    # check if subjects' SUMA directory exists
    suffix = fs_dir_check(fs_dir,subject)
    suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir,subject,suffix)
    
    if wm_mod is not 0.0 or gm_mod is not 0.0:
        # for gm, positive values makes the distance longer, for wm negative values
        steps = round(steps + steps * gm_mod - steps * wm_mod)
    
    print "MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod,gm_mod,steps)

    if surf_vol is "standard":
        vol_dir = "{0}/{1}{2}/SUMA".format(fs_dir,subject,suffix) 
        vol_file = "{0}{1}_SurfVol+orig".format(subject,suffix)
    else:
        vol_dir = '/'.join(surf_vol.split('/')[0:-1])
        vol_file = surf_vol.split('/')[-1]
        if not vol_dir: # if volume directory is empty
            vol_dir = cur_dir
    
    # make temporary copy of volume file     
    subprocess.call("3dcopy {0}/{1} {2}/{1}".format(vol_dir,vol_file,tmp_dir,vol_file), shell=True)
    
    # now get specfiles
    if prefix is None:
        prefix = "."
    else:
        prefix = ".{0}.".format(prefix)    
    
    if std141:
        specprefix = "std.141."
        prefix = ".std.141{0}".format(prefix)
    else:
        specprefix = ""    
    
    copy_suma_files(suma_dir,tmp_dir)

    os.chdir(tmp_dir)
    for cur_file in in_files:
        file_name, file_suffix = get_name_suffix(cur_file)   
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(file_name,cur_dir,file_suffix), shell=True)
        if mask is None:
            # no mask
            maskcode = ""
            if mask is 'data':
                # mask from input data
                maskcode = "-cmask '-a {0}[0] -expr notzero(a)' ".format(file_name)
            else:
                # mask from distinct dataset, copy mask to folder
                mask_name, mask_suffix = get_name_suffix(mask)
                subprocess.call("3dcopy {1}/{0}{2} mask+orig".format(mask_name,cur_dir,mask_suffix), shell=True)
                maskcode = "-cmask '-a mask+orig[0] -expr notzero(a)' "
        for hemi in ["lh","rh"]:
            subprocess.call("3dVol2Surf -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {5}+orig -map_func {6} \
                    -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                    -outcols_NSD_format -oob_value -0 {13}-out_niml {11}/{3}{12}{5}.niml.dset"
                    .format(specprefix,subject,suffix,hemi,vol_file,file_name,map_func,index,wm_mod,gm_mod,steps,cur_dir,prefix,maskcode), shell=True)
    
    os.chdir(cur_dir)    
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)

def Surf2Vol(subject, in_files, map_func='ave', wm_mod=0.0, gm_mod=0.0, prefix=None, index='voxels', steps=10, out_dir=None, fs_dir=None, surf_vol='standard', std141=False, keep_temp=False):
    """
    Function for converting from surface to volume space.  
    Supports suma surfaces both in native and std141 space.
    Surfave volume can be given using the --surf_vol argument.
    Various other options from 3dSurf2Vol are implemented, 
    sometimes with names that are more meaningful (to me).

    Author: pjkohler, Stanford University, 2016
    """
    # get current directory    
    cur_dir = os.getcwd()

    # make temporary, local folder
    tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))   
    
    # check if subjects' freesurfer directory exists
    if fs_dir is None:
        fs_dir = os.environ["SUBJECTS_DIR"]
    if out_dir is None:
        out_dir = cur_dir
    # check if subjects' SUMA directory exists
    suffix = fs_dir_check(fs_dir,subject)
    suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir,subject,suffix)
    
    if wm_mod is not 0.0 or gm_mod is not 0.0:
        # for gm, positive values makes the distance longer, for wm negative values
        steps = round(steps + steps * gm_mod - steps * wm_mod)
    
    print "MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod,gm_mod,steps)

    if surf_vol is "standard":
        vol_dir = "{0}/{1}{2}/SUMA".format(fs_dir,subject,suffix) 
        vol_file = "{0}{1}_SurfVol+orig".format(subject,suffix)
    else:
        vol_dir = '/'.join(surf_vol.split('/')[0:-1])
        vol_file = surf_vol.split('/')[-1]
        if not vol_dir: # if volume directory is empty
            vol_dir = cur_dir
    
    # make temporary copy of volume file     
    subprocess.call("3dcopy {0}/{1} {2}/{1}".format(vol_dir,vol_file,tmp_dir,vol_file), shell=True)
    
    # now get specfiles
    if prefix is None:
        prefix = "."
    else:
        prefix = ".{0}.".format(prefix)    
    
    if std141:
        specprefix = "std.141."
        prefix = ".std.141{0}".format(prefix)
    else:
        specprefix = ""    
    
    copy_suma_files(suma_dir,tmp_dir)
    
    os.chdir(tmp_dir)
    for curfile in in_files:
        shutil.copy("{0}/{1}".format(cur_dir,cur_file),tmp_dir)
        file_name, file_suffix = get_name_suffix(cur_file,surface=True)   
                
        if 'lh' in file_name.split('.'):
            hemi = 'lh'
        elif 'rh' in file_name.split('.'):
            hemi = 'rh'
        else:
            sys.exit("ERROR! Hemisphere could not be deduced from: '{0}'."
                .format(cur_file))

        subprocess.call("3dSurf2Vol -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {4} \
                    -sdata {5}.niml.dset -map_func {6} -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                    -prefix {11}/{5}"
                    .format(specprefix,subject,suffix,hemi,vol_file,file_name,mapfunc,index,wm_mod,gm_mod,steps,tmp_dir), shell=True)

        subprocess.call("3dcopy {2}/{0}+orig {1}/{0}.nii.gz".format(file_name,out_dir,tmp_dir), shell=True)
    
    os.chdir(cur_dir)    
    if keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir) 
        
