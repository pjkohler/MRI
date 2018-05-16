import os, subprocess, sys, glob, shutil, tempfile
from os.path import expanduser
from nilearn import datasets, surface
import numpy as np
import scipy as scp

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

def rsync(input, output):
    cmd = "rsync -avz --progress --remove-source-files %s/* %s/." % (input, output)
    p = subprocess.Popen(cmd, shell=True)
    stdout, stderr = p.communicate()
    return stderr

def shell_cmd(main_cmd, fsdir=None, do_print=False):
    if fsdir is not None:
        main_cmd = "export SUBJECTS_DIR={0}; {1}".format(fsdir, main_cmd)
    if do_print:
        print(main_cmd+'\n')
    subprocess.call("{0}".format(main_cmd), shell=True)

def fft_offset(complex_in, offset_rad):
    amp = np.absolute(complex_in)
    phase = np.angle(complex_in)
    # subtract offset from phase
    phase = phase - offset_rad
    phase = ( phase + np.pi) % (2 * np.pi ) - np.pi
    # convert back to complex
    complex_out = np.absolute(amp)*np.exp(1j*phase)
    amp_out = np.mean(np.absolute(complex_out))
    phase_out = np.mean(np.angle(complex_out))
    return complex_out, amp_out, phase_out

## CLASSES

class roiobject:
    def __init__(self, curdata=np.zeros((120, 1)), curobject=None, roiname="unknown", tr=99, stimfreq=99, nharm=99, num_vox=0):
        if curobject is None:
            self.data = []
            self.roi_name = roiname
            self.tr = tr
            self.stim_freq = stimfreq
            self.num_harmonics = nharm
            self.num_vox = num_vox
        else:
            # if curobject provided, inherit all values
            self.data = curobject.data
            self.roi_name = curobject.roi_name
            self.tr = curobject.tr
            self.stim_freq = curobject.stim_freq
            self.num_harmonics = curobject.num_harmonics
            self.num_vox = curobject.num_vox
        if curdata.any():
            self.data.append( curdata.reshape(curdata.shape[0],1) )
            self.mean = self.average()
            self.fft = self.fourieranalysis()
    def average(self):
        if len(self.data) is 0:
            return []
        else:
            return np.mean(self.data,axis=0)  
    def fourieranalysis(self):
        return MriFFT(self.average(),
            tr=self.tr,
            stimfreq=self.stim_freq,
            nharm=self.num_harmonics)
     
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

def Neuro2Radio(in_files):
    for scan in in_files:
        name, suffix = get_name_suffix(scan)
        old_orient = subprocess.check_output("fslorient -getorient {0}".format(scan), shell=True, universal_newlines=True)
        print("Old orientation: {0}".format(old_orient))
        temp_scan = name+"_temp"+suffix
        shutil.copyfile(scan,temp_scan)
        try:
            shell_cmd("fslswapdim {0} z x y {0}".format(scan))
            shell_cmd("fslswapdim {0} -x -y -z {0}".format(scan))
            shell_cmd("fslorient -swaporient {0}".format(scan))
        except:
            # replace with original
            shutil.copyfile(temp_scan, scan)
            print("Orientation could not be changed for file {0}".format(scan))
        os.remove(temp_scan)
        new_orient = subprocess.check_output("fslorient -getorient {0}".format(scan), shell=True, universal_newlines=True)
        print("New orientation: {0}".format(new_orient))

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
    
    print("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod,gm_mod,steps))

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
    
    print("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod,gm_mod,steps))

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

def RoiTemplates(subjects, run_ben_glass_wng_kgs="all", atlasdir=None, fsdir=None, outdir="standard", forcex=False, separate_out=False, keeptemp=False, skipclust=False, intertype="NearestNode"):
    """Function for generating V1-V3 ROIs in subject's native space 
    predicted from the cortical surface anatomy
    as described in Benson et al. (PLoS Comput Biol., 2014).
    Requires template data, which can be downloaded at:
    https://cfn.upenn.edu/aguirre/wiki/public:retinotopy_template

    Author: pjkohler & fthomas, Stanford University, 2016
    
    This function also generates ROIs based on Wang, Glasser and KGS methodology.

    Parameters
    ------------
    subjects : list of strings
            A list of subjects that are to be run.
    run_ben_glass_wng_kgs : string or list of strings, default "all"
            This defaults to "All" - resulting in Benson, Glasser, Wang or KGS being
             run. Options: ['Benson','Glasser','Wang','KGS','All']
    atlasdir : string, default None
            The atlas directory, containing ROI template data
    fsdir : string, default None
            The freesurfer directory of subjects
    outdir : string, default "standard"
            Output directory
    forcex : boolean, default False
            If there is no xhemi directiory, then as part of Benson;
            register lh to fsaverage sym & mirror-reverse subject rh 
            and register to lh fsaverage_sym
    separate_out : boolean, default False
            Can choose to separate out as part of Benson into ["angle", "eccen", "areas", "all"]
    keeptemp : boolean, default False
            Option to keep the temporary files that are generated
    skipclust : boolean, default False
            If True then will do ptional surface-based clustering
    intertype : string, default "NearestNode"
            Argument for SurfToSurf (AFNI). Options: 
            - NearestTriangleNodes
            - NearestNode
            - NearestTriangle
            - DistanceToSurf
            - ProjectionOnSurf
            - NearestNodeCoords
            - Data
    """
    
    
    if fsdir is None:
        old_subject_dir=os.environ["SUBJECTS_DIR"]
        fsdir = os.environ["SUBJECTS_DIR"]
    else:
        old_subject_dir=os.environ["SUBJECTS_DIR"]
        os.environ["SUBJECTS_DIR"] = fsdir
    if atlasdir is None:
        atlasdir = "{0}/ROI_TEMPLATES".format(fsdir)

    # get current directory    
    curdir = os.getcwd()

    # Assessing user input - need to identify which elements they want to run
    run_benson, run_glasser, run_wang, run_kgs = False, False, False, False
    run_possible_arguments=['benson','glasser','wang','kgs']
    run_ben_glass_wng_kgs = str(run_ben_glass_wng_kgs).lower()
    confirmation_str = 'Running: '
    if 'all' in run_ben_glass_wng_kgs:
        run_benson, run_glasser, run_wang, run_kgs = True, True, True, True
        confirmation_str += 'Benson, Glasser, Wang, KGS'
    elif [name for name in run_possible_arguments if name in run_ben_glass_wng_kgs]:
        if 'benson' in run_ben_glass_wng_kgs:
            run_benson = True
            confirmation_str += 'Benson, '
        if 'glasser' in run_ben_glass_wng_kgs:
            run_glasser = True
            confirmation_str += 'Glasser, '
        if 'wang' in run_ben_glass_wng_kgs:
            run_wang = True
            confirmation_str += 'Wang, '
        if 'kgs' in run_ben_glass_wng_kgs:
            run_kgs = True
            confirmation_str += 'KGS'
    else:
        print('Error - no correct option selected. Please input: benson, glasser, wang or KGS')
        return None
    print(confirmation_str)

    for sub in subjects: # loop over list of subjects
        # check if subjects' freesurfer directory exists
        if os.path.isdir("{0}/{1}".format(fsdir,sub)):
            # no suffix needed
            suffix=""
        elif os.path.isdir("{0}/sub-{1}".format(fsdir,sub)):
            sub = "sub-{0}".format(sub)
            suffix=""
        else:
            # suffix needed
            suffix="_fs4"
            if not os.path.isdir("{0}/{1}{2}".format(fsdir,sub,suffix)):
                sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                    .format(fsdir,sub,suffix))        
        
        if outdir in "standard":
            outdir = "{0}/{1}{2}/TEMPLATE_ROIS".format(fsdir,sub,suffix)
        else:
            outdir = "{0}/{1}/TEMPLATE_ROIS".format(outdir,sub,suffix) # force sub in name, in case multiple subjects

        # make temporary, local folder
        tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))

        # and subfoldes
        os.mkdir(tmpdir+"/surf")
        os.mkdir(tmpdir+"/TEMPLATE_ROIS")
        os.mkdir(tmpdir+"/SUMA")
        
        # copy relevant freesurfer files & establish surface directory
        surfdir = "{0}/{1}{2}".format(fsdir,sub,suffix)

        for file in glob.glob(surfdir+"/surf/*h.white"):
            shutil.copy(file,tmpdir+"/surf")
        sumadir = "{0}/{1}{2}/SUMA".format(fsdir,sub,suffix)
        for file in glob.glob(sumadir+"/*h.smoothwm.asc"):
            shutil.copy(file,tmpdir+"/SUMA")
        for file in glob.glob("{0}/{1}_*.spec".format(sumadir,sub)):
            shutil.copy(file,tmpdir+"/SUMA")
        for file in glob.glob("{0}/{1}{2}.std141_to_native.*.niml.M2M".format(sumadir,sub,suffix)):
            shutil.copy(file,tmpdir+"/SUMA")
        os.chdir(tmpdir)
        
        # BENSON ROIS *******************************************************************
        if run_benson == True:

            outname = 'Benson2014'

            if os.path.isdir(surfdir+"/xhemi") is False or forcex is True:
                #Invert the right hemisphere - currently removed as believed not needed
                #shell_cmd("xhemireg --s {0}{1}".format(sub,suffix), fsdir,do_print=True)
                # register lh to fsaverage sym
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh".format(sub,suffix), fsdir,do_print=True)
                # mirror-reverse subject rh and register to lh fsaverage_sym
                # though the right hemisphere is not explicitly listed below, it is implied by --lh --xhemi
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh --xhemi".format(sub,suffix), fsdir,do_print=True)
            else:
                print("Skipping fsaverage_sym registration")

            if separate_out:
                datalist = ["angle", "eccen", "areas", "all"]
            else:
                datalist = ["all"]

            for bdata in datalist:

                # resample right and left hemisphere data to symmetric hemisphere
                shell_cmd("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1} --trgsurfreg {2}.sphere.reg \
                    --hemi lh --sval {3}/{5}/{4}-template-2.5.sym.mgh --tval ./TEMPLATE_ROIS/lh.{5}.{4}.mgh"
                    .format(sub,suffix,"fsaverage_sym",atlasdir,bdata,outname,tmpdir), fsdir)
                shell_cmd("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1}/xhemi --trgsurfreg {2}.sphere.reg \
                    --hemi lh --sval {3}/{5}/{4}-template-2.5.sym.mgh --tval ./TEMPLATE_ROIS/rh.{5}.{4}.mgh"                
                    .format(sub,suffix,"fsaverage_sym",atlasdir,bdata,outname,tmpdir), fsdir)
                # convert to suma
                for hemi in ["lh","rh"]:
                    shell_cmd("mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.{2}.mgh ./surf/{0}.white ./TEMPLATE_ROIS/{0}.{1}.{2}.gii".format(hemi,outname,bdata,tmpdir))
                    shell_cmd("ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.{2}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.{2}.niml.dset".format(hemi,outname,bdata,tmpdir))

        # GLASSER ROIS *******************************************************************
        if run_glasser == True:

            outname = 'Glasser2016'

            for hemi in ["lh","rh"]:
                # convert from .annot to mgz
                shell_cmd("mri_annotation2label --subject fsaverage --hemi {0} --annotation {1}/{2}/{0}.HCPMMP1.annot --seg {0}.glassertemp1.mgz"
                    .format(hemi,atlasdir,outname))
                # convert to subjects native space
                shell_cmd("mri_surf2surf --srcsubject fsaverage --trgsubject {2}{3} --sval {0}.glassertemp1.mgz --hemi {0} --tval ./TEMPLATE_ROIS/{0}.{1}.mgz"
                    .format(hemi,outname,sub,suffix), fsdir)
                # convert mgz to gii
                shell_cmd("mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.mgz ./surf/{0}.white ./TEMPLATE_ROIS/{0}.{1}.gii"
                    .format(hemi,outname))
                # convert gii to niml.dset
                shell_cmd("ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.niml.dset"
                    .format(hemi,outname))

        ## WANG ROIS *******************************************************************
        if run_wang == True:

            outname = 'Wang2015'

            for file in glob.glob("{0}/Wang2015/subj_surf_all/maxprob_surf_*.1D.dset".format(atlasdir)): 
                shutil.copy(file,tmpdir+"/.")

            for hemi in ["lh","rh"]:
                # if you have a mapping file, this is much faster.  see SurfToSurf -help
                # you can still run without a mapping file, but it is generated on-the-fly (slow!)
                mapfile = "./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi)
                if os.path.isfile(mapfile):
                    print("Using existing mapping file {0}".format(mapfile))
                    subprocess.call("SurfToSurf -i_fs ./SUMA/{0}.smoothwm.asc -i_fs ./SUMA/std.141.{0}.smoothwm.asc -output_params {1} -mapfile {2} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi,intertype,mapfile), shell=True)
                    newmap = False
                else:
                    print "Generating new mapping file"
                    newmap = True
                    subprocess.call("SurfToSurf -i_fs ./SUMA/{0}.smoothwm.asc -i_fs ./SUMA/std.141.{0}.smoothwm.asc -output_params {1} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi,intertype), shell=True)       
                    # update M2M file name to be more informative and not conflict across hemispheres
                    os.rename("./SurfToSurf.niml.M2M".format(outname, hemi), "./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi))
                
                # give output file a more informative name
                os.rename("./SurfToSurf.maxprob_surf_{0}.niml.dset".format(hemi),"./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname,hemi))
                #convert output to gii
                shell_cmd("ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.gii".format(outname,hemi))
                # we don't need this and it conflicts across hemisphere                    
                os.remove("./SurfToSurf.1D".format(outname, hemi))
                #for file in glob.glob("./maxprob_surf_*.1D.dset"):
                #    os.remove(file)
                
                # make a 1D.dset copy using the naming conventions of other rois,
                # so we can utilize some other script more easily (e.g., roi1_copy_surfrois_locally.sh)
                # mainly for Kastner lab usage
                subprocess.call("ConvertDset -o_1D -input ./{1}/{0}.{1}.niml.dset -prepend_node_index_1D -prefix ./{1}/{0}.{1}.1D.dset"
                    .format(hemi, outname), shell=True)
                
                if skipclust: # do optional surface-based clustering
                    print '######################## CLUSTERING ########################'
                    for idx in range(1,26):
                        # clustering steps
                        specfile="./SUMA/{0}{1}_{2}.spec".format(sub,suffix,hemi)  
                        surffile="./SUMA/{0}.smoothwm.asc".format(hemi)
            
                        # isolate ROI
                        subprocess.call("3dcalc -a ./{0}/{2}.{0}.niml.dset -expr 'iszero(a-{1})' -prefix {2}.temp.niml.dset"
                            .format(outname, idx,hemi), shell=True)
                        
                        # do clustering, only consider cluster if they are 1 edge apart
                        subprocess.call("SurfClust -spec {0} -surf_A {1} -input {2}.temp.niml.dset 0 -rmm -1 -prefix {2}.temp2 -out_fulllist -out_roidset"
                            .format(specfile,surffile,hemi), shell=True)
                            
                        # pick only biggest cluster
                        if idx is 1:
                            if os.path.isfile("./{0}/{1}.{0}_cluster.niml.dset".format(outname,hemi)):
                                print("Removing existing file ./{0}/{1}.{0}_cluster.niml.dset".format(outname,hemi)) 
                                os.remove("./{0}/{1}.{0}_cluster.niml.dset".format(outname,hemi))
                            subprocess.call("3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)*{2}' -prefix {1}.{0}_cluster.niml.dset"
                                .format(outname,hemi,idx), shell=True)
                        else:
                            subprocess.call("3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -b {1}.{0}_cluster.niml.dset -expr 'b+iszero(a-1)*{2}' -prefix {1}.temp3.niml.dset"
                                .format(outname,hemi,idx), shell=True)
                            #os.remove("./{0}/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                            os.rename("{0}.temp3.niml.dset".format(hemi), "{1}.{0}_cluster.niml.dset".format(outname, hemi))
                                
                        for file in glob.glob("./*temp*"):
                            os.remove(file)
                    # is this step necessary?
                    subprocess.call("ConvertDset -input {1}.{0}_cluster.niml.dset -o_niml_asc -prefix {1}.temp4.niml.dset"
                        .format(outname,hemi,idx), shell=True)
                    os.remove("{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    os.rename("{0}.temp4.niml.dset".format(hemi), "./{0}/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    #convert output to gii
                    shell_cmd("ConvertDset -o_gii_asc -input ./{0}/{1}.{0}_cluster.niml.dset -prefix ./{0}/{1}.{0}_cluster.gii".format(outname,hemi))
                # copy mapping file to subjects' home SUMA directory
                if newmap:            
                    shutil.move("./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi),
                                "{3}/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi,sumadir))
            
            
        ##KGS ROIs *********************************************************************
        if run_kgs == True:

            outname='KGS2016'

            os.chdir(tmpdir)
            for hemi in ["lh","rh"]:

                idx = 0
                for roi in ["IOG","OTS","mFUS","pFUS","PPA","VWFA1","VWFA2"]:

                    idx += 1

                    if not os.path.isfile("{1}/{3}/{0}.MPM_{2}.label".format(hemi,atlasdir,roi,outname)):
                        # if label file does not exist, skip it
                        print("file doesn't exist")
                        print("{1}/{3}/{0}.MPM_{2}.label".format(hemi,atlasdir,roi,outname))
                        continue
                    # Make the intermediate (subject-native) surface:
                    #   --srcsubject is always fsaverage since we assume the input file is an fsaverage file
                    #   --trgsubject is the subject we want to convert to
                    #   --sval is the file containing the surface data
                    #   --hemi is just the hemisphere we want to surf-over
                    #   --tval is the output file  
                    subprocess.call("mri_label2label --srcsubject fsaverage --trgsubject {2}{3} --regmethod surface --hemi {0} \
                        --srclabel {1}/{5}/{0}.MPM_{4}.label --trglabel ./{0}.{4}_TEMP.label".format(hemi,atlasdir,sub,suffix,roi,outname), shell=True)
                        
                    # convert to gifti
                    subprocess.call("mris_convert --label {0}.{1}_TEMP.label {1} ./surf/{0}.white {0}.{1}_TEMP.gii".format(hemi,roi), shell=True)
                    
                    # convert to .niml.dset
                    subprocess.call("ConvertDset -o_niml_asc -input {0}.{1}_TEMP.gii -prefix {0}.{1}_TEMP.niml.dset".format(hemi,roi), shell=True)

                    # isolate roi of interest
                    # do clustering, only consider cluster if they are 1 edge apart
                    subprocess.call("SurfClust -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.asc -input {0}.{1}_TEMP.niml.dset 0 \
                        -rmm -1 -prefix {0}.{1}_TEMP2.niml.dset -out_fulllist -out_roidset".format(hemi,roi,sub,suffix), shell=True)
                
                
                    # create mask, pick only biggest cluster
                    subprocess.call("3dcalc -a {0}.{1}_TEMP2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)' -prefix {0}.{1}_TEMP3.niml.dset".format(hemi,roi), shell=True)
                
                    # dilate mask
                    subprocess.call("ROIgrow -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.asc -roi_labels {0}.{1}_TEMP3.niml.dset -lim 1 -prefix {0}.{1}_TEMP4"
                        .format(hemi,roi,sub,suffix), shell=True)
                    
                    numnodes = subprocess.check_output("3dinfo -ni {0}.{1}_TEMP3.niml.dset".format(hemi,roi), shell=True)
                    numnodes = int(numnodes.rstrip("\n"))
                    print(numnodes)
                    numnodes = numnodes - 1
                    subprocess.call("ConvertDset -o_niml_asc -i_1D -input {0}.{1}_TEMP4.1.1D -prefix {0}.{1}_TEMP4.niml.dset -pad_to_node {2} -node_index_1D {0}.{1}_TEMP4.1.1D[0]"
                        .format(hemi,roi,numnodes), shell=True)

                    if idx == 1:
                        subprocess.call("3dcalc -a {0}.{1}_TEMP4.niml.dset -expr 'notzero(a)' -prefix {0}.{2}.niml.dset".format(hemi,roi,outname), shell=True)
                    else:
                        subprocess.call("3dcalc -a {0}.{1}_TEMP4.niml.dset -b {0}.{2}.niml.dset \
                            -expr '(b+notzero(a)*{3})*iszero(and(notzero(b),notzero(a)))' -prefix {0}.{1}_TEMP5.niml.dset".format(hemi,roi,outname,idx), shell=True)
                        shutil.move("{0}.{1}_TEMP5.niml.dset".format(hemi,roi), "{0}.{1}.niml.dset".format(hemi,outname))
                shutil.move("{0}.{1}.niml.dset".format(hemi,outname), "./TEMPLATE_ROIS/{0}.{1}.niml.dset".format(hemi,outname))
                #convert from niml.dset to gii
                shell_cmd("ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prefix ./TEMPLATE_ROIS/{0}.{1}.gii".format(hemi,outname))

        os.chdir(curdir)

        if os.path.isdir(outdir):
            print "Output directory {0} exists, adding '_new'".format(outdir) 
            shutil.move("{0}/TEMPLATE_ROIS".format(tmpdir), "{0}_new".format(outdir)) 
        else:
            shutil.move("{0}/TEMPLATE_ROIS".format(tmpdir), "{0}".format(outdir)) 
        if keeptemp is not True:
            # remove temporary directory
            shutil.rmtree(tmpdir)
    #reset the subjects dir
    os.environ["SUBJECTS_DIR"] = old_subject_dir

def MriSurfSmooth(subject,infiles,fsdir=None,std141=None,blursize=0,keeptemp=False,outdir="standard"):
    """
    Function smooths surface MRI data
    - this takes in regular and standardised data
    """
    if fsdir is None:
        fsdir = os.environ["SUBJECTS_DIR"]
     # check if subjects' freesurfer directory exists
    if os.path.isdir("{0}/{1}".format(fsdir,subject)):
        # no suffix needed
        suffix=""
    else:
        # suffix needed
        suffix="_fs4"
        if not os.path.isdir("{0}/{1}{2}".format(fsdir,subject,suffix)):
            sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                .format(fsdir,subject,suffix))
    if std141:
        specprefix = "std.141."
    else:
        specprefix = ""
    # create current,temporary,output directories
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    
    #Need to raise with Peter if this is okay - this will output files straight to end location
    # may also be that this breaks the BIDS format
    if outdir =="standard":
        outdir = "{0}/{1}{2}/".format(fsdir,subject,suffix)
    
    # copy relevant SUMA files
    sumadir = "{0}/{1}{2}/SUMA".format(fsdir,subject,suffix)
    print(sumadir)
    for file in glob.glob(sumadir + "/*h.smoothwm.asc"):
        shutil.copy(file,tmpdir)
    for file in glob.glob("{0}/{1}{2}{3}*.spec".format(sumadir,specprefix,subject,suffix)):
        shutil.copy(file,tmpdir)
    
    os.chdir(tmpdir)
    
    print(infiles)
    
    for curfile in infiles:
        if ".niml.dset" in curfile:
            filename = curfile[:-10]
            splitString = filename.split(".",1)
            #print(splitString)
            hemi = splitString[0]
            #print(hemi)
            outname = "{0}_{1}fwhm".format(filename,blursize)
        else:
            sys.exit("ERROR!\n{0} is not .niml.dset format".format(curfile))
        
        # move files
        subprocess.call("3dcopy {1}/{0}.niml.dset {0}.niml.dset".format(filename,curdir),shell=True)
        print("files moved")
        # compute mean
        subprocess.call("SurfSmooth -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -met HEAT_07 -target_fwhm {4} -input {5}.niml.dset \
                    -cmask '-a {5}.niml.dset[0] -expr bool(a)' -output  {8}/{7}.niml.dset"
                    .format(specprefix,subject,suffix,hemi,blursize,filename,tmpdir,outname,outdir), shell=True)
        print("mean computed")
    os.chdir(curdir)
    if keeptemp is not True:
        #remove temporary directory
        shutil.rmtree(tmpdir)        

def MriFFT(signal,tr=2.0,stimfreq=10,nharm=5,offset=0):
    """
    offset=0, positive values means the first data frame was shifted forwards relative to stimulus
              negative values means the first data frame was shifted backwards relative to stimulus
    """
    # define output object
    class fftobject:
        def __init__(self):
            for key in [ "spectrum", "frequencies", "mean_cycle", "sig_zscore", "sig_snr", 
                        "sig_amp", "sig_phase", "sig_complex", "noise_complex", "noise_amp", "noise_phase" ]:
                setattr(self, key, [])
    output = fftobject()
    if len(signal) == 0:
        return output
    nT = signal.size
    sample_rate = 1/tr
    run_time = tr * nT
    
    complex_vals = np.fft.rfft(signal,axis=0)
    complex_vals = complex_vals.reshape(int(nT/2+1),1)
    complex_vals = complex_vals[1:]/nT
    freq_vals = np.fft.rfftfreq(nT, d=1./sample_rate)
    freq_vals = freq_vals.reshape(int(nT/2+1),1)
    freq_vals = freq_vals[1:]
    
    # compute full spectrum
    output.spectrum = np.abs(complex_vals)
    output.frequencies = freq_vals
    
    # compute mean cycle
    cycle_len = nT/stimfreq
    nu_signal = signal
    # this code should work whether offset is 0, negative or positive
    pre_add = int(offset % cycle_len)
    post_add = int(cycle_len-pre_add)
    # add "fake cycle" by adding nans to the beginning and end of time series
    pre_nans = np.empty((1,pre_add,)).reshape(pre_add,1)
    pre_nans[:]=np.nan
    nu_signal = np.insert(nu_signal, 0, pre_nans, axis=0)
    post_nans = np.empty((1,post_add,)).reshape(post_add,1)
    post_nans[:]=np.nan
    nu_signal = np.append(nu_signal,post_nans)
    # reshape, add one to stimfreq to account for fake cycle
    nu_signal = nu_signal.reshape(int(stimfreq+1), int(nu_signal.shape[0] / (stimfreq+1)) )
    # nan-average to get mean cycle
    mean_cycle = np.nanmean(nu_signal, axis = 0).reshape( int(signal.shape[0] / stimfreq),1)
    # zero the mean_cycle    
    mean_cycle = mean_cycle - np.mean(mean_cycle[0:2])
    assert cycle_len == len(mean_cycle), "Mean cycle length {0} is different from computed cycle length {1}".format(len(mean_cycle),cycle_len)
    output.mean_cycle = mean_cycle

    for harm in range(1,nharm+1):
        idx_list = freq_vals == (stimfreq * harm) / run_time

        # Calculate Z-score
        # Compute the mean and std of the non-signal amplitudes.  Then compute the
        # z-score of the signal amplitude w.r.t these other terms.  This appears as
        # the Z-score in the plot.  This measures how many standard deviations the
        # observed amplitude differs from the distribution of other amplitudes.
        sig_zscore = (output.spectrum[idx_list]-np.mean(output.spectrum[np.invert(idx_list)])) / np.std(output.spectrum[np.invert(idx_list)])
        output.sig_zscore.append( sig_zscore )
        
        # calculate Signal-to-Noise Ratio. 
        # Signal divided by mean of 4 side bands
        signal_idx = int(np.where(idx_list)[0])
        noise_idx = [signal_idx-2,signal_idx-1,signal_idx+1,signal_idx+2]
        sig_snr = output.spectrum[signal_idx] / np.mean(output.spectrum[noise_idx])
        output.sig_snr.append( sig_snr )
        
        # compute complex, amp and phase
        # compute offset in radians, as fraction of cycle length
        offset_rad = float(offset)/cycle_len*(2*np.pi)
        
        sig_complex, sig_amp, sig_phase = fft_offset(complex_vals[signal_idx],offset_rad)
        noise_complex, noise_amp, noise_phase = fft_offset(complex_vals[noise_idx],offset_rad)

        output.sig_complex.append( sig_amp )
        output.sig_amp.append( sig_amp )
        output.sig_phase.append( sig_phase )
        output.noise_complex.append( noise_amp )
        output.noise_amp.append( noise_amp )
        output.noise_phase.append( noise_phase )
    return output   

def RoiSurfData(surf_files, roi="wang", sub=False, pre_tr=0, offset=0, TR=2.0, roilabel=None, fsdir=os.environ["SUBJECTS_DIR"]):

    if not sub:
        # determine subject from input data
        sub = surf_files[0][(surf_files[0].index('sub-')+4):(surf_files[0].index('sub-')+8)]
    elif "sub" in sub:
        # get rid of "sub-" string
        sub = sub[4:]
    print("SUBJECT:" + sub)
    # check if data from both hemispheres can be found in input
    # check_data
    l_files = []
    r_files = []
    for s in surf_files:
        if ".R." in s:
            s_2 = s.replace('.R.','.L.')
            r_files.append(s)
        elif ".L." in s:
            s_2 = s.replace('.L.','.R.')
            l_files.append(s)
        else:
            print("Hemisphere could not be determined from file %s" % s)
            #return
        if s_2 not in surf_files:
            print("File %s does not have a matching file from the other hemisphere" % s)
            #return
    l_files = sorted(l_files)
    r_files = sorted(r_files)

    # define roi files
    if roi.lower() == "wang":
        l_roifile = "{0}/sub-{1}/surf/lh.wang2015_atlas.mgz".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        roilabel = ["V1v", "V1d","V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                "IPS5", "SPL1", "FEF"]
        newlabel = ["V1v", "V1d", "V1","V2v", "V2d", "V2", "V3v", "V3d", "V3", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                "IPS5", "SPL1", "FEF"]
    elif roi.lower() == "benson":
        l_roifile = "{0}/sub-{1}/surf/lh.template_areas.mgz".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        roilabel = ["V1","V2","V3"]
        newlabel = roilabel
    elif roi.lower() == "wang+benson":
        l_roifile = "{0}/sub-{1}/surf/lh.wang2015_atlas.mgz".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        l_eccenfile = "{0}/sub-{1}/surf/lh.template_eccen.mgz".format(fsdir,sub)
        r_eccenfile = l_eccenfile.replace("lh","rh")
        # define roilabel based on ring centers
        ring_incr = 0.25
        ring_size = .5
        ring_max = 6
        ring_min = 1
        ring_centers = np.arange(ring_min, ring_max, ring_incr) # list of ring extents
        ring_extents = [(x-ring_size/2,x+ring_size/2) for x in ring_centers ]
        roilabel = [ y+"_{:0.2f}".format(x) for y in ["V1","V2","V3"] for x in ring_centers ]
        newlabel = roilabel
    else:
        #This is reference before assignment
        if ".L." in l_roifile:
            l_roifile = l_roifile
            r_roifile = r_roifile.replace('.L.','.R.')
        elif ".R." in r_roifile:
            r_roifile = r_roifile
            l_roifile =l_roifile.replace('.R.','.L.')
        elif "lh" in l_roifile:
            l_roifile = l_roifile
            r_roifile = r_roifile.replace('lh','rh')
        elif "rh" in r_roifile:
            r_roifile = r_roifile
            l_roifile = l_roifile.replace('rh','lh')
        print("Unknown ROI labels, using numeric labels")
        max_idx = int(max(surface.load_surf_data(l_roifile))+1)
        newlabel = ["roi_{:02.0f}".format(x) for x in range(1,max_idx)]
    
    outnames = [x+"-L" for x in newlabel] + [x+"-R" for x in newlabel] + [x+"-BL" for x in newlabel]
    # create a list of outdata, with shared values 
    outdata = [ roiobject(roiname=x,tr=TR,nharm=5,stimfreq=10) for x in outnames ]
    
    print("APPLYING RH ROI: " + l_roifile.split("/")[-1] + " TO DATA:")
    for x in l_files: print(x.split("/")[-1])
    print("APPLYING LH ROI: " + r_roifile.split("/")[-1] + " TO DATA:")
    for x in r_files: print(x.split("/")[-1])
        
    for hemi in ["L","R"]:
        if "L" in hemi:
            cur_roi = l_roifile
            cur_files = l_files
        else:
            cur_roi = r_roifile
            cur_files = r_files
        try:
            roi_data = surface.load_surf_data(cur_roi)
        except OSError as err:
            print("ROI file: {0} could not be opened".format(cur_roi))
        roi_n = roi_data.shape[0]
        
        if roi.lower() == "wang+benson":
            if "L" in hemi:
                cur_eccen = l_eccenfile
            else:
                cur_eccen = r_eccenfile
            try:
                eccen_data = surface.load_surf_data(cur_eccen)
            except OSError as err:
                print("Template eccen file: {0} could not be opened".format(cur_eccen))
            eccen_n = eccen_data.shape[0]
            assert eccen_n == roi_n, "ROIs and Template Eccen have different number of surface vertices"
            ring_data = np.zeros_like(roi_data)
            for r, evc in enumerate(["V1","V2","V3"]):
                # find early visual cortex rois in wang rois
                wanglabel = ["V1v", "V1d","V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                    "IPS5", "SPL1", "FEF"]
                roi_set = set([i+1 for i, s in enumerate(wanglabel) if evc in s])
                roi_index = [i for i, item in enumerate(roi_data) if item in roi_set]
                # define indices based on ring extents
                for e, extent in enumerate(ring_extents):
                    eccen_idx = np.where((eccen_data > extent[0]) * (eccen_data < extent[1]))[0]
                    idx_val = e + (r*len(ring_centers))
                    ready_idx = list(set(eccen_idx) & set(roi_index))
                    ring_data[ready_idx] = idx_val+1
            # now set ring values as new roi values
            roi_data = ring_data
        for run_file in cur_files:
            try:
                cur_data = surface.load_surf_data(run_file)
            except OSError as err:
                print("Data file: {0} could not be opened".format(run_file))

            data_n = cur_data.shape[0]
            assert data_n == roi_n, "Data and ROI have different number of surface vertices"
            for roi_name in newlabel:
                # note,  account for one-indexing of ROIs
                roi_set = set([i+1 for i, s in enumerate(roilabel) if roi_name in s])
                roi_index = [i for i, item in enumerate(roi_data) if item in roi_set]
                num_vox = len(roi_index)
                if num_vox == 0:
                    print(roi_name+"-"+hemi+" "+str(roi_set))
                roi_t = np.mean(cur_data[roi_index],axis=0)
                roi_t = roi_t[pre_tr:]
                out_idx = outnames.index(roi_name+"-"+hemi)
                outdata[out_idx].num_vox = num_vox
                outdata[out_idx] = roiobject(roi_t, outdata[out_idx])
                if "R" in hemi and run_file == cur_files[-1]:
                    # do bilateral
                    other_idx = outnames.index(roi_name+"-"+"L")
                    bl_idx = outnames.index(roi_name+"-"+"BL")
                    bl_data = [ (x + y)/2 for x, y in zip(outdata[other_idx].data, outdata[out_idx].data) ]
                    num_vox = np.add(outdata[other_idx].num_vox, outdata[out_idx].num_vox)
                    outdata[bl_idx].num_vox = num_vox
                    for bl_run in bl_data:
                        outdata[bl_idx] = roiobject(bl_run, outdata[bl_idx])
    return (outdata, outnames)

def HotT2Test(in_vals, alpha=0.05,test_mu=np.zeros((1,1), dtype=np.complex), test_type="Hot"):
    assert np.all(np.iscomplex(in_vals)), "input variable is not complex"
    assert (alpha > 0.0) & (alpha < 1.0), "alpha must be between 0 and 1"

    # compare against zero?
    if in_vals.shape[1] == 1:
        in_vals = np.append(in_vals,np.zeros(in_vals.shape, dtype=np.complex),axis=1)
        num_cond = 1
    else:
        num_cond = 2
        assert all(test_mu) == 0, "when two-dimensional complex input provided, test_mu must be complex(0,0)"
    assert in_vals.shape[1] <= 2, 'length of second dimension of complex inputs may not exceed two'

    # replace NaNs
    in_vals = in_vals[~np.isnan(in_vals)]
    # determine number of trials
    M = int(in_vals.shape[0]/2);
    in_vals = np.reshape(in_vals, (M,2))
    p = 2; # number of variables
    df1 = p;  # numerator degrees of freedom.

    if "hot" in test_type.lower():
        # subtract conditions
        in_vals = np.reshape(np.subtract(in_vals[:,0],in_vals[:,1]),(M,1))
        df2 = M-p; # denominator degrees of freedom.
        in_vals = np.append(np.real(in_vals),np.imag(in_vals),axis=1)
        samp_mu = np.mean(in_vals,0)
        test_mu = np.append(np.real(test_mu),np.imag(test_mu))
        samp_cov_mat = np.cov(in_vals[:,0],in_vals[:,1])

        # Eqn. 2 in Sec. 5.3 of Anderson (1984), multiply by inverse of fraction used below::
        t_crit = np.float(( (M-1) * p )/ ( df2 ) * scp.stats.f.ppf( 1-alpha, df1, df2 )); 
        #try
        inv_cov_mat  = np.linalg.inv(samp_cov_mat)
        # Eqn. 2 of Sec. 5.1 of Anderson (1984):
        tsqrd = np.float(np.matmul(np.matmul(M * (samp_mu - test_mu) , inv_cov_mat) , np.reshape(samp_mu - test_mu,(2,1))))
        # F approximation 
        tsqrdf = df2/( (M-1) * p ) * tsqrd; 
        # use scipys F cumulative distribution function.
        p_val = np.float(1 - scp.stats.f.cdf(tsqrdf, df1, df2))
    else:
        # note, if two experiment conditions are to be compared, we assume that the number of samples is equal
        df2 = num_cond * (2*M-p); # denominator degrees of freedom.

        # compute estimate of sample mean(s) from V & M 1991 
        samp_mu = np.mean(in_vals,0)

        # compute estimate of population variance, based on individual estimates
        v_indiv = 1/df2 * ( np.sum( np.square( np.abs(in_vals[:,0]-samp_mu[0] ) ) )  
                          + np.sum( np.square( np.abs(in_vals[:,1]-samp_mu[1] ) ) ) )

        if num_cond == 1:
            # comparing against zero
            v_group = M/p * np.square(np.abs(samp_mu[0]-samp_mu[1]))
            # note, distinct multiplication factor
            mult_factor = 1/M
        else:
            # comparing two conditions
            v_group = (np.square( M ))/( 2 * ( M * 2 ) ) * np.square(np.abs(samp_mu[0]-samp_mu[1]))
            # note, distinct multiplication factor
            mult_factor = ( M * 2 )/(np.square( M ))

        # Find critical F for corresponding alpha level drawn from F-distribution F(2,2M-2)
        # Use scipys percent point function (inverse of `cdf`) for f
        # multiply by inverse of multiplication factor to get critical t_circ
        t_crit = scp.stats.f.ppf(1-alpha,df1,df2) * (1/mult_factor)
        # compute the tcirc-statistic
        tsqrd = (v_group/v_indiv) * mult_factor;
        # M x T2Circ ( or (M1xM2/M1+M2)xT2circ with 2 conditions)
        # is distributed according to F(2,2M-2)
        # use scipys F probability density function
        p_val = 1-scp.stats.f.cdf(tsqrd * (1/mult_factor),df1,df2);            
    return(tsqrd,p_val,t_crit)
