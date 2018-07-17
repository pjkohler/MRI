import os, subprocess, sys, glob, shutil, tempfile, re, stat
from os.path import expanduser
from nilearn import datasets, surface
import numpy as np
import pandas as pd
import scipy as scp
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as special
from itertools import combinations
from sklearn.linear_model import LinearRegression

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

def copy_suma_files(suma_dir,tmp_dir,subject,suffix="",spec_prefix=""):
    for files in glob.glob(suma_dir+"/*h.smoothwm.gii"):
        shutil.copy(files,tmp_dir)
    for files in glob.glob(suma_dir+"/*h.pial.gii"):
        shutil.copy(files,tmp_dir)
    for files in glob.glob(suma_dir+"/*h.smoothwm.asc"):
        shutil.copy(files,tmp_dir)
    for files in glob.glob(suma_dir+"/*h.pial.asc"):
        shutil.copy(files,tmp_dir)
    for files in glob.glob("{0}/{1}{2}{3}*.spec".format(suma_dir,spec_prefix,subject,suffix)):
        shutil.copy(files,tmp_dir)
    # for some reason, 3dVol2Surf requires these files, so copy them as well
    for files in glob.glob(suma_dir+"/*aparc.*.annot.niml.dset"):
        shutil.copy(files,tmp_dir)

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

def realImagSplit(sig_complex):
    # takes complex data and splits into real and imaginary
    sig_complex=np.concatenate((np.real(sig_complex),np.imag(sig_complex)),axis=1)
    return sig_complex

def eigFourierCoefs(xyData):
    """Performs eigenvalue decomposition
    on 2D data. Function assumes 2D
    data are Fourier coefficients
    with real coeff in 1st col and 
    imaginary coeff in 2nd col.
    """
    # ensure data is in Array
    xyData = np.array(xyData)
    m, n = xyData.shape
    if n != 2:
        # ensure data is split into real and imaginary
        xyData = realImagSplit(xyData)
        m, n = xyData.shape
        if n != 2:
            print('Data in incorrect shape - please correct')
            return None
    realData = xyData[:,0]
    imagData = xyData[:,1]
    # mean and covariance
    meanXy= np.mean(xyData,axis=0)
    sampCovMat = np.cov(np.array([realData, imagData]))
    # calc eigenvalues, eigenvectors
    eigenval, eigenvec = np.linalg.eigh(sampCovMat)
    # sort the eigenvector by the eigenvalues
    orderedVals = np.sort(eigenval)
    eigAscendIdx = np.argsort(eigenval)
    smaller_eigenvec = eigenvec[:,eigAscendIdx[0]]
    larger_eigenvec = eigenvec[:,eigAscendIdx[1]]
    smaller_eigenval = orderedVals[0]
    larger_eigenval = orderedVals[1]
    
    phi = np.arctan2(larger_eigenvec[1],larger_eigenvec[0])
    # this angle is between -pi & pi, shift to 0 and 2pi
    if phi < 0:
        phi = phi + 2*np.pi
    return (meanXy, sampCovMat, smaller_eigenvec, 
           smaller_eigenval,larger_eigenvec, 
           larger_eigenval, phi)

def subjectFmriData(sub, fmriFolder,std141=False,session='01'):
    fmridir= '{0}/{1}/ses-{2}/func/'.format(fmriFolder,sub,session)
    # returns files needed to run RoiSurfData
    if std141==True:
        return [fmridir+x for x in os.path.os.listdir(fmridir) if x[-3:]=='gii' and 'surf' in x and 'std141' in x]
    else:
        return [fmridir+x for x in os.path.os.listdir(fmridir) if x[-3:]=='gii' and 'surf' in x and 'std141' not in x]

def create_file_dictionary(experiment_fmri_dir):
    # creates file dictionary necessary for Vol2Surf
    subjects = [folder for folder in os.listdir(experiment_fmri_dir) if 'sub' in folder and '.html' not in folder]
    subject_session_dic = {subject: [session for session in os.listdir("{0}/{1}".format(experiment_fmri_dir,subject)) if 'ses' in session] for subject in subjects}
    subject_session_directory = []
    for subject in subjects:
        subject_session_directory += ["{0}/{1}/{2}/func".format(experiment_fmri_dir,subject,session) for session in subject_session_dic[subject]]
    files_dic = {directory : [files for files in os.listdir(directory) if 'preproc.nii.gz' in files] for directory in subject_session_directory}
    return files_dic
def make_hard_links(bidsdir,experiment,subjects,fsdir):
    """Function creates hardlinks from freesurfer directory to the experiment folder

    Parameters
    ------------
    bidsdir : string
        The directory for BIDS Analysis. Should contain the freesurfer folder and experiment folder.
    experiment : string
        Used for location of the experiment folder within the BIDS directory
    subjects : list of strings
        This is a list of the subjects that require hardlinks to be made
    fsdir : string
        The freesurfer directory
    Returns
    ------------
    checking_dic : dictionary
        Contains the source and destination of the files. Used for checking that the new directory 
        is actually a hard link of the old one.
    """
    checking_dic = {}
    for sub in subjects:
        src = "{0}/{1}".format(fsdir,sub)
        dst = "{0}/{1}/freesurfer/{2}".format(bidsdir,experiment,sub)
        os.link(src,dst)
        checking_dic[sub] = [src,dst]
    check_hard_links(checking_dic)
    return checking_dic
def check_hard_links(checking_dic):
    correct_int = 0
    error_log = []
    for key in checking_dic.keys():
        l1 = checking_dic[key][0]
        l2 = checking_dic[key][1]
        if (l1[stat.ST_INO],l1[stat.ST_DEV]) == (l2[stat.ST_INO], l2[stat.ST_DEV]):
            correct_int+=1
        else:
            error_log.append(l2)
    if correct_int == len(checking_dic):
        print('All new files are hardlinks')
    else:
        print('Files not hard link: \n {0}'.format(error_log))   
    

## CLASSES

class roiobject:
    def __init__(self, curdata=np.zeros((120, 1)), curobject=None, is_time_series=True, roiname="unknown", tr=99, stimfreq=99, nharm=99, num_vox=0, offset=0):
        if curobject is None:
            self.data = []
            self.roi_name = roiname
            self.tr = tr
            self.stim_freq = stimfreq
            self.num_harmonics = nharm
            self.num_vox = num_vox
            self.is_time_series=is_time_series
            self.offset=offset
        else:
            # if curobject provided, inherit all values
            self.data = curobject.data
            self.roi_name = curobject.roi_name
            self.tr = curobject.tr
            self.stim_freq = curobject.stim_freq
            self.num_harmonics = curobject.num_harmonics
            self.num_vox = curobject.num_vox
            self.is_time_series=curobject.is_time_series
            self.offset = curobject.offset
        if curdata.any():
            self.data.append( curdata.reshape(curdata.shape[0],1) )
            self.mean = self.average()
            if self.is_time_series == True:
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
            nharm=self.num_harmonics,
            offset=self.offset)
# define output object
class fftobject:
    def __init__(self):
        for key in [ "spectrum", "frequencies", "mean_cycle", "sig_zscore", "sig_snr", 
                    "sig_amp", "sig_phase", "sig_complex", "noise_complex", "noise_amp", "noise_phase" ]:
            setattr(self, key, [])
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

def Vol2Surf(experiment_fmri_dir, fsdir=os.environ["SUBJECTS_DIR"], subjects=None, sessions=None, map_func='ave', wm_mod=0.0, gm_mod=0.0, prefix=None, index='voxels', steps=10, mask=None, surf_vol='standard', std141=False, keep_temp=False):
    """
    Function for converting from volume to surface space.  
    Supports suma surfaces both in native and std141 space.
    Surface volume can be given using the --surf_vol argument.
    Various other options from 3dVol2Surf are implemented, 
    sometimes with names that are more meaningful (to me).
    'data' option for mask still needs to be implemented. 
    
    For additional information on 3dVol2Surf parameters please
    see: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dVol2Surf.html

    Author: pjkohler, Stanford University, 2016
    Updated : fhethomas, 2018
    
    Parameters
    ------------
    experiment_fmri_dir : string
        The directory for the fmri data for the experiment
        Example: '/Volumes/Computer/Users/Username/Experiment/fmriprep'
    fsdir : string, default os.environ["SUBJECTS_DIR"]
        Freesurfer directory
    subjects : list of strings, Default None
        Optional parameter, if wish to limit function to only certain subjects
        that are to be run. Example : ['sub-0001','sub-0002']
    sessions : list of strings, Default None
        Optional parameter, if wish to limit function to only certain sessions
    map_func : string, default 'ave'
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        map_func. Options - 'ave', 'mask', 'seg_vals'
    wm_mod : float, default 0.0
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_p1_fr. This specifies a change to point p1 in direction
        of point pn. Change is a fraction i.e. -0.2 & 0.2
    gm_mod : float, default 0.0
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_pn_fr. To extend segment past pn fraction will be postive.
        To reduce segment back to p1 will be negative
    prefix : string, default None
        File may need a specific prefix
    index : string, default 'voxels'
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_index. Specifies whether to use all seg points or
        unique volume voxels. Options: nodes, voxels
    steps : integer, default 10
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_steps. Specify number of evenly spaced points along
        each segment
    mask : string, default None
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        cmask. Produces a mask to be applied to input AFNI dataset.
    surf_vol : string, default 'standard'
        File location of volume directory/file
    std141 : Boolean, default False
        Is subject to be run with standard 141?
    keep_temp : Boolean, default False
        Should temporary folder that is set up be kept after function 
        runs?
    Returns 
    ------------
    file_list : list of strings
        This is a list of all files created by the function
    """
    # Create a dictionary of files - keys are the directory for each session
    file_dictionary = create_file_dictionary(experiment_fmri_dir)
    # Remove unwanted subjects and sessions
    if subjects != None:
        file_dictionary = {directory : file_dictionary[directory] for directory in file_dictionary.keys() 
                           if directory[len(experiment_fmri_dir)+1:].split('/')[0] in subjects}
    if sessions != None:
        file_dictionary = {directory : file_dictionary[directory] for directory in file_dictionary.keys()
                          if directory[len(experiment_fmri_dir)+1:].split('/')[1] in sessions}
    # list of files created by this function
    file_list = []
    # Dict to convert between old and new hemisphere notation
    hemi_dic = {'lh' : 'L', 'rh' : 'R'}
        
    # Iterate over subjects
    for directory in file_dictionary.keys():
        # pull out subject title - i.e. 'sub-0001'
        subject = directory[len(experiment_fmri_dir)+1:].split('/')[0]
        print('Running subject: {0}'.format(subject))
        cur_dir = directory
        in_files = file_dictionary[directory]
        # Define the names to be used for file output
        criteria_list = ['sub','ses','task','run','space']
        input_format = 'bids'
        output_name_dic = {}
        for cur_file in in_files:
            file_name, file_suffix = get_name_suffix(cur_file)
            # Checking for bids formatting
            if not sum([crit in file_name for crit in criteria_list]) == len(criteria_list):
                input_format = 'non-bids'
            if input_format == 'bids':
                old = re.findall('space-\w+_',file_name)[0]
                if std141 == False:
                    new = 'space-surf.native_'
                else:
                    new = 'space-surf.std141_'
                output_name_dic[file_name] = file_name.replace(old,new)
            else:
                if std141 == False:
                    new = 'space-surf.native'
                elif std141 == True:
                    new = 'space-surf.std141'
                output_name_dic[file_name] = '{0}_{1}'.format(file_name, new)

        # make temporary, local folder
        tmp_dir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))   

        # check if subjects' SUMA directory exists
        suffix = fs_dir_check(fsdir,subject)
        suma_dir = "{0}/{1}{2}/SUMA".format(fsdir,subject,suffix)


        if wm_mod is not 0.0 or gm_mod is not 0.0:
            # for gm, positive values makes the distance longer, for wm negative values
            steps = round(steps + steps * gm_mod - steps * wm_mod)

        print("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod,gm_mod,steps))

        if surf_vol is "standard":
            vol_dir = "{0}/{1}{2}/SUMA".format(fsdir,subject,suffix) 
            vol_file = "{0}{1}_SurfVol.nii".format(subject,suffix)
        else:
            vol_dir = '/'.join(surf_vol.split('/')[0:-1])
            vol_file = surf_vol.split('/')[-1]
            if not vol_dir: # if volume directory is empty
                vol_dir = cur_dir

        # make temporary copy of volume file     
        subprocess.call("3dcopy {0}/{1} {2}/{1}".format(vol_dir,vol_file,tmp_dir), shell=True)

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
        # copy Suma files into the temporary directory
        copy_suma_files(suma_dir,tmp_dir,subject,spec_prefix=specprefix)

        os.chdir(tmp_dir)
        for cur_file in in_files:
            file_name, file_suffix = get_name_suffix(cur_file)
            # unzip the .nii.gz files into .nii files
            shell_cmd("gunzip -c {0}/{1}.nii.gz > {2}/{1}.nii".format(cur_dir,file_name,tmp_dir),do_print=False)
            if mask is None:
                # no mask
                maskcode = ""
            else:
                if mask is 'data':
                    # mask from input data
                    maskcode = "-cmask '-a {0}[0] -expr notzero(a)' ".format(file_name)
                else:
                    # mask from distinct dataset, copy mask to folder
                    mask_name, mask_suffix = get_name_suffix(mask)
                    subprocess.call("3dcopy {1}/{0}{2} mask+orig".format(mask_name,cur_dir,mask_suffix), shell=True)
                    maskcode = "-cmask '-a mask+orig[0] -expr notzero(a)' "
            for hemi in ["lh","rh"]:
                output_file_name = "{0}.{1}.func".format(output_name_dic[file_name],hemi_dic[hemi])
                # Converts volume to surface space - output in .niml.dset
                shell_cmd("3dVol2Surf -spec {0}{1}{2}_{3}.spec \
                        -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {5}.nii -map_func {6} \
                        -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                        -outcols_NSD_format -oob_value -0 {12}-out_niml {14}/{13}.niml.dset"
                        .format(specprefix,subject,suffix,hemi,vol_file,file_name,map_func,index,wm_mod,gm_mod,steps,cur_dir,maskcode,output_file_name,tmp_dir), do_print=False)
                # Converts the .niml.dset into a .gii file in the functional directory
                shell_cmd("ConvertDset -o_gii_asc -input {1}/{0}.niml.dset -prefix {2}/{0}.gii".format(output_file_name,tmp_dir,cur_dir),do_print=False)
                file_list.append('{1}/{0}'.format(output_file_name,cur_dir))            
        os.chdir(cur_dir) 
        if keep_temp is not True:
            # remove temporary directory
            shutil.rmtree(tmp_dir)
    print('Vol2Surf run complete')
    return file_list

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
    
    copy_suma_files(suma_dir,tmp_dir,subject)
    
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

def RoiTemplates(subjects, roi_type="all", atlasdir=None, fsdir=None, outdir="standard", forcex=False, separate_out=False, keeptemp=False, skipclust=False, intertype="NearestNode",force_new_mapping=False):
    """Function for generating ROIs in subject's native space 
    predicted from the cortical surface anatomy.
    Allows this to be done using methods described in:
    - Benson et al. (PLoS Comput Biol., 2014).
    - Glasser et al. (Nature, 2016)
    - Wang et al. (Neuron, 2015)
    - KGS (2016)
    Requires template data, which can be downloaded at:
    https://cfn.upenn.edu/aguirre/wiki/public:retinotopy_template

    Author: pjkohler, Stanford University, 2016
    Updates: fhethomas, 2018
    
    This function also generates ROIs based on Wang, Glasser and KGS methodology.

    Parameters
    ------------
    subjects : list of strings
            A list of subjects that are to be run.
    roi_type : string or list of strings, default "all"
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
            Can choose to separate out as part of Benson into ["angle", "eccen", 
            "areas", "all"]
    keeptemp : boolean, default False
            Option to keep the temporary files that are generated
    skipclust : boolean, default False
            If True then will do optional surface-based clustering as part
            of Wang
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
    # is a userinput output directory used?
    if outdir != 'standard':
        outdir_flag = 'custom'
    else:
        outdir_flag = outdir

    # get current directory    
    curdir = os.getcwd()

    # Assessing user input - need to identify which elements they want to run
    run_benson, run_glasser, run_wang, run_kgs = False, False, False, False
    run_possible_arguments=['benson','glasser','wang','kgs']
    roi_type = str(roi_type).lower()
    confirmation_str = 'Running: '
    if 'all' in roi_type:
        run_benson, run_glasser, run_wang, run_kgs = True, True, True, True
        confirmation_str += 'Benson, Glasser, Wang, KGS'
    elif [name for name in run_possible_arguments if name in roi_type]:
        if 'benson' in roi_type:
            run_benson = True
            confirmation_str += 'Benson, '
        if 'glasser' in roi_type:
            run_glasser = True
            confirmation_str += 'Glasser, '
        if 'wang' in roi_type:
            run_wang = True
            confirmation_str += 'Wang, '
        if 'kgs' in roi_type:
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
        # check the voxel size is even and res is 1x1x1
        vox_hdr = nib.load("{0}/{1}{2}/mri/orig.mgz".format(fsdir,sub,suffix)).header
        vox_shape = vox_hdr.get_data_shape()
        assert len([shape for shape in vox_shape if shape%2!=0])==0, 'Voxel Shape incorrect {0}'.format(vox_shape)
        vox_res = vox_hdr.get_zooms()
        assert vox_res == (1.0, 1.0, 1.0), 'Voxel Resolution incorrect: {0}'.format(vox_res)
        
        if outdir_flag != "custom":
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

        # Does SUMA folder exist - if not run @SUMA_Make_Spec_FS -NIFTI -sid subject1
        if not os.path.isdir(sumadir):
            print('Running @SUMA_Make_Spec_FS')
            os.chdir("{0}/{1}{2}".format(fsdir,sub,suffix))
            shell_cmd("@SUMA_Make_Spec_FS -NIFTI -sid {0}{1}".format(sub,suffix))
            file_format="gii"
        else:
            # is SUMA data in .gii or .asc format?
            if len(glob.glob("{0}/{1}{2}/SUMA/*.asc".format(fsdir,sub,suffix))) > 0:
                file_format="asc"
            elif len(glob.glob("{0}/{1}{2}/SUMA/*.gii".format(fsdir,sub,suffix))) > 0:
                file_format='gii'
            else:
                print('SUMA Error - no .asc or .gii files located')
                return None
        for file in glob.glob(sumadir+"/*h.smoothwm.{0}".format(file_format)):
            shutil.copy(file,tmpdir+"/SUMA")
        for file in glob.glob("{0}/{1}_*.spec".format(sumadir,sub)):
            shutil.copy(file,tmpdir+"/SUMA")
        # Copy existing mapping files
        mapfiles={}
        for file in glob.glob("{0}/{1}{2}.std141_to_native.*.niml.M2M".format(sumadir,sub,suffix)):
            shutil.copy(file,tmpdir+"/SUMA")
            if 'lh' in file:
                mapfiles['lh'] = file
            else:
                mapfiles['rh'] = file
        """
        #Currently not able to use this mapping file - potentially this could be used in future?
        for file in glob.glob("{0}/std.141.{1}{2}*.niml.M2M".format(sumadir,sub,suffix)):
            shutil.copy(file,tmpdir+"/SUMA")
            if len(mapfiles)!=2:
                if 'lh' in file:
                    mapfiles['lh'] = file
                else:
                    mapfiles['rh'] = file
        """

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
            surf_to_surf_i = 'fs' if file_format == 'asc' else 'gii'
            for hemi in ["lh","rh"]:
                # if you have a mapping file, this is much faster.  see SurfToSurf -help
                # you can still run without a mapping file, but it is generated on-the-fly (slow!)
                # mapping file may have already been generated - option 2 maybe generated
                try:
                    mapfile = mapfiles[hemi]
                except:
                    mapfile = ""
                if os.path.isfile(mapfile) and not force_new_mapping:

                    print("Using existing mapping file {0}".format(mapfile))
                    subprocess.call("SurfToSurf -i_{4} ./SUMA/{0}.smoothwm.{3} -i_{4} ./SUMA/std.141.{0}.smoothwm.{3} -output_params {1} -mapfile {2} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi,intertype,mapfile,file_format,surf_to_surf_i), shell=True)
                    newmap = False
                else:
                    print("Generating new mapping file")
                    newmap = True
                    subprocess.call("SurfToSurf -i_{3} ./SUMA/{0}.smoothwm.{2} -i_{3} ./SUMA/std.141.{0}.smoothwm.{2} -output_params {1} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi,intertype,file_format,surf_to_surf_i), shell=True)       
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
                subprocess.call("ConvertDset -o_1D -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prepend_node_index_1D -prefix ./TEMPLATE_ROIS/{0}.{1}.1D.dset"
                    .format(hemi, outname), shell=True)

                if not skipclust: # do optional surface-based clustering
                    print('######################## CLUSTERING ########################')
                    for idx in range(1,26):
                        # clustering steps
                        specfile="./SUMA/{0}{1}_{2}.spec".format(sub,suffix,hemi)  
                        surffile="./SUMA/{0}.smoothwm.{1}".format(hemi,file_format)
            
                        # isolate ROI
                        subprocess.call("3dcalc -a ./TEMPLATE_ROIS/{2}.{0}.niml.dset -expr 'iszero(a-{1})' -prefix {2}.temp.niml.dset"
                            .format(outname, idx,hemi), shell=True)
                        # do clustering, only consider cluster if they are 1 edge apart
                        subprocess.call("SurfClust -spec {0} -surf_A {1} -input {2}.temp.niml.dset 0 -rmm -1 -prefix {2}.temp2 -out_fulllist -out_roidset"
                            .format(specfile,surffile,hemi), shell=True)
                            
                        # pick only biggest cluster
                        if idx is 1:
                            if os.path.isfile("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,hemi)):
                                print("Removing existing file ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,hemi)) 
                                os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,hemi))
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
                    subprocess.call("ConvertDset -input {1}.{0}_cluster.niml.dset -o_niml_asc -prefix ./TEMPLATE_ROIS/{1}.temp4.niml.dset"
                        .format(outname,hemi,idx), shell=True)
                    os.remove("{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    os.rename("./TEMPLATE_ROIS/{0}.temp4.niml.dset".format(hemi), "./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    #convert output to gii
                    shell_cmd("ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.gii".format(outname,hemi))
                # copy mapping file to subjects' home SUMA directory
                if newmap:            
                    shutil.move("./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi),
                                "{3}/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub,suffix,hemi,sumadir))
                #convert data set to asc
                shell_cmd("ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname,hemi),do_print=True)
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname,hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname,hemi),"./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname,hemi))
                if not skipclust:
                    shell_cmd("ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(outname,hemi))
                    os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,hemi))
                    os.rename("./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(outname,hemi),"./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,hemi))
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
                    subprocess.call("SurfClust -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.{4} -input {0}.{1}_TEMP.niml.dset 0 \
                        -rmm -1 -prefix {0}.{1}_TEMP2.niml.dset -out_fulllist -out_roidset".format(hemi,roi,sub,suffix,file_format), shell=True)
                
                
                    # create mask, pick only biggest cluster
                    subprocess.call("3dcalc -a {0}.{1}_TEMP2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)' -prefix {0}.{1}_TEMP3.niml.dset".format(hemi,roi), shell=True)
                
                    # dilate mask
                    subprocess.call("ROIgrow -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.{4} -roi_labels {0}.{1}_TEMP3.niml.dset -lim 1 -prefix {0}.{1}_TEMP4"
                        .format(hemi,roi,sub,suffix,file_format), shell=True)
                    
                    numnodes = subprocess.check_output("3dinfo -ni {0}.{1}_TEMP3.niml.dset".format(hemi,roi), shell=True)
                    numnodes = numnodes.decode('ascii')
                    numnodes = int(numnodes.rstrip("\n"))
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
                shell_cmd("ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname,hemi))
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname,hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname,hemi),"./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname,hemi))
        os.chdir(curdir)

        if os.path.isdir(outdir):
            print("Output directory {0} exists, adding '_new'".format(outdir))
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

        output.sig_complex.append( sig_complex )
        output.sig_amp.append( sig_amp )
        output.sig_phase.append( sig_phase )
        output.noise_complex.append( noise_complex )
        output.noise_amp.append( noise_amp )
        output.noise_phase.append( noise_phase )
    return output   

def RoiSurfData(surf_files, roi="wang", is_time_series=True, sub=False, pre_tr=2, offset=0, TR=2.0, roilabel=None, fsdir=os.environ["SUBJECTS_DIR"]):
    """
    region of interest surface data
    
    Parameters
    ------------
    surf_files : list of strings
            A list of files to be run - this should be surface 
            data in .gii format
    roi : string, default "wang"
            This defaults to "wang". Only acceptable inputs are
            "wang", "benson", "wang+benson"
    is_time_series : boolean, default True
            Data contain time series. FourierT will be carried out
            if True
    sub : boolean, default False
            The subject, if not clearly defined in the file name
    pre_tr : integer, default 0
            If this is required to slice off certain time region
    offset : integer, default 0
            If an Offset is required
    TR : float, default 2.0
            Repetition Time
    roilabel : Roi label, default None
            Region of Interest label if required
    fsdir : directory, default os.environ["SUBJECTS_DIR"]
            Freesurfer directory

    Returns
    ------------
    outdata : A list of objects
            outdata contains a list of objects (RoIs) with attributes:
                - data : The data relevant to this RoI
                - roi_name : The RoI name
                - tr : Repetition Time
                - stim_freq : Stimulus frequency
                - num_harmonics : Number of harmonics
                - num_vox : number of voxels
                - mean : Array of the averaged data
                - fft : fourierT with attributes [ "spectrum", 
                    "frequencies", "mean_cycle", "sig_zscore", "sig_snr", 
                    "sig_amp", "sig_phase", "sig_complex", "noise_complex",
                    "noise_amp", "noise_phase" ]
    outnames : A list of RoIs

    Raises
    ------------
    AssertError : Where surface vertices do not match
    """
    #dictionary of RoIs
    roi_dic={'wang': ["V1v", "V1d","V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                    "IPS5", "SPL1", "FEF"],
            'benson' : ["V1","V2","V3"],
            'wang_newlabel' : ["V1v", "V1d", "V1","V2v", "V2d", "V2", "V3v", "V3d", "V3", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                    "IPS5", "SPL1", "FEF"]
            }
    roi=roi.lower()
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
            return None
        if s_2 not in surf_files:
            print("File %s does not have a matching file from the other hemisphere" % s)
            
    l_files = sorted(l_files)
    r_files = sorted(r_files)

    # define roi files
    if roi == "wang":
        l_roifile = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        roilabel = roi_dic[roi]
        newlabel = roi_dic[roi+'_newlabel']
    elif roi == "benson":
        l_roifile = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        roilabel = roi_dic['benson']
        newlabel = roilabel
    elif roi == "wang+benson":
        l_roifile = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fsdir,sub)
        r_roifile = l_roifile.replace("lh","rh")
        l_eccenfile = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fsdir,sub)
        r_eccenfile = l_eccenfile.replace("lh","rh")
        # define roilabel based on ring centers
        ring_incr = 0.25
        ring_size = .5
        ring_max = 6
        ring_min = 1
        ring_centers = np.arange(ring_min, ring_max, ring_incr) # list of ring extents
        ring_extents = [(x-ring_size/2,x+ring_size/2) for x in ring_centers ]
        roilabel = [ y+"_{:0.2f}".format(x) for y in roi_dic['benson'] for x in ring_centers ]
        newlabel = roilabel
    else:
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
    outdata = [ roiobject(is_time_series=is_time_series,roiname=name,tr=TR,nharm=5,stimfreq=10,offset=offset) for name in outnames ]

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
            # uses surface module of nilearn to import data in .gii format
            roi_data = surface.load_surf_data(cur_roi)
            # Benson should just use ROIs
            if roi=='benson':
                roi_data=roi_data[:,2]
        except OSError as err:
            print("ROI file: {0} could not be opened".format(cur_roi))
        roi_n = roi_data.shape[0]
        
        if roi == "wang+benson":
            if "L" in hemi:
                cur_eccen = l_eccenfile
            else:
                cur_eccen = r_eccenfile
            try:
                eccen_data = surface.load_surf_data(cur_eccen)
                # select eccen data from Benson
                eccen_data=eccen_data[:,1]
            except OSError as err:
                print("Template eccen file: {0} could not be opened".format(cur_eccen))
            eccen_n = eccen_data.shape[0]
            assert eccen_n == roi_n, "ROIs and Template Eccen have different number of surface vertices"
            ring_data = np.zeros_like(roi_data)
            for r, evc in enumerate(roi_dic['benson']):
                # find early visual cortex rois in wang rois
                roi_set = set([i+1 for i, s in enumerate(roi_dic['wang']) if evc in s])
                # Find index of each roi in roi_data
                roi_index=np.array([])
                for item in roi_set:
                    roi_temp_index=np.array(np.where(roi_data==item)).flatten()
                    roi_index=np.concatenate((roi_index,roi_temp_index))
                roi_index=roi_index.astype(int)
                # define indices based on ring extents
                for e, extent in enumerate(ring_extents):
                    #get indexes that are populated in both i.e. between lower and higher extent
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
                # Find index of each roi in roi_data
                roi_index=np.array([])
                for item in roi_set:
                    roi_temp_index=np.array(np.where(roi_data==item)).flatten()
                    roi_index=np.concatenate((roi_index,roi_temp_index))
                roi_index=roi_index.astype(int)
                num_vox = len(roi_index)
                if num_vox == 0:
                    print(roi_name+"-"+hemi+" "+str(roi_set))
                roi_t = np.mean(cur_data[roi_index],axis=0)
                roi_t = roi_t[pre_tr:]
                out_idx = outnames.index(roi_name+"-"+hemi)
                outdata[out_idx].num_vox = num_vox
                outdata[out_idx] = roiobject(roi_t, curobject=outdata[out_idx])
                
                if "R" in hemi and run_file == cur_files[-1]:
                    # do bilateral
                    other_idx = outnames.index(roi_name+"-"+"L")
                    bl_idx = outnames.index(roi_name+"-"+"BL")
                    bl_data = [ (x + y)/2 for x, y in zip(outdata[other_idx].data, outdata[out_idx].data) ]
                    num_vox = np.add(outdata[other_idx].num_vox, outdata[out_idx].num_vox)
                    outdata[bl_idx].num_vox = num_vox
                    for bl_run in bl_data:
                        outdata[bl_idx] = roiobject(bl_run, outdata[bl_idx])
    print('ROI Surf Data run complete.')
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

def fitErrorEllipse(xyData, ellipseType='SEM', makePlot=False, returnRad=False):
    """ Function uses eigen value decomposition
    to find two perpendicular axes aligned with xyData
    where the eigen vector correspond to variances 
    along each direction. An ellipse is fit to
    data at a distance from mean datapoint,
    depending on ellipseType specified.
    
    Calculation for error ellipses based on
    alpha-specified confidence region (e.g. 
    95%CI or 68%CI) are calculated following
    information from Chapter 5 of Johnson & 
    Wickern (2007) Applied Multivatiate 
    Statistical Analysis, Pearson Prentice Hall
    
    Parameters
    ------------
    xyData : N x 2 matrix of 2D array
        Data contains real and imaginary data
        xyData should be [real, imag]
    ellipseType : string, default 'SEM'
        Options are - SEM, 95%CI, or specific
        different percentage in format 'x%CI'
    makePlot : Boolean, default False
        Specifies whether or not to generate
        a plot of the data & ellipse & eigen
        vector
    returnRad : Boolean, default False
        Specifies whether to return values
        in radians or degrees.
    Returns
    ------------
    ampDiff : numpy array,
            differences of lower and upper bound
            for the mean amplitude
    phaseDiff : numpy array,
            differences of lower and upper bound
            for the mean phase
    zSNR : float,
            z signal to noise ratio
    errorEllipse : numpy array,
            Array of error ellipses
    """
    xyData=np.array(xyData)
    
    # convert returnRad to an integer for indexing purposes later
    returnRad = int(returnRad)

    if len(xyData[np.iscomplex(xyData)])==len(xyData):
    
        xyData=realImagSplit(xyData)
    else:
        check=input('Data not complex. Should run continue? True/False')
        if check == False:
            print('Run stopped')
            return None
    n = xyData.shape[0]
    assert xyData.shape[1]==2, 'data should be of dimensions: N x 2, currently: {0}'.format(xyData.shape[1])
    try:
        (meanXy, sampCovMat, smaller_eigenvec, 
           smaller_eigenval,larger_eigenvec, 
           larger_eigenval, phi)=eigFourierCoefs(xyData)
    except:
        print('Unable to run eigen value decomposition. Probably data have only 1 sample')
        return None
    theta_grid = np.linspace(0,2*np.pi,num=100)
    if ellipseType == '1STD':
        a = np.sqrt(larger_eigenval)
        b = np.sqrt(smaller_eigenval)
    elif ellipseType == '2STD':
        a = 2*np.sqrt(larger_eigenval)
        b = 2*np.sqrt(smaller_eigenval)
    elif ellipseType == 'SEMarea':
        # scale ellipse's area by sqrt(n)
        a = np.sqrt(larger_eigenval/np.sqrt(n))
        b = np.sqrt(smaller_eigenval/np.sqrt(n))
    elif ellipseType == 'SEM'or ellipseType=='SEMellipse':
        # contour at stdDev/sqrt(n)
        a = np.sqrt(larger_eigenval)/np.sqrt(n)
        b = np.sqrt(smaller_eigenval)/np.sqrt(n)
    elif 'CI' in ellipseType:
        # following Eqn. 5-19 Johnson & Wichern (2007)
        try:
            critVal = float(ellipseType[:-3])/100
        except:
            print('EllipseType incorrectly formatted, please see docstring')
            return None
        assert critVal < 1.0 and critVal > 0.0,'EllipseType CI range must be between 0 & 100'
        t0_sqrd = ((n - 1) * 2)/(n * (n - 2)) * stats.f.ppf(critVal, 2, n - 2)
        a = sqrt(larger_eigenval * t0_sqrd)
        b = sqrt(smaller_eigenval * t0_sqrd)
    else:
        print('EllipseType Input incorrect, please see docstring')
        return None
    # the ellipse in x & y coordinates
    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_x_r = np.reshape(ellipse_x_r, (ellipse_x_r.shape[0], 1))
    ellipse_y_r = b * np.sin(theta_grid)
    ellipse_y_r = np.reshape(ellipse_y_r, (ellipse_y_r.shape[0], 1))
    
    # Define a rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]]) 
    # rotate ellipse to some angle phi
    errorEllipse = np.dot(np.concatenate((ellipse_x_r, ellipse_y_r), axis=1), R)
    # shift to be centered on mean coordinate
    errorEllipse = np.add(errorEllipse, meanXy)
    
    # find vector length of each point on ellipse
    norms = np.array([np.linalg.norm(errorEllipse[point,:]) for point in range(errorEllipse.shape[0])])
    ampMinNorm = min(norms)
    ampMinNormIx = np.argmin(norms)
    ampMaxNorm = max(norms)
    ampMaxNormIx = np.argmax(norms)
    norm_meanXy = np.linalg.norm(meanXy)
    ampDiff = np.array([norm_meanXy - ampMinNorm, ampMaxNorm - norm_meanXy])
    ampEllipseExtremes = np.array([ampMinNorm,ampMaxNorm])
    
    # calculate phase angles & find max pairwise difference to determine phase bounds
    phaseAngles = np.arctan2(errorEllipse[:,1], errorEllipse[:,0])
    pairs = np.array([np.array(comb) for comb in list(combinations(phaseAngles,2))])
    diffPhase = np.absolute(pairs[:,1] - pairs[:,0]) # find absolute difference of each pair
    diffPhase[diffPhase > np.pi] = 2 * np.pi - diffPhase[diffPhase > np.pi] # unwrap the difference
    maxDiffIdx = np.argmax(diffPhase)
    anglesOI = pairs[maxDiffIdx,:]
    phaseMinIx = np.argwhere(phaseAngles == anglesOI[0])[0]
    phaseMaxIx = np.argwhere(phaseAngles == anglesOI[1])[0]

    
    # convert to degrees (if necessary) & find diff between (max bound & mean phase) & (mean phase & min bound)
    # everything converted from [-pi, pi] to [0, 2*pi] for unambiguous calculation
    convFactor = np.array([180 / np.pi,1])
    unwrapFactor = np.array([360, 2 * np.pi])
    phaseEllipseExtremes = anglesOI*convFactor[returnRad]
    phaseEllipseExtremes[phaseEllipseExtremes < 0 ] = phaseEllipseExtremes[phaseEllipseExtremes < 0] + unwrapFactor[returnRad]
    
    phaseBounds = np.array([min(phaseEllipseExtremes),max(phaseEllipseExtremes)])
    meanPhase = np.arctan2(meanXy[1],meanXy[0]) * convFactor[returnRad]
    if meanPhase < 0:
        meanPhase = meanPhase + unwrapFactor[returnRad]
    
    # if ellipse overlaps with origin, defined by whether phase angles in all 4 quadrants
    phaseAngles[phaseAngles < 0] = phaseAngles[phaseAngles < 0] + 2*np.pi
    
    quad1 = phaseAngles[(phaseAngles > 0) & (phaseAngles < np.pi/2)]
    quad2 = phaseAngles[(phaseAngles > np.pi/2) & (phaseAngles < np.pi)]
    quad3 = phaseAngles[(phaseAngles > np.pi/2) & (phaseAngles < 3 * np.pi/2)]
    quad4 = phaseAngles[(phaseAngles > 3 * np.pi/2) & (phaseAngles < 2 * np.pi)]
    if len(quad1) > 0 and len(quad2) > 0 and len(quad3) > 0 and len(quad4) > 0:
        amplBounds = np.array([0, ampMaxNorm])
        maxVals = np.array([360, 2 * np.pi])
        phaseBounds = np.array([0, maxVals[returnRad]])
        phaseDiff = np.array([np.absolute(phaseBounds[0] - meanPhase), 
                              np.absolute([phaseBounds[1] - meanPhase])])
    else:
        amplBounds = ampEllipseExtremes
        phaseDiff = np.array([np.absolute(phaseBounds[0] - meanPhase), np.absolute(phaseBounds[1] - meanPhase)])
    
    # unwrap phase diff for any ellipse that overlaps with positive x axis
    
    phaseDiff[phaseDiff > unwrapFactor[returnRad] / 2] = unwrapFactor[returnRad] - phaseDiff[phaseDiff > unwrapFactor[returnRad]/2]
    
    zSNR = norm_meanXy / np.mean(np.array([norm_meanXy - ampMinNorm, ampMaxNorm - norm_meanXy]))
    
    # Data plot
    if makePlot:
        # Below makes 2 subplots
        plt.figure(figsize=(9,9))
        font={'size':16,'color':'k','weight':'light'}
        # Figure 1 - eigen vector & SEM ellipse
        plt.subplot(1,2,1)
        plt.plot(xyData[:,0],xyData[:,1],'ko',markerfacecolor='k')
        plt.plot([0,meanXy[0]],[0,meanXy[1]],linestyle = 'solid',color = 'k', linewidth = 1)
        # plot ellipse
        plt.plot(errorEllipse[:,0],errorEllipse[:,1],'b-',linewidth = 1, label = ellipseType + ' ellipse')
        # plot smaller eigen vec
        small_eigen_mean = [np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[0]) + meanXy[0],
                            np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[1]) + meanXy[1]]
        plt.plot([meanXy[0], small_eigen_mean[0]], [meanXy[1], small_eigen_mean[1]],'g-',
                 linewidth = 1, label = 'smaller eigen vec')
        # plot larger eigen vec
        large_eigen_mean = [np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[0]) + meanXy[0],
                            np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[1]) + meanXy[1]]
        plt.plot([meanXy[0], large_eigen_mean[0]], [meanXy[1], large_eigen_mean[1]],'m-',
                 linewidth = 1, label = 'larger eigen vec')
        # add axes
        plt.axhline(color = 'k', linewidth = 1)
        plt.axvline(color = 'k', linewidth = 1)
        plt.legend(loc=3,frameon=False)
        plt.axis('equal')
        
        # Figure 2 - mean amplitude, phase and amplitude bounds
        plt.subplot(1,2,2)
        # plot error Ellipse
        plt.plot(errorEllipse[:,0],errorEllipse[:,1], color = 'k', linewidth = 1)
        
        # plot ampl. bounds
        plt.plot([0,errorEllipse[ampMinNormIx,0]], [0,errorEllipse[ampMinNormIx,1]],
                 color = 'r', linestyle = '--')
        plt.plot([0,errorEllipse[ampMaxNormIx,0]], [0,errorEllipse[ampMaxNormIx,1]],
                 color = 'r', label = 'ampl. bounds', linestyle = '--')
        font['color']='r'
        plt.text(errorEllipse[ampMinNormIx,0], errorEllipse[ampMinNormIx,1],
                 round(ampMinNorm, 2), fontdict = font)
        plt.text(errorEllipse[ampMaxNormIx,0], errorEllipse[ampMaxNormIx,1],
                 round(ampMaxNorm, 2), fontdict = font)
        
        # plot phase bounds
        plt.plot([0,errorEllipse[phaseMinIx,0]], [0,errorEllipse[phaseMinIx,1]], 
                 color = 'b', linewidth = 1)
        plt.plot([0,errorEllipse[phaseMaxIx,0]], [0,errorEllipse[phaseMaxIx,1]], 
                 color = 'b', linewidth = 1, label = 'phase bounds')
        font['color']='b'
        plt.text(errorEllipse[phaseMinIx,0], errorEllipse[phaseMinIx,1],
                 round(phaseEllipseExtremes[0], 2), fontdict = font)
        plt.text(errorEllipse[phaseMaxIx,0], errorEllipse[phaseMaxIx,1],
                 round(phaseEllipseExtremes[1], 2), fontdict = font)
        
        # plot mean vector
        plt.plot([0, meanXy[0]],[0,meanXy[1]], color = 'k', linewidth = 1, label = 'mean ampl.')
        font['color'] = 'k'
        plt.text(meanXy[0],meanXy[1],round(norm_meanXy,2),fontdict=font)
        
        # plot major/minor axis
        plt.plot([meanXy[0], a * larger_eigenvec[0] + meanXy[0]], 
                 [meanXy[1], a * larger_eigenvec[1] + meanXy[1]],
                 color='m',linewidth=1)
        plt.plot([meanXy[0], -a * larger_eigenvec[0] + meanXy[0]], 
                 [meanXy[1], -a * larger_eigenvec[1] + meanXy[1]],
                 color='m',linewidth=1)
        plt.plot([meanXy[0], b * smaller_eigenvec[0] + meanXy[0]], 
                 [meanXy[1], b * smaller_eigenvec[1] + meanXy[1]],
                 color='g',linewidth=1)
        plt.plot([meanXy[0], -b * smaller_eigenvec[0] + meanXy[0]], 
                 [meanXy[1], -b * smaller_eigenvec[1] + meanXy[1]],
                 color='g',linewidth=1)
        
        plt.axhline(color = 'k', linewidth = 1)
        plt.axvline(color = 'k', linewidth = 1)
        plt.legend(loc = 3, frameon = False)
        plt.axis('equal')
        plt.show()
    return ampDiff, phaseDiff, zSNR, errorEllipse

def combineHarmonics(subjects, fmriFolder, fsdir=os.environ["SUBJECTS_DIR"], pre_tr=0, roi='wang+benson',session='01',tasks=None, offset=None,std141=False):
    """ Combine data across subjects - across RoIs & Harmonics
    So there might be: 180 RoIs x 5 harmonics x N subjects.
    This can be further split out by tasks. If no task is 
    provided then 
    This uses the RoiSurfData function carry out FFT.
    Parameters
    ------------
    subjects : list
        List of all subjects to submit
    fmriFolder : string
        Folder location, required to identify files
        to be used
    fsdir : string/os file location, default to SUBJECTS_DIR
        Freesurfer folder directory
    pre_tr : int, default 0
        input for RoiSurfData - please see for more info
    roi : str, default 'wang+benson'
        other options as per RoiSurfData function
    session : str, default '01'
        The fmri session
    tasks : list, default None
        Processing to be split by task and run 
        separately
    offset : dictionary, default None
        offset to be applied to relevant tasks:
        Positive values means the first data frame was 
        shifted forwards relative to stimulus.
        Negative values means the first data frame was 
        shifted backwards relative to stimulus.
        example: {'cont':2,'disp':8,'mot':2}
    std141 : boolean, default False
        Should files for combined harmonics
        be std141 or native
    Returns
    ------------
    task_dic : dictionary
        dictionary contains data broken down by task
        into:
            outdata_arr : numpy array
                a numpy array with dimensions of
                (roi_number, harmonic_number, subjects_number)
    outnames : list
        a list of strings, containing RoIs
    """
    task_dic={}
    if tasks==None:
        print('No task provided. Data to be run together.')
        for sub_int, sub in enumerate(subjects):
            # produce list of files 
            surf_files = subjectFmriData(sub, fmriFolder,std141=std141,session=session)
            # run RoiSurfData
            outdata, outnames = RoiSurfData(surf_files,roi=roi,fsdir=fsdir,pre_tr=pre_tr,offset=offset)
            # Define the empty array we want to fill or concatenate together
            if sub_int == 0:
                outdata_arr =  np.array([np.array(roiobj.fft.sig_complex) for roiobj in outdata])
            else:
                outdata_arr = np.concatenate((outdata_arr, np.array([np.array(roiobj.fft.sig_complex) for roiobj in outdata])),axis=2)
        task_dic['task'], outnames = outdata_arr, outnames
        return task_dic
    else:
        for task in tasks:
            for sub_int, sub in enumerate(subjects):
                # produce list of files 
                surf_files = [f for f in subjectFmriData(sub, fmriFolder, std141=std141,session=session) if task in f]
                # run RoiSurfData
                outdata, outnames = RoiSurfData(surf_files,roi=roi,fsdir=fsdir,pre_tr=pre_tr,offset=offset[task])
                # Define the empty array we want to fill or concatenate together
                if sub_int == 0:
                    outdata_arr =  np.array([np.array(roiobj.fft.sig_complex) for roiobj in outdata])
                else:
                    outdata_arr = np.concatenate((outdata_arr, np.array([np.array(roiobj.fft.sig_complex) for roiobj in outdata])),axis=2)
            task_dic[task], outnames = outdata_arr, outnames
        return task_dic, outnames

def applyFitErrorEllipse(combined_harmonics, outnames, ampPhaseZsnr_output='all',ellipseType='SEM', makePlot=False, returnRad=False):
    """ Apply fitErrorEllipse to output data from RoiSurfData.
    Parameters
    ------------
    combined_harmonics : numpy array
        create this input using combineHarmonics()
        array dimensions: (roi_number, harmonic_number, subject_number)
    outnames : list of strings
        list of all RoIs
    ampPhaseZsnr_output : string or list or strs, default 'all'
        Can specify what output you would like.
        These are phase difference, amp difference and zSNR
        Options: 'all', 'phase', 'amp', 'zSNR',['phase','amp'] etc
    ellipseType : string, default 'SEM'
        ellipse type SEM or in format: 'x%CI'
    makePlot : boolean, default False
        If True, will produce a plot for each RoI
    returnRad : boolean, default False
        Specify to return values in radians or degrees
    
    Returns
    ------------
    ampPhaseZSNR_df : pd.DataFrame,
        contains RoIs as index, amp difference lower/upper, 
        phase difference lower/upper, zSNR
    errorEllipse_dic : dictionary,
        contains a dictionary of numpy arrays of the 
        error ellipses broken down by RoI.
    """ 

    # dictionary for output of error Ellipse
    errorEllipse_dic={}
    # number of rois, harmonics, subjects
    roi_n, harmonics_n, subjects_n = combined_harmonics.shape
    # to be used to create the final columns in the dataframe
    harmonic_name_list = ['RoIs']

    # Loop through harmonics & rois
    for harmonic in range(harmonics_n):
        print('Working on harmonic: {0}'.format(harmonic + 1))
        harmonic_measures=[measure+str(harmonic+1) for measure in ['AmpDiffLower','AmpDiffHigher','PhaseDiffLower','PhaseDiffHigher','zSNR']]
        harmonic_name_list+=harmonic_measures
        for roi in range(roi_n):
            errorEllipseName = '{0}_harmonic_{1}'.format(outnames[roi],harmonic)
            xyData = combined_harmonics[roi,harmonic,:]
            xyData = np.reshape(xyData,(len(xyData),1))
            ampDiff, phaseDiff, zSNR, errorEllipse = fitErrorEllipse(xyData,ellipseType,makePlot,returnRad)
            errorEllipse_dic[errorEllipseName] = errorEllipse
            if roi==0 and harmonic ==0:
                t=np.array([[outnames[roi],ampDiff[0],ampDiff[1],phaseDiff[0],phaseDiff[1],zSNR]])
            elif harmonic == 0:
                t=np.concatenate((t,np.array([[outnames[roi],ampDiff[0],ampDiff[1],phaseDiff[0],phaseDiff[1],zSNR]])),axis=0)
            elif roi==0:
                t=np.array([[ampDiff[0],ampDiff[1],phaseDiff[0],phaseDiff[1],zSNR]])
            else:
                t=np.concatenate((t,np.array([[ampDiff[0],ampDiff[1],phaseDiff[0],phaseDiff[1],zSNR]])),axis=0)
        if harmonic == 0:
            overall=t
        else:
            overall=np.concatenate((overall,t),axis=1)
    
    # construct dataframe for output
    ampPhaseZSNR_df=pd.DataFrame(data=overall,columns=harmonic_name_list,index=outnames)
    ampPhaseZsnr_output = str(ampPhaseZsnr_output).lower()
    phase = ampPhaseZSNR_df[[col for col in ampPhaseZSNR_df.columns if 'Phase' in col]]
    amp = ampPhaseZSNR_df[[col for col in ampPhaseZSNR_df.columns if 'Amp' in col]]
    zSNR = ampPhaseZSNR_df[[col for col in ampPhaseZSNR_df.columns if 'z' in col]]
    
    if ampPhaseZsnr_output == 'all':
        concat_columns = [phase, amp, zSNR]
    else:
        concat_columns = []
        if 'phase' in ampPhaseZsnr_output:
            concat_columns.append(phase)
        if 'amp' in ampPhaseZsnr_output:
            concat_columns.append(phase)
        if 'zsnr' in ampPhaseZsnr_output:
            concat_columns.append(zSNR)
    ampPhaseZSNR_df = pd.concat((concat_columns),axis=1)
    print('DataFrame constructed')
    return ampPhaseZSNR_df, errorEllipse_dic
def graphRois(combined_harmonics,outnames,subjects=None,plot_by_subject=False,harmonic=1,rois=['V1','V2','V3'],hemisphere='BL',figsize=6):
    """This function will graph RoI data. 
    Parameters
    ------------
    combined_harmonics : numpy array
        Array of complex data. Must be in
        below shape: RoI x Hemisphere x Subjects.
        If data has been averaged across subjects
        then will be in shape: RoI x Hemisphere
    outnames : list
        This will be a list of RoI names produced
        by RoiSurfData
    subjects : list, default None
        If plotting across subjects this is
        required to iterate over and to add legend
        to graph. If data has been averaged across
        subjects then not required.
    plot_by : boolean, default False
        How should data be plotted? Either will
        be plotted by subject or will
        be plotted by above or below inverse neg regression
    harmonic : integer, default 1
        Select harmonic required. Options start at 1,
        as in 1st harmonic, 2nd etc.
    rois : list, default ['V1','V2','V3']
        List of RoIs interested in
    hemisphere : string, default 'BL'
        Options are 'L', 'R' or 'BL'
    figsize : integer, default 6
        Sets the figsize of graphs produced
    Returns
    ------------
    output : numpy array
        Returns the numpy array of values
        with the addition of sign (+1/-1)
        for above or below the perpendicular
        of regression line through the origin
    """
    output = []
    
    # Initial user input checking
    harmonic-=1
    if subjects == None:
        assert combined_harmonics.ndim == 2, 'Data shape implies subjects should be supplied.'
    elif subjects != None:
        assert combined_harmonics.ndim == 3, 'Data shape implies subjects should not be supplied.'
    hemisphere = hemisphere.upper()
    assert hemisphere in ['L','R','BL'], 'Hemisphere incorrect. See docstring for options'
    
    # select only specific hemisphere to look at
    ROIs_hemisphere={x:i for i,x in enumerate(outnames) if hemisphere in x}
    # Create linear regression model for later use
    linear_reg=LinearRegression()
    ROI_coefs = {}
    # Loop through RoIs and produce scatter plot
    for roi in rois:
        ROI_coef = []
        v = {k:ROIs_hemisphere[k] for k in ROIs_hemisphere.keys() if roi in k}
        plt.figure(figsize=(figsize,figsize))
        # If using data averaged across subjects
        if subjects == None or len(subjects) ==1 :
            # select the data based on desired harmonic and the RoI
            harmonic_user_data = combined_harmonics[min(v.values()):max(v.values())+1,harmonic]
            m, = harmonic_user_data.shape
            user_data = harmonic_user_data
            user_data = user_data.reshape(m,1)
            # split complex data into real and imaginary
            user_data = realImagSplit(user_data)
            if plot_by_subject==True:
                plt.scatter(user_data[:,0],user_data[:,1])
            all_user_data = user_data
            subjects = ['Mean Subject Data']
        else:
            first_iteration = True
            # select the data across subjects based on desired harmonic and the RoI
            harmonic_user_data = combined_harmonics[min(v.values()):max(v.values())+1,harmonic,:]
            m, u = harmonic_user_data.shape
            for s, subject in enumerate(subjects):
                user_data = harmonic_user_data[:,s]
                user_data = user_data.reshape(m,1)
                user_data = realImagSplit(user_data)
                if first_iteration == True:
                    all_user_data = user_data
                    first_iteration = False
                else:
                    all_user_data = np.concatenate((all_user_data,user_data),axis=0)
                if plot_by_subject==True:
                    # plot individual subjects
                    plt.scatter(user_data[:,0],user_data[:,1])
        # Create graph
        plt.title(roi)
        plt.axis('equal')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        
        # Add line for axes
        plt.axhline(color = 'k', linewidth = 1)
        plt.axvline(color = 'k', linewidth = 1)
        # Create legend
        if plot_by_subject == True:
            additional_legend = subjects
            plt.legend(['linear fit','perpendicular fit']+additional_legend,loc=4)
        else:
            additional_legend = ['Positive','Negative']
        
        m,n = all_user_data.shape
        # fit a linear regression
        X=all_user_data[:,0].reshape(m,1)
        y=all_user_data[:,1].reshape(m,1)
        linear_reg.fit(X,y)
        ROI_coef.append(linear_reg.coef_.flatten()[0])
        # set the intercept to the origin
        linear_reg.intercept_=0
        m=max(np.absolute(np.min(X)),np.absolute(np.max(X)))
        # plot regression line & perpendicular of regression
        X=np.linspace(-m,+m)
        X=X.reshape(X.shape[0],1)
        plt.plot(X,linear_reg.predict(X),'k',label='linear fit')
        linear_reg.coef_ = 1/(-linear_reg.coef_)
        ROI_coef.append(linear_reg.coef_.flatten()[0])
        ROI_coefs[roi] = ROI_coef
        plt.plot(X,linear_reg.predict(X),'k',label='perpendicular fit')
        # give the figures a symbol +1/-1
        roi_output = np.concatenate((all_user_data,np.zeros((all_user_data.shape[0],1))),axis=1)
        roi_output[roi_output[:,1]<roi_output[:,0]*linear_reg.coef_.flatten(),2]=-1
        roi_output[roi_output[:,1]>roi_output[:,0]*linear_reg.coef_.flatten(),2]=+1
        output.append(roi_output)
        if plot_by_subject == False:
            plt.scatter(roi_output[roi_output[:,2]==1,0],roi_output[roi_output[:,2]==1,1],label='Positive')
            plt.scatter(roi_output[roi_output[:,2]==-1,0],roi_output[roi_output[:,2]==-1,1],label='Negative')
            plt.legend(loc=4)
        plt.show()
    # returns RoIs - coefficients & negative inverse of coeff
    return output