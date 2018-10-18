import os, subprocess, sys, glob, shutil, tempfile, re, time, textwrap
from nilearn import surface as nl_surf
from nilearn import signal as nl_signal
from os.path import expanduser
import numpy as np
import pandas as pd
import scipy as scp
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


## HELPER FUNCTIONS
def print_wrap(msg, indent=0):
    if int(indent) > 0:
        wrapper = textwrap.TextWrapper(initial_indent=" " * 4 * indent, subsequent_indent=" " * 4 * (indent + 1),
                                       width=100)
        print(wrapper.fill(msg))
    else:
        print(msg)


def fs_dir_check(fs_dir, subject):
    # check if subjects' SUMA directory exists
    if os.path.isdir("{0}/{1}/SUMA".format(fs_dir, subject)):
        # no suffix needed
        suffix = ""
    else:
        # suffix needed f
        suffix = "_fs4"
        if not os.path.isdir("{0}/{1}{2}".format(fs_dir, subject, suffix)):
            sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                     .format(fs_dir, subject, suffix))
    return suffix


def copy_suma_files(suma_dir, tmp_dir, subject, suffix="", spec_prefix=""):
    for files in glob.glob(suma_dir + "/*h.smoothwm.gii"):
        shutil.copy(files, tmp_dir)
    for files in glob.glob(suma_dir + "/*h.pial.gii"):
        shutil.copy(files, tmp_dir)
    for files in glob.glob(suma_dir + "/*h.smoothwm.asc"):
        shutil.copy(files, tmp_dir)
    for files in glob.glob(suma_dir + "/*h.pial.asc"):
        shutil.copy(files, tmp_dir)
    for files in glob.glob("{0}/{1}{2}{3}*.spec".format(suma_dir, spec_prefix, subject, suffix)):
        shutil.copy(files, tmp_dir)
    # for some reason, 3dVol2Surf requires these files, so copy them as well
    for files in glob.glob(suma_dir + "/*aparc.*.annot.niml.dset"):
        shutil.copy(files, tmp_dir)


def get_name_suffix(cur_file, surface=False):
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
        print(main_cmd + "\n")
    subprocess.call("{0}".format(main_cmd), shell=True)


def fft_offset(complex_in, offset_rad):
    amp = np.absolute(complex_in)
    phase = np.angle(complex_in)
    # subtract offset from phase
    phase = phase - offset_rad
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    # convert back to complex
    complex_out = np.mean(np.absolute(amp) * np.exp(1j * phase), keepdims=True, axis=0)
    amp_out = np.mean(np.absolute(complex_out), keepdims=True, axis=0)
    phase_out = np.mean(np.angle(complex_out), keepdims=True, axis=0)
    return complex_out, amp_out, phase_out


def real_imag_split(sig_complex):
    # takes complex data and splits into real and imaginary
    if len(sig_complex.shape) > 1:
        sig_complex = np.dstack((np.real(sig_complex), np.imag(sig_complex)))
        # make real/imag second axis
        sig_complex = np.transpose(sig_complex, (0, 2, 1))
    else:
        sig_complex = sig_complex.reshape(sig_complex.shape[0], -1)
        sig_complex = np.concatenate((np.real(sig_complex), np.imag(sig_complex)), axis=1)
    return sig_complex


def flatten_1d(arr):
    # function to flatten data into 1 dimension
    arr = arr.flatten()
    arr = arr.reshape(len(arr), 1)
    return arr


def eig_fouriercoefs(xydata):
    """Performs eigenvalue decomposition
    on 2D data. Function assumes 2D
    data are Fourier coefficients
    with real coeff in 1st col and 
    imaginary coeff in 2nd col.
    """
    # ensure data is in Array
    xydata = np.array(xydata)
    m, n = xydata.shape
    if n != 2:
        # ensure data is split into real and imaginary
        xydata = real_imag_split(xydata)
        m, n = xydata.shape
        if n != 2:
            print_wrap('Data in incorrect shape - please correct')
            return None
    realData = xydata[:, 0]
    imagData = xydata[:, 1]
    # mean and covariance
    mean_xy = np.mean(xydata, axis=0)
    sampCovMat = np.cov(np.array([realData, imagData]))
    # calc eigenvalues, eigenvectors
    eigenval, eigenvec = np.linalg.eigh(sampCovMat)
    # sort the eigenvector by the eigenvalues
    orderedVals = np.sort(eigenval)
    eigAscendIdx = np.argsort(eigenval)
    smaller_eigenvec = eigenvec[:, eigAscendIdx[0]]
    larger_eigenvec = eigenvec[:, eigAscendIdx[1]]
    smaller_eigenval = orderedVals[0]
    larger_eigenval = orderedVals[1]

    phi = np.arctan2(larger_eigenvec[1], larger_eigenvec[0])
    # this angle is between -pi & pi, shift to 0 and 2pi
    if phi < 0:
        phi = phi + 2 * np.pi
    return (mean_xy, sampCovMat, smaller_eigenvec,
            smaller_eigenval, larger_eigenvec,
            larger_eigenval, phi)


def create_file_dictionary(experiment_dir):
    # creates file dictionary necessary for Vol2Surf
    subjects = [folder for folder in os.listdir(experiment_dir) if 'sub' in folder and '.html' not in folder]
    subject_session_dic = {
    subject: [session for session in os.listdir("{0}/{1}".format(experiment_dir, subject)) if 'ses' in session] for
    subject in subjects}
    subject_session_directory = []
    for subject in subjects:
        subject_session_directory += ["{0}/{1}/{2}/func".format(experiment_dir, subject, session) for session in
                                      subject_session_dic[subject]]
    files_dic = {directory: [files for files in os.listdir(directory) if 'preproc.nii.gz' in files] for directory in
                 subject_session_directory}
    return files_dic


## CLASSES

# used by roi_surf_data
class roiobject:
    def __init__(self, curdata=np.zeros((120, 1)), curobject=None, is_time_series=True, roiname="unknown", tr=99,
                 stimfreq=99, nharm=99, num_vox=0, offset=0):
        if not curobject:
            self.data = np.zeros((120, 1))
            self.roi_name = roiname
            self.tr = tr
            self.stim_freq = stimfreq
            self.num_harmonics = nharm
            self.num_vox = num_vox
            self.is_time_series = is_time_series
            self.offset = offset
        else:
            # if curobject provided, inherit all values
            self.data = curobject.data
            self.roi_name = curobject.roi_name
            self.tr = curobject.tr
            self.stim_freq = curobject.stim_freq
            self.num_harmonics = curobject.num_harmonics
            self.num_vox = curobject.num_vox
            self.is_time_series = curobject.is_time_series
            self.offset = curobject.offset
        if curdata.any():
            if self.data.any():
                self.data = np.dstack((self.data, curdata))
            else:
                self.data = curdata
            self.mean = self.average()
            if self.is_time_series == True:
                self.fft = self.fourieranalysis()

    def average(self):
        if len(self.data.shape) < 3:
            return self.data
        else:
            return np.mean(self.data, axis=2, keepdims=False)

    def fourieranalysis(self):
        return fft_analysis(self.average(),
                            tr=self.tr,
                            stimfreq=self.stim_freq,
                            nharm=self.num_harmonics,
                            offset=self.offset)


# used by mrifft
class fftobject:
    def __init__(self, **kwargs):
        allowed_keys = ["spectrum", "frequencies", "mean_cycle", "sig_z", "sig_snr",
                        "sig_amp", "sig_phase", "sig_complex", "noise_complex", "noise_amp", "noise_phase"]
        for key in allowed_keys:
            setattr(self, key, [])
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)


# used by roi_group_analysis
class groupobject:
    def __init__(self, amp=np.zeros((3, 1)), phase=np.zeros((3, 1)), zSNR=0, hotT2=np.zeros((2, 1)),
                 cycle=np.zeros((2, 12)), harmonics="1", roi_name="unknown"):
        self.amp = amp
        self.phase = phase
        self.zSNR = zSNR
        self.hotT2 = hotT2
        self.cycle_average = cycle[0,]
        dim1, dim2 = np.shape(cycle)
        if dim1 == 2:
            self.cycle_err = cycle[1,]
        else:
            self.cycle_err = []
        self.roi_name = roi_name
        self.harmonics = harmonics


## MAIN FUNCTIONS

def write(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, -1)


def read(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out

def get_bids_data(target_folder, file_type='suma_native', session='01', smooth=0):
    file_list = []
    if session == 'all':
        session_folders = ["{0}/{1}".format(target_folder, s) for s in os.listdir(target_folder) if 'ses' in s]
    else:
        session_folders = ["{0}/ses-{1}".format(target_folder, str(session).zfill(2))]
    for cur_ses in session_folders:
        cur_dir = '{0}/func/'.format(cur_ses)
        if file_type in ['suma_std141', 'sumastd141']:
            file_list += [cur_dir + x for x in os.path.os.listdir(cur_dir) if x[-3:] == 'gii' and 'surf.std141' in x]
        elif file_type in ['suma_native', 'sumanative']:
            file_list += [cur_dir + x for x in os.path.os.listdir(cur_dir) if x[-3:] == 'gii' and 'surf.native' in x]
        elif file_type in ['fs_native', 'fsnative']:
            file_list += [cur_dir + x for x in os.path.os.listdir(cur_dir) if x[-3:] == 'gii' and 'fsnative' in x]
        else:
            print_wrap('unknown file_type {0} provided'.format(file_type), indent=3)
            print_wrap('... using fsnative', indent=3)
            file_list += [cur_dir + x for x in os.path.os.listdir(cur_dir) if x[-3:] == 'gii' and 'fsnative' in x]
    if smooth > 0:
        file_list = [x for x in file_list if "{0}fwhm".format(smooth) in x]
    else:
        file_list = [x for x in file_list if "fwhm" not in x]
    return file_list


def run_suma(subject, hemi='both', open_vol=False, surf_vol='standard', std141=False, fs_dir=None):
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

    suffix = fs_dir_check(fs_dir, subject)

    suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir, subject, suffix)
    if std141:
        spec_file = "{0}/std.141.{1}{2}_{3}.spec".format(suma_dir, subject, suffix, hemi)
    else:
        spec_file = "{0}/{1}{2}_{3}.spec".format(suma_dir, subject, suffix, hemi)

    if surf_vol is "standard":
        surf_vol = "{0}/{1}{2}/SUMA/{1}{2}_SurfVol+orig".format(fs_dir, subject, suffix)
    else:
        # if surfvol was assigned, assume user wants to open volume
        open_vol = True

    if open_vol:
        vol_dir = '/'.join(surf_vol.split('/')[0:-1])
        vol_file = surf_vol.split('/')[-1]
        if vol_dir:  # if volume directory is not empty
            os.chdir(vol_dir)
        subprocess.call("afni -niml & SUMA -spec {0} -sv {1} &".format(spec_file, vol_file), shell=True)
    else:
        subprocess.call("SUMA -spec {0} &".format(spec_file), shell=True)


def neuro_to_radio(in_files):
    for scan in in_files:
        name, suffix = get_name_suffix(scan)
        old_orient = subprocess.check_output("fslorient -getorient {0}".format(scan), shell=True,
                                             universal_newlines=True)
        print_wrap("old orientation: {0}".format(old_orient))
        temp_scan = name + "_temp" + suffix
        shutil.copyfile(scan, temp_scan)
        try:
            shell_cmd("fslswapdim {0} z x y {0}".format(scan))
            shell_cmd("fslswapdim {0} -x -y -z {0}".format(scan))
            shell_cmd("fslorient -swaporient {0}".format(scan))
        except:
            # replace with original
            shutil.copyfile(temp_scan, scan)
            print_wrap("orientation could not be changed for file {0}".format(scan))
        os.remove(temp_scan)
        new_orient = subprocess.check_output("fslorient -getorient {0}".format(scan), shell=True,
                                             universal_newlines=True)
        print_wrap("new orientation: {0}".format(new_orient))


def pre_slice(in_files, ref_file='last', tr_dur=0, pre_tr=0, total_tr=0, slice_time_file=None, pad_ap=0, pad_is=0,
              diff_mat=False, keep_temp=False):
    """
    Function for first stage of preprocessing
    Slice-time correction and deobliqueing, in that order.
    Also supports data with different number of slices,
    and padding of the matrices, via flags 
    --diffmat, --pad_ap and --pad_is. 

    Author: pjkohler, Stanford University, 2016"""

    # assign remaining defaults
    if ref_file in "last":
        ref_file = in_files[-1]  # use last as reference
    if tr_dur is 0:
        # TR not given, so compute
        tr_dur = subprocess.check_output("3dinfo -tr -short {0}".format(ref_file), shell=True)
        tr_dur = tr_dur.rstrip("\n")
    if total_tr is 0:
        # include all TRs, so get max subbrick value
        total_tr = subprocess.check_output("3dinfo -nvi -short {0}".format(ref_file), shell=True)
        total_tr = total_tr.rstrip("\n")
    else:
        # subject has given total number of TRs to include, add preTR to that
        total_tr = eval("pre_tr + total_tr")

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))
    os.chdir(tmp_dir)

    name_list = []

    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)

        # crop and move files
        cur_total_tr = subprocess.check_output("3dinfo -nvi -short {1}/{0}{2}"
                                               .format(file_name, cur_dir, suffix), shell=True)
        cur_total_tr = cur_total_tr.rstrip("\n")

        subprocess.call("3dTcat -prefix {0}+orig {1}/{0}{2}''[{3}..{4}]''"
                        .format(file_name, cur_dir, suffix, pre_tr, min(cur_total_tr, total_tr)), shell=True)

        # slice timing correction
        if slice_time_file is None:
            subprocess.call(
                "3dTshift -quintic -prefix {0}.ts+orig -TR {1}s -tzero 0 -tpattern alt+z {0}+orig".format(file_name,
                                                                                                          tr_dur),
                shell=True)
        else:
            subprocess.call("3dTshift -quintic -prefix {0}/{1}.ts+orig -TR {2}s -tzero 0 -tpattern @{3} {0}/{1}+orig"
                            .format(tmp_dir, file_name, tr_dur, slice_time_file), shell=True)

        # deoblique
        subprocess.call("3dWarp -deoblique -prefix {0}.ts.do+orig {0}.ts+orig".format(file_name), shell=True)

        # pad 
        if pad_ap is not 0 or pad_is is not 0:
            subprocess.call(
                "3dZeropad -A {1} -P {1} -I {2} -S {2} -prefix {0}.ts.do.pad+orig {0}.ts.do+orig".format(file_name,
                                                                                                         pad_ap,
                                                                                                         pad_is),
                shell=True)
            subprocess.call("rm {0}.ts.do+orig*".format(file_name), shell=True)
            subprocess.call("3dRename {0}.ts.do.pad+orig {0}.ts.do+orig".format(file_name), shell=True)

        if diff_mat:
            # add file_names to list, move later
            name_list.append(file_name)
            if cur_file == ref_file:
                ref_name = file_name
        else:
            subprocess.call("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.nii.gz {0}.ts.do+orig".format(file_name, cur_dir),
                            shell=True)

    if diff_mat:
        # take care of different matrices, and move
        for file_name in name_list:
            subprocess.call("@Align_Centers -base {1}.ts.do+orig -dset {0}.ts.do+orig".format(file_name, ref_name),
                            shell=True)
            subprocess.call(
                "3dresample -master {1}.ts.do+orig -prefix {0}.ts.do.rs+orig -inset {0}.ts.do_shft+orig".format(
                    file_name, ref_name), shell=True)
            subprocess.call(
                "3dAFNItoNIFTI -prefix {1}/{0}.ts.do.rs.nii.gz {0}.ts.do.rs+orig".format(file_name, cur_dir),
                shell=True)

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def pre_volreg(in_files, ref_file='last', slow=False, keep_temp=False):
    """
    Function for second stage of preprocessing: Volume registation.
    Typically run following mriPre.py
    Use option --slow for difficult cases
    Author: pjkohler, Stanford University, 2016
    """
    # assign remaining defaults
    if ref_file in "last":
        ref_file = in_files[-1]  # use last as reference

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))
    os.chdir(tmp_dir)

    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)

        # move files
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(file_name, cur_dir, suffix), shell=True)

        # do volume registration
        if slow:
            subprocess.call(
                "3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -heptic -twopass -maxite 50 {0}+orig"
                .format(file_name, ref_file, cur_dir), shell=True)
        else:
            subprocess.call(
                "3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -Fourier {0}+orig"
                .format(file_name, ref_file, cur_dir), shell=True)

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def pre_scale(in_files, no_dt=False, keep_temp=False):
    """
    Function for third stage of preprocessing: Scaling and Detrending.
    Typically run following mriPre.py and mriVolreg.py 
    Detrending currently requires motion registration parameters
    as .1D files: motparam.xxx.1D 

    Author: pjkohler, Stanford University, 2016
    """

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))
    os.chdir(tmp_dir)

    for cur_file in in_files:
        file_name, suffix = get_name_suffix(cur_file)

        # move files
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(file_name, cur_dir, suffix), shell=True)
        # compute mean
        subprocess.call("3dTstat -prefix mean_{0}+orig {0}+orig".format(file_name), shell=True)
        # scale
        if no_dt:
            # save scaled data in data folder directly
            subprocess.call(
                "3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {1}/{0}.sc.nii.gz"
                .format(file_name, cur_dir), shell=True)
        else:
            # scale, then detrend and store in data folder
            subprocess.call(
                "3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {0}.sc+orig"
                .format(file_name), shell=True)
            subprocess.call("3dDetrend -prefix {1}/{0}.sc.dt.nii.gz -polort 2 -vector {1}/motparam.{0}.1D {0}.sc+orig"
                            .format(file_name, cur_dir), shell=True)

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def vol_to_surf(experiment_dir, fsdir=os.environ["SUBJECTS_DIR"], subjects=None, sessions=None, map_func='ave',
                wm_mod=0.0, gm_mod=0.0, prefix=None, index='voxels', steps=10, mask=None, surf_vol='standard',
                blur_size = 0, delete_unsmoothed = False, std141=False, keep_temp=False):
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
    experiment_dir : string
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
    blur_size : Int, default 0
        if blur_size > 0, smoothing will be applied
    delete_unsmoothed : Boolean, default False
        If True, unsmoothed data will be removed after smoothing
    keep_temp : Boolean, default False
        Should temporary folder that is set up be kept after function 
        runs?
    Returns 
    ------------
    file_list : list of strings
        This is a list of all files created by the function
    """
    # Create a dictionary of files - keys are the directory for each session
    file_dictionary = create_file_dictionary(experiment_dir)
    # Remove unwanted subjects and sessions
    if subjects != None:
        file_dictionary = {directory: file_dictionary[directory] for directory in file_dictionary.keys()
                           if directory[len(experiment_dir) + 1:].split('/')[0] in subjects}
    if sessions != None:
        file_dictionary = {directory: file_dictionary[directory] for directory in file_dictionary.keys()
                           if directory[len(experiment_dir) + 1:].split('/')[1] in sessions}
    # list of files created by this function
    file_list = []
    # Dict to convert between old and new hemisphere notation
    hemi_dic = {'lh': 'L', 'rh': 'R'}

    # Iterate over subjects
    for directory in file_dictionary.keys():
        # pull out subject title - i.e. 'sub-0001'
        subject = directory[len(experiment_dir) + 1:].split('/')[0]
        if std141:
            print_wrap("Running subject {0}, std141 template:".format(subject))
        else:
            print_wrap("Running subject {0}, native:".format(subject))
        cur_dir = directory
        in_files = file_dictionary[directory]
        # Define the names to be used for file output
        criteria_list = ['sub', 'ses', 'task', 'run', 'space']
        input_format = 'bids'
        output_name_dic = {}
        for cur_file in in_files:
            file_name, file_suffix = get_name_suffix(cur_file)
            # Checking for bids formatting
            if not sum([crit in file_name for crit in criteria_list]) == len(criteria_list):
                input_format = 'non-bids'
            if input_format == 'bids':
                old = re.findall('space-\w+_', file_name)[0]
                if std141 == False:
                    new = 'space-surf.native_'
                else:
                    new = 'space-surf.std141_'
                output_name_dic[file_name] = file_name.replace(old, new)
            else:
                if std141 == False:
                    new = 'space-surf.native'
                elif std141 == True:
                    new = 'space-surf.std141'
                output_name_dic[file_name] = '{0}_{1}'.format(file_name, new)

        # make temporary, local folder
        tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))

        # check if subjects' SUMA directory exists
        suffix = fs_dir_check(fsdir, subject)
        suma_dir = "{0}/{1}{2}/SUMA".format(fsdir, subject, suffix)

        if wm_mod is not 0.0 or gm_mod is not 0.0:
            # for gm, positive values makes the distance longer, for wm negative values
            steps = round(steps + steps * gm_mod - steps * wm_mod)

        print_wrap("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod, gm_mod, steps), indent=1)
        if surf_vol is "standard":
            vol_dir = "{0}/{1}{2}/SUMA".format(fsdir, subject, suffix)
            vol_file = "{0}{1}_SurfVol.nii".format(subject, suffix)
        else:
            vol_dir = '/'.join(surf_vol.split('/')[0:-1])
            vol_file = surf_vol.split('/')[-1]
            if not vol_dir:  # if volume directory is empty
                vol_dir = cur_dir

        # make temporary copy of volume file     
        subprocess.call("3dcopy {0}/{1} {2}/{1}".format(vol_dir, vol_file, tmp_dir), shell=True)

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
        copy_suma_files(suma_dir, tmp_dir, subject, spec_prefix=specprefix)

        os.chdir(tmp_dir)
        for cur_file in in_files:
            file_name, file_suffix = get_name_suffix(cur_file)
            # unzip the .nii.gz files into .nii files
            shell_cmd("gunzip -c {0}/{1}.nii.gz > {2}/{1}.nii".format(cur_dir, file_name, tmp_dir), do_print=False)
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
                    subprocess.call("3dcopy {1}/{0}{2} mask+orig".format(mask_name, cur_dir, mask_suffix), shell=True)
                    maskcode = "-cmask '-a mask+orig[0] -expr notzero(a)' "
            for hemi in ["lh", "rh"]:
                output_file_name = "{0}.{1}.func".format(output_name_dic[file_name], hemi_dic[hemi])
                # Converts volume to surface space - output in .niml.dset
                shell_cmd("3dVol2Surf -spec {0}{1}{2}_{3}.spec \
                        -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {5}.nii -map_func {6} \
                        -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                        -outcols_NSD_format -oob_value -0 {11}-out_niml {12}/{13}.niml.dset"
                          .format(specprefix, subject, suffix, hemi, vol_file, file_name, map_func, index, wm_mod,
                                  gm_mod, steps, maskcode, tmp_dir, output_file_name), do_print=False)
                # Removes output gii file if it exists
                out_path = "{0}/{1}.gii".format(cur_dir, output_file_name)
                if os.path.isfile(out_path):
                    os.remove(out_path)
                # Converts the .niml.dset into a .gii file in the functional directory
                shell_cmd("ConvertDset -o_gii_b64 -input {1}/{0}.niml.dset -prefix {2}/{0}.gii"
                          .format(output_file_name, tmp_dir, cur_dir), do_print=False)
                file_list.append('{1}/{0}'.format(output_file_name, cur_dir))

                if blur_size > 0:
                    # Removes output gii file if it exists
                    out_path = "{0}/{1}_{2}fwhm.gii".format(cur_dir, output_file_name, blur_size)
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                    # run smoothing
                    shell_cmd("SurfSmooth -spec {0}{1}{2}_{3}.spec \
                              -surf_A smoothwm -met HEAT_07 -target_fwhm {4} -input {5}/{6}.gii \
                                -cmask '-a {5}/{6}.gii[0] -expr bool(a)' -output {5}/{6}_{4}fwhm.gii"
                                    .format(specprefix, subject, suffix, hemi, blur_size, cur_dir, output_file_name), do_print=False)
                    if delete_unsmoothed:
                        os.remove("{1}/{0}.gii".format(output_file_name, cur_dir))
                else:
                    assert not delete_unsmoothed, "blur size set 0, but users wants unsmoothed data deleted"
        os.chdir(cur_dir)
        if keep_temp is not True:
            # remove temporary directory
            shutil.rmtree(tmp_dir)
    print_wrap('Vol2Surf run complete')
    return file_list


def surf_to_vol(subject, in_files, map_func='ave', wm_mod=0.0, gm_mod=0.0, prefix=None, index='voxels', steps=10,
                out_dir=None, fs_dir=None, surf_vol='standard', std141=False, keep_temp=False):
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
    tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))

    # check if subjects' freesurfer directory exists
    if fs_dir is None:
        fs_dir = os.environ["SUBJECTS_DIR"]
    if out_dir is None:
        out_dir = cur_dir
    # check if subjects' SUMA directory exists
    suffix = fs_dir_check(fs_dir, subject)
    suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir, subject, suffix)

    if wm_mod is not 0.0 or gm_mod is not 0.0:
        # for gm, positive values makes the distance longer, for wm negative values
        steps = round(steps + steps * gm_mod - steps * wm_mod)

    print_wrap("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod, gm_mod, steps))

    if surf_vol is "standard":
        vol_dir = "{0}/{1}{2}/SUMA".format(fs_dir, subject, suffix)
        vol_file = "{0}{1}_SurfVol+orig".format(subject, suffix)
    else:
        vol_dir = '/'.join(surf_vol.split('/')[0:-1])
        vol_file = surf_vol.split('/')[-1]
        if not vol_dir:  # if volume directory is empty
            vol_dir = cur_dir

    # make temporary copy of volume file     
    subprocess.call("3dcopy {0}/{1} {2}/{1}".format(vol_dir, vol_file, tmp_dir, vol_file), shell=True)

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

    copy_suma_files(suma_dir, tmp_dir, subject)

    os.chdir(tmp_dir)
    for cur_file in in_files:
        shutil.copy("{0}/{1}".format(cur_dir, cur_file), tmp_dir)
        file_name, file_suffix = get_name_suffix(cur_file, surface=True)

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
                        .format(specprefix, subject, suffix, hemi, vol_file, file_name, map_func, index, wm_mod, gm_mod,
                                steps, tmp_dir), shell=True)

        subprocess.call("3dcopy {2}/{0}+orig {1}/{0}.nii.gz".format(file_name, out_dir, tmp_dir), shell=True)

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def roi_templates(subjects, roi_type="all", atlasdir=None, fsdir=None, outdir="standard", forcex=False,
                  separate_out=False, keeptemp=False, skipclust=False, intertype="NearestNode",
                  force_new_mapping=False):
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
        old_subject_dir = os.environ["SUBJECTS_DIR"]
        fsdir = os.environ["SUBJECTS_DIR"]
    else:
        old_subject_dir = os.environ["SUBJECTS_DIR"]
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
    run_possible_arguments = ['benson', 'glasser', 'wang', 'kgs']
    roi_type = str(roi_type).lower()
    confirmation_str = ''
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
        print_wrap('Error - no correct option selected. Please input: benson, glasser, wang or KGS')
        return None
    for sub in subjects:  # loop over list of subjects
        print_wrap("Running {0}: {1}".format(sub, confirmation_str))
        # check if subjects' freesurfer directory exists
        if os.path.isdir("{0}/{1}".format(fsdir, sub)):
            # no suffix needed
            suffix = ""
        elif os.path.isdir("{0}/sub-{1}".format(fsdir, sub)):
            sub = "sub-{0}".format(sub)
            suffix = ""
        else:
            # suffix needed
            suffix = "_fs4"
            if not os.path.isdir("{0}/{1}{2}".format(fsdir, sub, suffix)):
                sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                         .format(fsdir, sub, suffix))
                # check the voxel size is even and res is 1x1x1
        vox_hdr = nib.load("{0}/{1}{2}/mri/orig.mgz".format(fsdir, sub, suffix)).header
        vox_shape = vox_hdr.get_data_shape()
        assert len([shape for shape in vox_shape if shape % 2 != 0]) == 0, 'Voxel Shape incorrect {0}'.format(vox_shape)
        vox_res = vox_hdr.get_zooms()
        assert vox_res == (1.0, 1.0, 1.0), 'Voxel Resolution incorrect: {0}'.format(vox_res)

        if outdir_flag != "custom":
            outdir = "{0}/{1}{2}/TEMPLATE_ROIS".format(fsdir, sub, suffix)
        else:
            outdir = "{0}/{1}/TEMPLATE_ROIS".format(outdir, sub, suffix)  # force sub in name, in case multiple subjects

        # make temporary, local folder
        tmpdir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))

        # and subfoldes
        os.mkdir(tmpdir + "/surf")
        os.mkdir(tmpdir + "/TEMPLATE_ROIS")
        os.mkdir(tmpdir + "/SUMA")

        # copy relevant freesurfer files & establish surface directory
        surfdir = "{0}/{1}{2}".format(fsdir, sub, suffix)

        for file in glob.glob(surfdir + "/surf/*h.white"):
            shutil.copy(file, tmpdir + "/surf")
        sumadir = "{0}/{1}{2}/SUMA".format(fsdir, sub, suffix)

        # Does SUMA folder exist - if not run @SUMA_Make_Spec_FS -NIFTI -sid subject1
        if not os.path.isdir(sumadir):
            print_wrap('running @SUMA_Make_Spec_FS', indent=1)
            os.chdir("{0}/{1}{2}".format(fsdir, sub, suffix))
            shell_cmd("@SUMA_Make_Spec_FS -NIFTI -sid {0}{1}".format(sub, suffix))
            file_format = "gii"
        else:
            # is SUMA data in .gii or .asc format?
            if len(glob.glob("{0}/{1}{2}/SUMA/*.asc".format(fsdir, sub, suffix))) > 0:
                file_format = "asc"
            elif len(glob.glob("{0}/{1}{2}/SUMA/*.gii".format(fsdir, sub, suffix))) > 0:
                file_format = 'gii'
            else:
                print_wrap('SUMA Error - no .asc or .gii files located')
                return None
        for file in glob.glob(sumadir + "/*h.smoothwm.{0}".format(file_format)):
            shutil.copy(file, tmpdir + "/SUMA")
        for file in glob.glob("{0}/{1}_*.spec".format(sumadir, sub)):
            shutil.copy(file, tmpdir + "/SUMA")
        # Copy existing mapping files
        mapfiles = {}
        for file in glob.glob("{0}/{1}{2}.std141_to_native.*.niml.M2M".format(sumadir, sub, suffix)):
            shutil.copy(file, tmpdir + "/SUMA")
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
            print_wrap("running Benson2014:", indent=1)
            if os.path.isdir(surfdir + "/xhemi") is False or forcex is True:
                print_wrap("doing fsaverage_sym registration", indent=2)
                # Invert the right hemisphere - currently removed as believed not needed
                # shell_cmd("xhemireg --s {0}{1}".format(sub,suffix), fsdir,do_print=True)
                # register lh to fsaverage sym
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh".format(sub, suffix), fsdir)
                # mirror-reverse subject rh and register to lh fsaverage_sym
                # though the right hemisphere is not explicitly listed below, it is implied by --lh --xhemi
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh --xhemi".format(sub, suffix), fsdir)
            else:
                print_wrap("skipping fsaverage_sym registration", indent=2)

            if separate_out:
                datalist = ["angle", "eccen", "areas", "all"]
            else:
                datalist = ["all"]

            for bdata in datalist:

                # resample right and left hemisphere data to symmetric hemisphere
                shell_cmd("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1} --trgsurfreg {2}.sphere.reg \
                    --hemi lh --sval {3}/{5}/{4}-template-2.5.sym.mgh --tval ./TEMPLATE_ROIS/lh.{5}.{4}.mgh"
                          .format(sub, suffix, "fsaverage_sym", atlasdir, bdata, outname, tmpdir), fsdir)
                shell_cmd("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1}/xhemi --trgsurfreg {2}.sphere.reg \
                    --hemi lh --sval {3}/{5}/{4}-template-2.5.sym.mgh --tval ./TEMPLATE_ROIS/rh.{5}.{4}.mgh"
                          .format(sub, suffix, "fsaverage_sym", atlasdir, bdata, outname, tmpdir), fsdir)
                # convert to suma
                for hemi in ["lh", "rh"]:
                    shell_cmd(
                        "mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.{2}.mgh ./surf/{0}.white ./TEMPLATE_ROIS/{0}.{1}.{2}.gii".format(
                            hemi, outname, bdata, tmpdir))
                    shell_cmd(
                        "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.{2}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.{2}.niml.dset".format(
                            hemi, outname, bdata, tmpdir))

        # GLASSER ROIS *******************************************************************
        if run_glasser == True:

            outname = 'Glasser2016'
            print_wrap("running Glasser2016:", indent=1)
            for hemi in ["lh", "rh"]:
                # convert from .annot to mgz
                shell_cmd(
                    "mri_annotation2label --subject fsaverage --hemi {0} --annotation {1}/{2}/{0}.HCPMMP1.annot --seg {0}.glassertemp1.mgz"
                    .format(hemi, atlasdir, outname))
                # convert to subjects native space
                shell_cmd(
                    "mri_surf2surf --srcsubject fsaverage --trgsubject {2}{3} --sval {0}.glassertemp1.mgz --hemi {0} --tval ./TEMPLATE_ROIS/{0}.{1}.mgz"
                    .format(hemi, outname, sub, suffix), fsdir)
                # convert mgz to gii
                shell_cmd("mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.mgz ./surf/{0}.white ./TEMPLATE_ROIS/{0}.{1}.gii"
                          .format(hemi, outname))
                # convert gii to niml.dset
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.niml.dset"
                    .format(hemi, outname))

        ## WANG ROIS *******************************************************************
        if run_wang == True:

            outname = 'Wang2015'

            for file in glob.glob("{0}/Wang2015/subj_surf_all/maxprob_surf_*.1D.dset".format(atlasdir)):
                shutil.copy(file, tmpdir + "/.")
            surf_to_surf_i = 'fs' if file_format == 'asc' else 'gii'
            for hemi in ["lh", "rh"]:
                # if you have a mapping file, this is much faster.  see SurfToSurf -help
                # you can still run without a mapping file, but it is generated on-the-fly (slow!)
                # mapping file may have already been generated - option 2 maybe generated
                try:
                    mapfile = mapfiles[hemi]
                except:
                    mapfile = ""
                print_wrap("running Wang: {0}:".format(hemi), indent=1)
                if os.path.isfile(mapfile) and not force_new_mapping:
                    print_wrap("using existing mapping file from SUMA dir", indent=2)
                    subprocess.call(
                        "SurfToSurf -i_{4} ./SUMA/{0}.smoothwm.{3} -i_{4} ./SUMA/std.141.{0}.smoothwm.{3} -output_params {1} -mapfile {2} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi, intertype, mapfile, file_format, surf_to_surf_i), shell=True)
                    newmap = False
                else:
                    print_wrap("generating new mapping file", indent=2)
                    newmap = True
                    subprocess.call(
                        "SurfToSurf -i_{3} ./SUMA/{0}.smoothwm.{2} -i_{3} ./SUMA/std.141.{0}.smoothwm.{2} -output_params {1} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                        .format(hemi, intertype, file_format, surf_to_surf_i), shell=True)
                    # update M2M file name to be more informative and not conflict across hemispheres
                    os.rename("./SurfToSurf.niml.M2M".format(outname, hemi),
                              "./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi))

                # give output file a more informative name
                os.rename("./SurfToSurf.maxprob_surf_{0}.niml.dset".format(hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                # convert output to gii
                shell_cmd(
                    "ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.gii".format(
                        outname, hemi))
                # we don't need this and it conflicts across hemisphere                    
                os.remove("./SurfToSurf.1D".format(outname, hemi))
                # for file in glob.glob("./maxprob_surf_*.1D.dset"):
                #    os.remove(file)

                # make a 1D.dset copy using the naming conventions of other rois,
                # so we can utilize some other script more easily (e.g., roi1_copy_surfrois_locally.sh)
                # mainly for Kastner lab usage
                subprocess.call(
                    "ConvertDset -o_1D -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prepend_node_index_1D -prefix ./TEMPLATE_ROIS/{0}.{1}.1D.dset"
                    .format(hemi, outname), shell=True)

                if not skipclust:  # do optional surface-based clustering
                    print_wrap("doing clustering", indent=2)
                    for idx in range(1, 26):
                        # clustering steps
                        specfile = "./SUMA/{0}{1}_{2}.spec".format(sub, suffix, hemi)
                        surffile = "./SUMA/{0}.smoothwm.{1}".format(hemi, file_format)

                        # isolate ROI
                        subprocess.call(
                            "3dcalc -a ./TEMPLATE_ROIS/{2}.{0}.niml.dset -expr 'iszero(a-{1})' -prefix {2}.temp.niml.dset"
                            .format(outname, idx, hemi), shell=True)
                        # do clustering, only consider cluster if they are 1 edge apart
                        subprocess.call(
                            "SurfClust -spec {0} -surf_A {1} -input {2}.temp.niml.dset 0 -rmm -1 -prefix {2}.temp2 -out_fulllist -out_roidset"
                            .format(specfile, surffile, hemi), shell=True)

                        # pick only biggest cluster
                        if idx is 1:
                            if os.path.isfile("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi)):
                                print_wrap(
                                    "removing existing file ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,
                                                                                                              hemi),
                                    indent=2)
                                os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                            subprocess.call(
                                "3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)*{2}' -prefix {1}.{0}_cluster.niml.dset"
                                .format(outname, hemi, idx), shell=True)
                        else:
                            subprocess.call(
                                "3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -b {1}.{0}_cluster.niml.dset -expr 'b+iszero(a-1)*{2}' -prefix {1}.temp3.niml.dset"
                                .format(outname, hemi, idx), shell=True)
                            # os.remove("./{0}/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                            os.rename("{0}.temp3.niml.dset".format(hemi),
                                      "{1}.{0}_cluster.niml.dset".format(outname, hemi))

                        for file in glob.glob("./*temp*"):
                            os.remove(file)
                    # is this step necessary?
                    subprocess.call(
                        "ConvertDset -input {1}.{0}_cluster.niml.dset -o_niml_asc -prefix ./TEMPLATE_ROIS/{1}.temp4.niml.dset"
                        .format(outname, hemi, idx), shell=True)
                    os.remove("{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    os.rename("./TEMPLATE_ROIS/{0}.temp4.niml.dset".format(hemi),
                              "./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    # convert output to gii
                    shell_cmd(
                        "ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.gii".format(
                            outname, hemi))
                # copy mapping file to subjects' home SUMA directory
                if newmap:
                    shutil.move("./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi),
                                "{3}/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi, sumadir))
                # convert data set to asc
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(
                        outname, hemi))
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname, hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                if not skipclust:
                    shell_cmd(
                        "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(
                            outname, hemi))
                    os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    os.rename("./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(outname, hemi),
                              "./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
        ##KGS ROIs *********************************************************************
        if run_kgs == True:

            outname = 'KGS2016'
            print_wrap("running KGS2016:", indent=1)
            os.chdir(tmpdir)
            for hemi in ["lh", "rh"]:

                idx = 0
                for roi in ["IOG", "OTS", "mFUS", "pFUS", "PPA", "VWFA1", "VWFA2"]:

                    idx += 1

                    if not os.path.isfile("{1}/{3}/{0}.MPM_{2}.label".format(hemi, atlasdir, roi, outname)):
                        # if label file does not exist, skip it
                        print_wrap("file {0}.MPM_{1}.label doesn't exist".format(hemi, roi), indent=2)
                        continue
                    # Make the intermediate (subject-native) surface:
                    #   --srcsubject is always fsaverage since we assume the input file is an fsaverage file
                    #   --trgsubject is the subject we want to convert to
                    #   --sval is the file containing the surface data
                    #   --hemi is just the hemisphere we want to surf-over
                    #   --tval is the output file  
                    subprocess.call("mri_label2label --srcsubject fsaverage --trgsubject {2}{3} --regmethod surface --hemi {0} \
                        --srclabel {1}/{5}/{0}.MPM_{4}.label --trglabel ./{0}.{4}_TEMP.label".format(hemi, atlasdir,
                                                                                                     sub, suffix, roi,
                                                                                                     outname),
                                    shell=True)

                    # convert to gifti
                    subprocess.call(
                        "mris_convert --label {0}.{1}_TEMP.label {1} ./surf/{0}.white {0}.{1}_TEMP.gii".format(hemi,
                                                                                                               roi),
                        shell=True)

                    # convert to .niml.dset
                    subprocess.call(
                        "ConvertDset -o_niml_asc -input {0}.{1}_TEMP.gii -prefix {0}.{1}_TEMP.niml.dset".format(hemi,
                                                                                                                roi),
                        shell=True)

                    # isolate roi of interest
                    # do clustering, only consider cluster if they are 1 edge apart
                    subprocess.call("SurfClust -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.{4} -input {0}.{1}_TEMP.niml.dset 0 \
                        -rmm -1 -prefix {0}.{1}_TEMP2.niml.dset -out_fulllist -out_roidset".format(hemi, roi, sub,
                                                                                                   suffix, file_format),
                                    shell=True)

                    # create mask, pick only biggest cluster
                    subprocess.call(
                        "3dcalc -a {0}.{1}_TEMP2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)' -prefix {0}.{1}_TEMP3.niml.dset".format(
                            hemi, roi), shell=True)

                    # dilate mask
                    subprocess.call(
                        "ROIgrow -spec ./SUMA/{2}{3}_{0}.spec -surf_A ./SUMA/{0}.smoothwm.{4} -roi_labels {0}.{1}_TEMP3.niml.dset -lim 1 -prefix {0}.{1}_TEMP4"
                        .format(hemi, roi, sub, suffix, file_format), shell=True)

                    numnodes = subprocess.check_output("3dinfo -ni {0}.{1}_TEMP3.niml.dset".format(hemi, roi),
                                                       shell=True)
                    numnodes = numnodes.decode('ascii')
                    numnodes = int(numnodes.rstrip("\n"))
                    numnodes = numnodes - 1
                    subprocess.call(
                        "ConvertDset -o_niml_asc -i_1D -input {0}.{1}_TEMP4.1.1D -prefix {0}.{1}_TEMP4.niml.dset -pad_to_node {2} -node_index_1D {0}.{1}_TEMP4.1.1D[0]"
                        .format(hemi, roi, numnodes), shell=True)

                    if idx == 1:
                        subprocess.call(
                            "3dcalc -a {0}.{1}_TEMP4.niml.dset -expr 'notzero(a)' -prefix {0}.{2}.niml.dset".format(
                                hemi, roi, outname), shell=True)
                    else:
                        subprocess.call("3dcalc -a {0}.{1}_TEMP4.niml.dset -b {0}.{2}.niml.dset \
                            -expr '(b+notzero(a)*{3})*iszero(and(notzero(b),notzero(a)))' -prefix {0}.{1}_TEMP5.niml.dset".format(
                            hemi, roi, outname, idx), shell=True)
                        shutil.move("{0}.{1}_TEMP5.niml.dset".format(hemi, roi),
                                    "{0}.{1}.niml.dset".format(hemi, outname))
                shutil.move("{0}.{1}.niml.dset".format(hemi, outname),
                            "./TEMPLATE_ROIS/{0}.{1}.niml.dset".format(hemi, outname))
                # convert from niml.dset to gii
                shell_cmd(
                    "ConvertDset -o_gii_asc -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prefix ./TEMPLATE_ROIS/{0}.{1}.gii".format(
                        hemi, outname))
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(
                        outname, hemi))
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname, hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
        os.chdir(curdir)

        if os.path.isdir(outdir):
            print_wrap("ROI output directory ""TEMPLATE_ROIS"" exists, adding '_new'", indent=1)
            shutil.move("{0}/TEMPLATE_ROIS".format(tmpdir), "{0}_new".format(outdir))
        else:
            shutil.move("{0}/TEMPLATE_ROIS".format(tmpdir), "{0}".format(outdir))
        if keeptemp is not True:
            # remove temporary directory
            shutil.rmtree(tmpdir)
    # reset the subjects dir
    os.environ["SUBJECTS_DIR"] = old_subject_dir


def surf_smooth(subject, infiles, fsdir=None, std141=None, blursize=0, keeptemp=False, outdir="standard"):
    """
    Function smooths surface MRI data
    - this takes in regular and standardised data
    """
    if fsdir is None:
        fsdir = os.environ["SUBJECTS_DIR"]
    # check if subjects' freesurfer directory exists
    if os.path.isdir("{0}/{1}".format(fsdir, subject)):
        # no suffix needed
        suffix = ""
    else:
        # suffix needed
        suffix = "_fs4"
        if not os.path.isdir("{0}/{1}{2}".format(fsdir, subject, suffix)):
            sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                     .format(fsdir, subject, suffix))
    if std141:
        specprefix = "std.141."
    else:
        specprefix = ""
    # create current,temporary,output directories
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))

    # Need to raise with Peter if this is okay - this will output files straight to end location
    # may also be that this breaks the BIDS format
    if outdir == "standard":
        outdir = "{0}/{1}{2}/".format(fsdir, subject, suffix)

    # copy relevant SUMA files
    sumadir = "{0}/{1}{2}/SUMA".format(fsdir, subject, suffix)
    print(sumadir)
    for file in glob.glob(sumadir + "/*h.smoothwm.asc"):
        shutil.copy(file, tmpdir)
    for file in glob.glob("{0}/{1}{2}{3}*.spec".format(sumadir, specprefix, subject, suffix)):
        shutil.copy(file, tmpdir)

    os.chdir(tmpdir)

    print(infiles)

    for curfile in infiles:
        if ".niml.dset" in curfile:
            filename = curfile[:-10]
            splitString = filename.split(".", 1)
            # print(splitString)
            hemi = splitString[0]
            # print(hemi)
            outname = "{0}_{1}fwhm".format(filename, blursize)
        else:
            sys.exit("ERROR!\n{0} is not .niml.dset format".format(curfile))

        # move files
        subprocess.call("3dcopy {1}/{0}.niml.dset {0}.niml.dset".format(filename, curdir), shell=True)
        print("files moved")
        # compute mean
        subprocess.call("SurfSmooth -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -met HEAT_07 -target_fwhm {4} -input {5}.niml.dset \
                    -cmask '-a {5}.niml.dset[0] -expr bool(a)' -output  {8}/{7}.niml.dset"
                        .format(specprefix, subject, suffix, hemi, blursize, filename, tmpdir, outname, outdir),
                        shell=True)
        print("mean computed")
    os.chdir(curdir)
    if keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir)


def fft_analysis(signal, tr=2.0, stimfreq=10, nharm=5, offset=0):
    """
    offset=0, positive values means the first data frame was shifted forwards relative to stimulus
              negative values means the first data frame was shifted backwards relative to stimulus
    """

    nT = signal.shape[0]
    nS = signal.shape[1]
    assert nT < 1000, "unlikely number of TRs, time should be first dimension"
    assert any([nT > 0, nS > 0]), "input must be two dimensional"
    sample_rate = 1 / tr
    run_time = tr * nT

    # subprocess.call("3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {0}.sc+orig"
    #            .format(file_name), shell=True)

    complex_vals = np.fft.rfft(signal, axis=0)
    complex_vals = np.divide(complex_vals[1:, :], nT)
    freq_vals = np.fft.rfftfreq(nT, d=1. / sample_rate)
    freq_vals = freq_vals.reshape(int(nT / 2 + 1), 1)
    freq_vals = freq_vals[1:]

    # compute full spectrum
    spectrum = np.abs(complex_vals)
    frequencies = freq_vals

    # compute mean cycle
    cycle_len = int(nT / stimfreq)
    # this code should work whether offset is 0, negative or positive
    pre_add = int(offset % cycle_len)
    post_add = int(cycle_len - pre_add)
    # add "fake cycle" by adding nans to the beginning and end of time series
    pre_nans = np.empty((pre_add, nS))
    pre_nans[:] = np.nan
    nu_signal = np.append(pre_nans, signal, axis=0)
    post_nans = np.empty((post_add, nS))
    post_nans[:] = np.nan
    nu_signal = np.append(nu_signal, post_nans, axis=0)
    # reshape, add one to stimfreq to account for fake cycle
    nu_signal = nu_signal.reshape(int(stimfreq + 1), int(nu_signal.shape[0] / (stimfreq + 1)), nS)
    # nan-average to get mean cycle
    mean_cycle = np.nanmean(nu_signal, axis=0)
    # subtract the mean to zero the mean_cycle
    mean_cycle = mean_cycle - np.tile(np.mean(mean_cycle, axis=0, keepdims=True), (cycle_len, 1))

    if np.isnan(mean_cycle).any() == True:
        print("mean cycle should not contain NaNs")
    assert np.isnan(mean_cycle).any() == False, "mean cycle should not contain NaNs"

    sig_z = np.empty((nharm, nS))
    sig_snr = np.empty((nharm, nS))
    sig_complex = np.empty((nharm, nS), dtype=np.complex_)
    sig_amp = np.empty((nharm, nS))
    sig_phase = np.empty((nharm, nS))
    noise_complex = np.empty((nharm, nS), dtype=np.complex_)
    noise_amp = np.empty((nharm, nS))
    noise_phase = np.empty((nharm, nS))
    for h_idx in range(nharm):
        idx_list = freq_vals == stimfreq * (h_idx + 1) / run_time

        # Calculate Z-score
        # Compute the mean and std of the non-signal amplitudes.  Then compute the
        # z-score of the signal amplitude w.r.t these other terms.  This appears as
        # the Z-score in the plot.  This measures how many standard deviations the
        # observed amplitude differs from the distribution of other amplitudes.
        sig_z[h_idx, :] = spectrum[idx_list[:, 0], :].reshape(1, nS) - np.mean(
            spectrum[np.invert(idx_list[:, 0]), :].reshape(-1, nS), axis=0, keepdims=True)
        sig_z[h_idx, :] = np.divide(sig_z[h_idx, :],
                                    np.std(spectrum[np.invert(idx_list[:, 0])].reshape(-1, nS), axis=0, keepdims=True))

        # calculate Signal-to-Noise Ratio. 
        # Signal divided by mean of 4 side bands
        signal_idx = int(np.where(idx_list[:, 0])[0])
        noise_idx = [signal_idx - 2, signal_idx - 1, signal_idx + 1, signal_idx + 2]
        sig_snr[h_idx, :] = spectrum[signal_idx, :].reshape(1, nS) / np.mean(spectrum[noise_idx, :].reshape(-1, nS),
                                                                             axis=0, keepdims=True)

        # compute complex, amp and phase
        # compute offset in radians, as fraction of cycle length
        offset_rad = float(offset) / cycle_len * (2 * np.pi)

        sig_complex[h_idx, :], sig_amp[h_idx, :], sig_phase[h_idx, :] = fft_offset(
            complex_vals[signal_idx, :].reshape(1, nS), offset_rad)
        noise_complex[h_idx, :], noise_amp[h_idx, :], noise_phase[h_idx, :] = fft_offset(
            complex_vals[noise_idx, :].reshape(-1, nS), offset_rad)

    return fftobject(
        spectrum=spectrum, frequencies=frequencies, mean_cycle=mean_cycle,
        sig_z=sig_z, sig_snr=sig_snr, sig_amp=sig_amp, sig_phase=sig_phase, sig_complex=sig_complex,
        noise_amp=noise_amp, noise_phase=noise_phase, noise_complex=noise_complex)


def roi_get_data(surf_files, roi_type="wang", is_time_series=True, sub=False, pre_tr=2, do_scale=True, do_detrend=True,
                 use_regressors='standard', offset=0, TR=2.0, roilabel=None, fsdir=os.environ["SUBJECTS_DIR"],
                 do_scaling=True, report_timing=False):
    """
    region of interest surface data
    
    Parameters
    ------------
    surf_files : list of strings
            A list of files to be run - this should be surface 
            data in .gii format
    roi_type : string, default "wang"
            This defaults to "wang". Only acceptable inputs are
            "wang", "benson", "wang+benson" and "whole_brain"
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
    t = time.time()
    if not sub:
        # determine subject from input data
        sub = surf_files[0][(surf_files[0].index('sub-') + 4):(surf_files[0].index('sub-') + 8)]
    elif "sub" in sub:
        # get rid of "sub-" string
        sub = sub[4:]
    # check if data from both hemispheres can be found in input
    # check_data
    l_files = []
    r_files = []
    for s in surf_files:
        if ".R." in s:
            s_2 = s.replace('.R.', '.L.')
            r_files.append(s)
        elif ".L." in s:
            s_2 = s.replace('.L.', '.R.')
            l_files.append(s)
        else:
            print_wrap("Hemisphere could not be determined from file {0}".format(s), indent=1)
            return None
        if s_2 not in surf_files:
            print_wrap("File {0} does not have a matching file from the other hemisphere".format(s), indent=1)

    data_files = [None, None]
    data_files[0] = sorted(l_files)
    data_files[1] = sorted(r_files)

    roi_type = roi_type.lower()
    if roi_type in ["whole", "whole_brain", "wholebrain"]:
        roi_type = "whole"
        outnames = ["whole-L", "whole-R"]
    else:
        # dictionary of RoIs
        roi_dic = {'wang': ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                            "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                            "IPS5", "SPL1", "FEF"],
                   'benson': ["V1", "V2", "V3"],
                   'wang_newlabel': ["V1v", "V1d", "V1", "V2v", "V2d", "V2", "V3v", "V3d", "V3", "hV4", "VO1", "VO2",
                                     "PHC1", "PHC2",
                                     "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4",
                                     "IPS5", "SPL1", "FEF"]
                   }
        # define roi files
        roi_file = [None, None]
        if roi_type == "wang":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fsdir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            roilabel = roi_dic[roi_type]
            newlabel = roi_dic[roi_type + '_newlabel']
        elif roi_type == "benson":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fsdir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            roilabel = roi_dic['benson']
            newlabel = roilabel
        elif roi_type == "wang+benson":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fsdir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            eccen_file = [None, None]
            eccen_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fsdir, sub)
            eccen_file[1] = eccen_file[0].replace("lh", "rh")
            # define roilabel based on ring centers
            ring_incr = 0.25
            ring_size = .5
            ring_max = 6
            ring_min = 1
            ring_centers = np.arange(ring_min, ring_max, ring_incr)  # list of ring extents
            ring_extents = [(x - ring_size / 2, x + ring_size / 2) for x in ring_centers]
            roilabel = [y + "_{:0.2f}".format(x) for y in roi_dic['benson'] for x in ring_centers]
            newlabel = roilabel
        else:
            # NB: NOT CLEAR HOW THIS WOULD WORK?
            if ".L." in roi_file[0]:
                roi_file[0] = roi_file[0]
                roi_file[1] = roi_file[0].replace('.L.', '.R.')
            elif ".R." in roi_file[1]:
                roi_file[1] = roi_file[1]
                roi_file[0] = roi_file[1].replace('.R.', '.L.')
            elif "lh" in roi_file[0]:
                roi_file[0] = roi_file[0]
                roi_file[1] = roi_file[0].replace('lh', 'rh')
            elif "rh" in roi_file[1]:
                roi_file[1] = roi_file[1]
                roi_file[0] = roi_file[1].replace('rh', 'lh')
            print("Unknown ROI labels, using numeric labels")
            max_idx = int(max(nl_surf.load_surf_data(roi_file[0])) + 1)
            newlabel = ["roi_{:02.0f}".format(x) for x in range(1, max_idx)]

        outnames = [x + "-L" for x in newlabel] + [x + "-R" for x in newlabel] + [x + "-BL" for x in newlabel]

    # create a list of outdata, with shared values 
    outdata = [roiobject(is_time_series=is_time_series, roiname=name, tr=TR, nharm=5, stimfreq=10, offset=offset) for
               name in outnames]

    print_wrap("subject {0}".format(sub), indent=1)

    data_n = [None, None]
    for h, hemi in enumerate(["L", "R"]):
        cur_files = data_files[h]
        if roi_type == "whole":
            hemi_data = []
            print_wrap("doing whole brain analysis on {0} {1}H data files".format(len(cur_files), hemi), indent=2)
        else:
            # roi-specific code begins here
            cur_roi = roi_file[h]
            try:
                # uses surface module of nilearn to import data in .gii format
                roi_data = nl_surf.load_surf_data(cur_roi)
                # Benson should just use ROIs
                if roi_type == 'benson':
                    roi_data = roi_data[:, 2]
            except OSError:
                print_wrap("ROI file: {0} could not be opened".format(cur_roi))
            roi_n = roi_data.shape[0]
            # wang+benson-specific code begins here
            if roi_type == "wang+benson":
                cur_eccen = eccen_file[h]
                try:
                    eccen_data = nl_surf.load_surf_data(cur_eccen)
                    # select eccen data from Benson
                    eccen_data = eccen_data[:, 1]
                except OSError:
                    print_wrap("Template eccen file: {0} could not be opened".format(cur_eccen))
                eccen_n = eccen_data.shape[0]
                assert eccen_n == roi_n, "ROIs and Template Eccen have different number of surface vertices"
                ring_data = np.zeros_like(roi_data)
                for r, evc in enumerate(roi_dic['benson']):
                    # find early visual cortex rois in wang rois
                    roi_set = set([i + 1 for i, s in enumerate(roi_dic['wang']) if evc in s])
                    # Find index of each roi in roi_data
                    roi_index = np.array([])
                    for item in roi_set:
                        roi_temp_index = np.array(np.where(roi_data == item)).flatten()
                        roi_index = np.concatenate((roi_index, roi_temp_index))
                    roi_index = roi_index.astype(int)
                    # define indices based on ring extents
                    for e, extent in enumerate(ring_extents):
                        # get indexes that are populated in both i.e. between lower and higher extent
                        eccen_idx = np.where((eccen_data > extent[0]) * (eccen_data < extent[1]))[0]
                        idx_val = e + (r * len(ring_centers))
                        ready_idx = list(set(eccen_idx) & set(roi_index))
                        ring_data[ready_idx] = idx_val + 1
                # now set ring values as new roi values
                roi_data = ring_data
                print_wrap("applying {0}+{1} to {2} {3}H data files".
                           format(cur_roi.split("/")[-1], cur_eccen.split("/")[-1], len(cur_files), hemi), indent=2)
            else:
                print_wrap("applying {0} to {1} {2}H data files".format(cur_roi.split("/")[-1], len(cur_files), hemi),
                           indent=2)

        for run_file in cur_files:
            try:
                cur_data = nl_surf.load_surf_data(run_file)
            except:
                print_wrap("Data file: {0} could not be opened".format(run_file))

            # remove pre_tr before scaling and detrending
            cur_data = cur_data[:, pre_tr:]

            if do_scale:
                try:
                    cur_mean = np.mean(cur_data, axis=1, keepdims=True)
                    # repeat mean along second dim to match data
                    cur_mean = np.tile(cur_mean, [1, cur_data.shape[1]])

                    # if "sub-0014_ses-01_task-cont_run-01_bold_space-surf.native_preproc.R.func.gii" in run_file:
                    #    print("hello")
                    # print(run_file)

                    # create mask array, to ignore non-positive values
                    mask = np.multiply(cur_data > 0, cur_mean > 0)

                    # assign zero to values that are non-positive in both the data and the means
                    new_data = np.zeros_like(cur_data)
                    # assign minimum value of 200 and data/mean*100
                    new_data[mask] = np.minimum(200, np.multiply(np.divide(cur_data[mask], cur_mean[mask]), 100))
                    # values should now be approx. between 0 and 200, with mean ~100
                    # subtract mean to set values between -100 and 100, with mean ~0
                    new_data = new_data - np.mean(new_data)

                    # assign to original variable
                    cur_data = new_data
                except:
                    print_wrap("Data file: {0}, scaling failure".format(run_file.split('/')[-1]))

            if do_detrend:
                try:
                    # transpose data for detrending
                    cur_data = np.transpose(cur_data)

                    # find confounds
                    run_id = run_file.split('bold')[0]
                    conf_file = glob.glob(run_id + '*confounds*')
                    assert len(conf_file) > 0, "no confound file found matching run file {0}".format(run_file)
                    assert len(conf_file) < 2, "more than one confound file matching run file {0}".format(run_file)

                    # load confounds as pandas data frame
                    df = pd.read_csv(conf_file[0], '\t', na_values='n/a')
                    # drop pre_trs
                    df = df[pre_tr:]

                    df_trs = df.values.shape[0]
                    assert (df_trs == cur_data.shape[0])

                    # select columns to use as nuisance regressors
                    if "standard" in use_regressors:
                        df = df[['CSF', 'WhiteMatter', 'GlobalSignal', 'FramewiseDisplacement', 'X', 'Y', 'Z', 'RotX',
                                 'RotY', 'RotZ']]
                    elif use_regressors not in "all":
                        df = df[use_regressors]
                    # fill in missing nuisance values with mean for that variable
                    for col in df.columns:
                        if sum(df[col].isnull()) > 0:
                            # replacing nan values of each column with its average
                            df[col] = df[col].fillna(np.mean(df[col]))
                    if run_file == cur_files[0]:
                        print_wrap("using {0} confound regressors".format(len(df.columns)), indent=3)

                    new_data = nl_signal.clean(cur_data, detrend=True, standardize=False, confounds=df.values,
                                               low_pass=None, high_pass=None, t_r=TR, ensure_finite=False)
                    cur_data = np.transpose(new_data)
                except:
                    print_wrap("Data file: {0}, detrending failure".format(run_file.split('/')[-1]))

            if data_n[h]:
                assert data_n[h] == cur_data.shape[
                    0], "two runs from {0}H have different number of surface vertices".format(hemi)
            else:
                data_n[h] = cur_data.shape[0]

            if roi_type == "whole":
                hemi_data.append(cur_data)
                if run_file == cur_files[-1]:
                    out_idx = outnames.index("whole-" + hemi)
                    hemi_data = np.mean(hemi_data, axis=0)
                    outdata[out_idx] = roiobject(np.transpose(hemi_data), curobject=outdata[out_idx])
            else:
                assert data_n[h] == roi_n, "Data and ROI have different number of surface vertices"
                for roi_name in newlabel:
                    # note,  account for one-indexing of ROIs
                    roi_set = set([i + 1 for i, s in enumerate(roilabel) if roi_name in s])
                    # Find index of each roi in roi_data
                    roi_index = np.array([])
                    for item in roi_set:
                        roi_temp_index = np.array(np.where(roi_data == item)).flatten()
                        roi_index = np.concatenate((roi_index, roi_temp_index))
                    roi_index = roi_index.astype(int)
                    num_vox = len(roi_index)
                    if num_vox == 0:
                        print_wrap(roi_name + "-" + hemi + " " + str(roi_set))
                    roi_t = np.mean(cur_data[roi_index], axis=0, keepdims=True)
                    out_idx = outnames.index(roi_name + "-" + hemi)
                    outdata[out_idx].num_vox = num_vox
                    outdata[out_idx] = roiobject(np.transpose(roi_t), curobject=outdata[out_idx])

                    if "R" in hemi and run_file == cur_files[-1]:
                        # do bilateral
                        other_idx = outnames.index(roi_name + "-" + "L")
                        bl_idx = outnames.index(roi_name + "-" + "BL")
                        bl_data = np.divide(outdata[other_idx].data + outdata[out_idx].data, 2)
                        num_vox = np.add(outdata[other_idx].num_vox, outdata[out_idx].num_vox)
                        outdata[bl_idx].num_vox = num_vox
                        outdata[bl_idx] = roiobject(bl_data, outdata[bl_idx])
    if report_timing:
        elapsed = time.time() - t
        print_wrap("ROI Surf Data run complete, took {0} seconds".format(elapsed))
    return (outdata, outnames)


def hotelling_t2(in_vals, alpha=0.05, test_mu=np.zeros((1, 1), dtype=np.complex), test_type="Hot"):
    assert np.iscomplexobj(in_vals), "all values must be complex"
    assert (alpha > 0.0) & (alpha < 1.0), "alpha must be between 0 and 1"

    # compare against zero?
    in_vals = in_vals.reshape(in_vals.shape[0], -1)
    if in_vals.shape[1] == 1:
        in_vals = np.append(in_vals, np.zeros(in_vals.shape, dtype=np.complex), axis=1)
        num_cond = 1
    else:
        num_cond = 2
        assert all(test_mu) == 0, "when two-dimensional complex input provided, test_mu must be complex(0,0)"
    assert in_vals.shape[1] <= 2, 'length of second dimension of complex inputs may not exceed two'

    # replace NaNs
    in_vals = in_vals[~np.isnan(in_vals)]
    # determine number of trials
    M = int(in_vals.shape[0] / 2);
    in_vals = np.reshape(in_vals, (M, 2))
    p = np.float(2.0);  # number of variables
    df1 = p;  # numerator degrees of freedom.

    if "hot" in test_type.lower():
        # subtract conditions
        in_vals = np.reshape(np.subtract(in_vals[:, 0], in_vals[:, 1]), (M, 1))
        df2 = M - p;  # denominator degrees of freedom.
        in_vals = np.append(np.real(in_vals), np.imag(in_vals), axis=1)
        samp_mu = np.mean(in_vals, 0)
        test_mu = np.append(np.real(test_mu), np.imag(test_mu))
        samp_cov_mat = np.cov(in_vals[:, 0], in_vals[:, 1])

        # Eqn. 2 in Sec. 5.3 of Anderson (1984), multiply by inverse of fraction used below::        
        t_crit = np.multiply(np.divide((M - 1) * p, df2), scp.stats.f.ppf(1 - alpha, df1, df2))

        # try
        inv_cov_mat = np.linalg.inv(samp_cov_mat)
        # Eqn. 2 of Sec. 5.1 of Anderson (1984):
        tsqrd = np.float(
            np.matmul(np.matmul(M * (samp_mu - test_mu), inv_cov_mat), np.reshape(samp_mu - test_mu, (2, 1))))
        # F approximation 
        tsqrdf = np.divide(df2, (M - 1) * p) * tsqrd;
        # use scipys F cumulative distribution function.
        p_val = 1.0 - scp.stats.f.cdf(tsqrdf, df1, df2)
    else:
        # note, if two experiment conditions are to be compared, we assume that the number of samples is equal
        df2 = num_cond * (2.0 * M - p);  # denominator degrees of freedom.

        # compute estimate of sample mean(s) from V & M 1991 
        samp_mu = np.mean(in_vals, 0.0)

        # compute estimate of population variance, based on individual estimates
        v_indiv = 1 / df2 * (np.sum(np.square(np.abs(in_vals[:, 0] - samp_mu[0])))
                             + np.sum(np.square(np.abs(in_vals[:, 1] - samp_mu[1]))))

        if num_cond == 1:
            # comparing against zero
            v_group = M / p * np.square(np.abs(samp_mu[0] - samp_mu[1]))
            # note, distinct multiplication factor
            mult_factor = 1 / M
        else:
            # comparing two conditions
            v_group = (np.square(M)) / (2 * (M * 2)) * np.square(np.abs(samp_mu[0] - samp_mu[1]))
            # note, distinct multiplication factor
            mult_factor = (M * 2) / (np.square(M))

        # Find critical F for corresponding alpha level drawn from F-distribution F(2,2M-2)
        # Use scipys percent point function (inverse of `cdf`) for f
        # multiply by inverse of multiplication factor to get critical t_circ
        t_crit = scp.stats.f.ppf(1 - alpha, df1, df2) * (1 / mult_factor)
        # compute the tcirc-statistic
        tsqrd = (v_group / v_indiv) * mult_factor;
        # M x T2Circ ( or (M1xM2/M1+M2)xT2circ with 2 conditions)
        # is distributed according to F(2,2M-2)
        # use scipys F probability density function
        p_val = 1 - scp.stats.f.cdf(tsqrd * (1 / mult_factor), df1, df2);
    return (tsqrd, p_val, t_crit)


def fit_error_ellipse(xydata, ellipse_type='SEM', make_plot=False, return_rad=True):
    """ Function uses eigen value decomposition
    to find two perpendicular axes aligned with xydata
    where the eigen vector correspond to variances 
    along each direction. An ellipse is fit to
    data at a distance from mean datapoint,
    depending on ellipse_type specified.
    
    Calculation for error ellipses based on
    alpha-specified confidence region (e.g. 
    95%CI or 68%CI) are calculated following
    information from Chapter 5 of Johnson & 
    Wickern (2007) Applied Multivatiate 
    Statistical Analysis, Pearson Prentice Hall
    
    Parameters
    ------------
    xydata : N x 2 matrix of 2D array
        Data contains real and imaginary data
        xydata should be [real, imag]
    ellipse_type : string, default 'SEM'
        Options are - SEM, 95%CI, or specific
        different percentage in format 'x%CI'
    make_plot : Boolean, default False
        Specifies whether or not to generate
        a plot of the data & ellipse & eigen
        vector
    return_rad : Boolean, default False
        Specifies whether to return values
        in radians or degrees.t
    Returns
    ------------
    amp_mean
    amp_diff : numpy array,
            differences of lower and upper bound
            for the mean amplitude
    phase_mean :
    phase_diff : numpy array,
            differences of lower and upper bound
            for the mean phase
    zSNR : float,
            z signal to noise ratio
    error_ellipse : numpy array,
            Array of error ellipses
    """

    # convert return_rad to an integer for indexing purposes later
    return_rad = int(return_rad)

    assert np.iscomplexobj(xydata), "all values must be complex"

    xydata = real_imag_split(xydata)

    n = xydata.shape[0]
    xydata = xydata.reshape(xydata.shape[0], -1)
    assert xydata.shape[1] == 2, 'data should be of dimensions: N x 2, currently: {0}'.format(xydata.shape[1])
    assert len(xydata.shape) <= 2, "data should not have more than 2 dimensions"

    try:
        (mean_xy, sampCovMat, smaller_eigenvec,
         smaller_eigenval, larger_eigenvec,
         larger_eigenval, phi) = eig_fouriercoefs(xydata)
    except:
        print('Unable to run eigen value decomposition. Probably data have only 1 sample')
        return None
    theta_grid = np.linspace(0, 2 * np.pi, num=100)
    if ellipse_type == '1STD':
        a = np.sqrt(larger_eigenval)
        b = np.sqrt(smaller_eigenval)
    elif ellipse_type == '2STD':
        a = 2 * np.sqrt(larger_eigenval)
        b = 2 * np.sqrt(smaller_eigenval)
    elif ellipse_type == 'SEMarea':
        # scale ellipse's area by sqrt(n)
        a = np.sqrt(larger_eigenval / np.sqrt(n))
        b = np.sqrt(smaller_eigenval / np.sqrt(n))
    elif ellipse_type == 'SEM' or ellipse_type == 'SEMellipse':
        # contour at stdDev/sqrt(n)
        a = np.sqrt(larger_eigenval) / np.sqrt(n)
        b = np.sqrt(smaller_eigenval) / np.sqrt(n)
    elif 'CI' in ellipse_type:
        # following Eqn. 5-19 Johnson & Wichern (2007)
        try:
            critVal = float(ellipse_type[:-3]) / 100
        except:
            print('ellipse_type incorrectly formatted, please see docstring')
            return None
        assert critVal < 1.0 and critVal > 0.0, 'ellipse_type CI range must be between 0 & 100'
        t0_sqrd = ((n - 1) * 2) / (n * (n - 2)) * stats.f.ppf(critVal, 2, n - 2)
        a = np.sqrt(larger_eigenval * t0_sqrd)
        b = np.sqrt(smaller_eigenval * t0_sqrd)
    else:
        print('ellipse_type Input incorrect, please see docstring')
        return None
    # the ellipse in x & y coordinates
    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_x_r = np.reshape(ellipse_x_r, (ellipse_x_r.shape[0], 1))
    ellipse_y_r = b * np.sin(theta_grid)
    ellipse_y_r = np.reshape(ellipse_y_r, (ellipse_y_r.shape[0], 1))

    # Define a rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    # rotate ellipse to some angle phi
    error_ellipse = np.dot(np.concatenate((ellipse_x_r, ellipse_y_r), axis=1), R)
    # shift to be centered on mean coordinate
    error_ellipse = np.add(error_ellipse, mean_xy)

    # find vector length of each point on ellipse
    norms = np.array([np.linalg.norm(error_ellipse[point, :]) for point in range(error_ellipse.shape[0])])
    ampMinNorm = min(norms)
    ampMinNormIx = np.argmin(norms)
    ampMaxNorm = max(norms)
    ampMaxNormIx = np.argmax(norms)
    amp_mean = np.linalg.norm(mean_xy)
    amp_bounds = np.array([ampMinNorm, ampMaxNorm])

    # calculate phase angles & find max pairwise difference to determine phase bounds
    phaseAngles = np.arctan2(error_ellipse[:, 1], error_ellipse[:, 0])
    pairs = np.array([np.array(comb) for comb in list(combinations(phaseAngles, 2))])
    diffPhase = np.absolute(pairs[:, 1] - pairs[:, 0])  # find absolute difference of each pair
    diffPhase[diffPhase > np.pi] = 2 * np.pi - diffPhase[diffPhase > np.pi]  # unwrap the difference
    maxDiffIdx = np.argmax(diffPhase)
    anglesOI = pairs[maxDiffIdx, :]
    phaseMinIx = np.argwhere(phaseAngles == anglesOI[0])[0]
    phaseMaxIx = np.argwhere(phaseAngles == anglesOI[1])[0]

    # convert to degrees (if necessary) & find diff between (max bound & mean phase) & (mean phase & min bound)
    # everything converted from [-pi, pi] to [0, 2*pi] for unambiguous calculation
    convFactor = np.array([180 / np.pi, 1])
    unwrap_factor = np.array([360, 2 * np.pi])
    phaseEllipseExtremes = anglesOI * convFactor[return_rad]
    phaseEllipseExtremes[phaseEllipseExtremes < 0] = phaseEllipseExtremes[phaseEllipseExtremes < 0] + unwrap_factor[
        return_rad]

    phaseBounds = np.array([min(phaseEllipseExtremes), max(phaseEllipseExtremes)])
    phase_mean = np.arctan2(mean_xy[1], mean_xy[0]) * convFactor[return_rad]
    if phase_mean < 0:
        phase_mean = phase_mean + unwrap_factor[return_rad]

    # if ellipse overlaps with origin, defined by whether phase angles in all 4 quadrants
    phaseAngles[phaseAngles < 0] = phaseAngles[phaseAngles < 0] + 2 * np.pi

    quad1 = phaseAngles[(phaseAngles > 0) & (phaseAngles < np.pi / 2)]
    quad2 = phaseAngles[(phaseAngles > np.pi / 2) & (phaseAngles < np.pi)]
    quad3 = phaseAngles[(phaseAngles > np.pi / 2) & (phaseAngles < 3 * np.pi / 2)]
    quad4 = phaseAngles[(phaseAngles > 3 * np.pi / 2) & (phaseAngles < 2 * np.pi)]
    if len(quad1) > 0 and len(quad2) > 0 and len(quad3) > 0 and len(quad4) > 0:
        amp_bounds = np.array([0, ampMaxNorm])
        maxVals = np.array([360, 2 * np.pi])
        phaseBounds = np.array([0, maxVals[return_rad]])
        phase_diff = np.array([np.absolute(phaseBounds[0] - phase_mean),
                               np.absolute([phaseBounds[1] - phase_mean])], ndmin=2)
    else:
        phase_diff = np.array([np.absolute(phaseBounds[0] - phase_mean), np.absolute(phaseBounds[1] - phase_mean)],
                              ndmin=2)

    # unwrap phase diff for any ellipse that overlaps with positive x axis

    phase_diff[phase_diff > unwrap_factor[return_rad] / 2] = unwrap_factor[return_rad] - phase_diff[
        phase_diff > unwrap_factor[return_rad] / 2]
    amp_diff = np.array([amp_mean - amp_bounds[0], amp_bounds[1] - amp_mean], ndmin=2)

    zSNR = amp_mean / np.mean(np.array([amp_mean - amp_bounds[0], amp_bounds[1] - amp_mean]))

    # Data plot
    if make_plot:
        # Below makes 2 subplots
        plt.figure(figsize=(9, 9))
        font = {'size': 16, 'color': 'k', 'weight': 'light'}
        # Figure 1 - eigen vector & SEM ellipse
        plt.subplot(1, 2, 1)
        plt.plot(xydata[:, 0], xydata[:, 1], 'ko', markerfacecolor='k')
        plt.plot([0, mean_xy[0]], [0, mean_xy[1]], linestyle='solid', color='k', linewidth=1)
        # plot ellipse
        plt.plot(error_ellipse[:, 0], error_ellipse[:, 1], 'b-', linewidth=1, label=ellipse_type + ' ellipse')
        # plot smaller eigen vec
        small_eigen_mean = [np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[0]) + mean_xy[0],
                            np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[1]) + mean_xy[1]]
        plt.plot([mean_xy[0], small_eigen_mean[0]], [mean_xy[1], small_eigen_mean[1]], 'g-',
                 linewidth=1, label='smaller eigen vec')
        # plot larger eigen vec
        large_eigen_mean = [np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[0]) + mean_xy[0],
                            np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[1]) + mean_xy[1]]
        plt.plot([mean_xy[0], large_eigen_mean[0]], [mean_xy[1], large_eigen_mean[1]], 'm-',
                 linewidth=1, label='larger eigen vec')
        # add axes
        plt.axhline(color='k', linewidth=1)
        plt.axvline(color='k', linewidth=1)
        plt.legend(loc=3, frameon=False)
        plt.axis('equal')

        # Figure 2 - mean amplitude, phase and amplitude bounds
        plt.subplot(1, 2, 2)
        # plot error Ellipse
        plt.plot(error_ellipse[:, 0], error_ellipse[:, 1], color='k', linewidth=1)

        # plot ampl. bounds
        plt.plot([0, error_ellipse[ampMinNormIx, 0]], [0, error_ellipse[ampMinNormIx, 1]],
                 color='r', linestyle='--')
        plt.plot([0, error_ellipse[ampMaxNormIx, 0]], [0, error_ellipse[ampMaxNormIx, 1]],
                 color='r', label='ampl. bounds', linestyle='--')
        font['color'] = 'r'
        plt.text(error_ellipse[ampMinNormIx, 0], error_ellipse[ampMinNormIx, 1],
                 round(ampMinNorm, 2), fontdict=font)
        plt.text(error_ellipse[ampMaxNormIx, 0], error_ellipse[ampMaxNormIx, 1],
                 round(ampMaxNorm, 2), fontdict=font)

        # plot phase bounds
        plt.plot([0, error_ellipse[phaseMinIx, 0]], [0, error_ellipse[phaseMinIx, 1]],
                 color='b', linewidth=1)
        plt.plot([0, error_ellipse[phaseMaxIx, 0]], [0, error_ellipse[phaseMaxIx, 1]],
                 color='b', linewidth=1, label='phase bounds')
        font['color'] = 'b'
        plt.text(error_ellipse[phaseMinIx, 0], error_ellipse[phaseMinIx, 1],
                 round(phaseEllipseExtremes[0], 2), fontdict=font)
        plt.text(error_ellipse[phaseMaxIx, 0], error_ellipse[phaseMaxIx, 1],
                 round(phaseEllipseExtremes[1], 2), fontdict=font)

        # plot mean vector
        plt.plot([0, mean_xy[0]], [0, mean_xy[1]], color='k', linewidth=1, label='mean ampl.')
        font['color'] = 'k'
        plt.text(mean_xy[0], mean_xy[1], round(amp_mean, 2), fontdict=font)

        # plot major/minor axis
        plt.plot([mean_xy[0], a * larger_eigenvec[0] + mean_xy[0]],
                 [mean_xy[1], a * larger_eigenvec[1] + mean_xy[1]],
                 color='m', linewidth=1)
        plt.plot([mean_xy[0], -a * larger_eigenvec[0] + mean_xy[0]],
                 [mean_xy[1], -a * larger_eigenvec[1] + mean_xy[1]],
                 color='m', linewidth=1)
        plt.plot([mean_xy[0], b * smaller_eigenvec[0] + mean_xy[0]],
                 [mean_xy[1], b * smaller_eigenvec[1] + mean_xy[1]],
                 color='g', linewidth=1)
        plt.plot([mean_xy[0], -b * smaller_eigenvec[0] + mean_xy[0]],
                 [mean_xy[1], -b * smaller_eigenvec[1] + mean_xy[1]],
                 color='g', linewidth=1)

        plt.axhline(color='k', linewidth=1)
        plt.axvline(color='k', linewidth=1)
        plt.legend(loc=3, frameon=False)
        plt.axis('equal')
        plt.show()
    return amp_mean, amp_diff, phase_mean, phase_diff, zSNR, error_ellipse

def subset_rois(in_file, roi_selection=["evc"], out_file=None, roi_labels="wang"):
    #roi_labels = ["V1d", "V1v", "V2d", "V2v", "V3A", "V3B", "V3d", "LO1", "TO1", "V3v", "VO1", "hV4"]
    if roi_labels == "wang":
        label_path = "{0}/ROI_TEMPLATES/Wang2015/ROIfiles_Labeling.txt".format(os.environ["SUBJECTS_DIR"])
        with open(label_path) as f:
            roi_labels = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
        roi_labels = [x.strip()[7:] for x in roi_labels[1:]]

    roi_match = []
    calc_expr = []
    for r, name in enumerate(roi_selection):
        idx = []
        # first handle annoying special cases, collections of rois
        if name.lower() == "v3":
            idx = [x for x, label in enumerate(roi_labels) if label.lower() in ["v3d", "v3v"]]
        elif name.lower() == "v3ab":
            idx = [x for x, label in enumerate(roi_labels) if label.lower() in ["v3a", "v3b"]]
        elif name.lower() == "evc":
            evc_names = ["v1d", "v1v", "v2d", "v2v", "v3d", "v3v"]
            idx = [x for x, label in enumerate(roi_labels) if label.lower() in evc_names]
        elif name.lower() == "to":
            idx = [x for x, label in enumerate(roi_labels) if label.lower() in ["hmt", "mst"]]
        else:
            idx = [x for x, label in enumerate(roi_labels) if name.lower() in label.lower()]
        # if idx is still empty, perhaps user is using other labels
        if not idx:
            if name.lower() == "v4":
                name = "hV4"
            elif name.lower() in ["to1", "mt"]:
                name = "hMT"
            elif name.lower() == "to2":
                name = "MST"
            idx = [x for x, label in enumerate(roi_labels) if name.lower() in label.lower()]
        if idx:
            roi_match.append((r + 1, [x + 1 for x in idx]))
            for x in idx:
                calc_expr.append("equals(a,{0})*{1}".format(x + 1, r + 1))
        else:
            roi_match.append((r + 1, [0]))
    # concatenate calc expressions, run and return command
    calc_expr = "+".join(calc_expr)

    if out_file:
        out_name = out_file.split("/")[-1]
        cur_dir = os.getcwd()
        tmp_dir = tempfile.mkdtemp("", "tmp", expanduser("~/Desktop"))
        os.chdir(tmp_dir)
        calc_cmd = "3dcalc -a {0} -expr '{1}' -prefix {2}".format(in_file, calc_expr, out_name)
        shell_cmd(calc_cmd)
        shutil.move(out_name, out_file)
        os.chdir(cur_dir)
        # remove temporary directory
        shutil.rmtree(tmp_dir)
    else:
        calc_cmd = "3dcalc -a {0} -expr '{1}' -prefix ????".format(in_file, calc_expr)

    return calc_cmd

def roi_subjects(exp_folder, fsdir=os.environ["SUBJECTS_DIR"], subjects='All', pre_tr=0, roi_type='wang+benson',
                 session='all', tasks='All', offset=None, file_type='fsnative', smooth=0, report_timing=True):
    """ Combine data across subjects - across RoIs & Harmonics
    So there might be: 180 RoIs x 5 harmonics x N subjects.
    This can be further split out by tasks. If no task is 
    provided then 
    This uses the RoiSurfData function carry out FFT.
    Parameters
    ------------ 
    exp_folder : string
        Folder location, required to identify files
        to be used
    fsdir : string/os file location, default to SUBJECTS_DIR
        Freesurfer folder directory
    subjects : string/list, Default 'All'
        Options:
            - 'All' - identifies and runs all subjects
            - ['sub-0001','sub-0002'] - runs only a list of subjects
    pre_tr : int, default 0
        input for RoiSurfData - please see for more info
    roi_type : str, default 'wang+benson'
        other options as per RoiSurfData function
    session : str, default '01'
        The fmri session
    tasks : list/string, default 'All'
        Options: 
            'All' - runs all tasks separately
            ['task1','task2'] - runs a subset of tasks
            None - If there is only one task
    offset : dictionary, default None
        offset to be applied to relevant tasks:
        Positive values means the first data frame was 
        shifted forwards relative to stimulus.
        Negative values means the first data frame was 
        shifted backwards relative to stimulus.
        example: {'cont':2,'disp':8,'mot':2}
    file_type : str, default 'fsnative'
        Should files for combined harmonics
        be 'fsnative','sumanative' or 'sumastd141'
    smooth: int, default 0
        if not zero, load in smoothed data
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
    t = time.time()
    if subjects == 'All':
        subjects = [files for files in os.listdir(exp_folder) if 'sub' in files and 'html' not in files]
    out_dic = {}
    if tasks == None:
        tasks = dict.fromkeys(["no task"], [pre_tr, offset])
    elif type(tasks) is not dict:
        if type(tasks) is str:
            # make it a list
            task_list = [tasks]
        else:
            task_list = tasks
        # if 'all' is in list, do all tasks
        if 'all' in [x.lower() for x in task_list]:
            task_list = []
            for sub in subjects:
                task_list += get_bids_data("{0}/{1}".format(exp_folder, sub))
            task_list = [re.findall('task-\w+_', x)[0][5:-1] for x in task_list]
            task_list = list(set(task_list))
        # make_dict and assign pre_tr and offset to dict
        tasks = dict.fromkeys(task_list, [pre_tr, offset])
    for task in tasks.keys():
        pre_tr = tasks[task][0]
        offset = tasks[task][1]

        for sub_int, sub in enumerate(subjects):
            # produce list of files
            if task is "no task":
                surf_files = get_bids_data("{0}/{1}".format(exp_folder, sub), file_type=file_type, session=session, smooth=smooth)
                if sub_int == 0:
                    print_wrap("Running SubjectAnalysis without considering task, pre-tr: {0}, offset: {1}".format(pre_tr, offset))
            else:
                surf_files = [f for f in
                              get_bids_data("{0}/{1}".format(exp_folder, sub), file_type=file_type, session=session, smooth=smooth) if
                              task in f]
                if sub_int == 0:
                    print_wrap("Running SubjectAnalysis on task {0}, pre-tr: {1}, offset: {2}".format(task, pre_tr, offset))
            if len(surf_files) > 0:
                # run RoiSurfData
                outdata, curnames = roi_get_data(surf_files, roi_type=roi_type, fsdir=fsdir, pre_tr=pre_tr,
                                                 offset=offset)
                # Define the empty array we want to fill or concatenate together
                if sub_int == 0:
                    outdata_arr = np.array(outdata, ndmin=2)
                    roinames = curnames
                else:
                    outdata_arr = np.concatenate((outdata_arr, np.array(outdata, ndmin=2)), axis=0)
                    assert roinames == curnames, "roi names do not match across subjects"
        out_dic[task] = {"data": outdata_arr, "pre_tr": pre_tr, "offset": offset, "roi_names": roinames,
                         "file_type": file_type}
    if report_timing:
        elapsed = time.time() - t
        print_wrap("SubjectAnalysis complete, took {0} seconds".format(elapsed))
    return out_dic


def whole_group(subject_data, harmonic_list=['1'], return_rad=True):
    """ Perform group analysis on subject data output from RoiSurfData.
    Parameters
    ------------
    subject_data : numpy array
        create this input using combineHarmonics()
        array dimensions: (roi_number, harmonic_number, subject_number)
    output : string or list or strs, default 'all'
        Can specify what output you would like.
        These are phase difference, amp difference and zSNR
        Options: 'all', 'phase', 'amp', 'zSNR',['phase','amp'] etc
    return_rad : boolean, default True
        Specify to return values in radians or degrees
    
    Returns
    ------------
    group_dictionary : dictionary
        broken down by task to include:
            ampPhaseZSNR_df : pd.DataFrame,
                contains RoIs as index, amp difference lower/upper, 
                phase difference lower/upper, zSNR
            error_ellipse_dic : dictionary,
                contains a dictionary of numpy arrays of the 
                error ellipses broken down by RoI.
    """
    start_time = time.time()
    print_wrap("Running group whole-brain analysis ...")
    group_dictionary = {}
    unwrap_factor = np.array([360, 2 * np.pi])
    wunwrap_factor = np.array([360, 2 * np.pi])
    for t, task in enumerate(subject_data.keys()):
        # dictionary for output of error Ellipse
        # ellipse_dic={}
        # number of rois, harmonics, subjects
        # subjects_n, roi_n = subject_data[task].shape
        # to be used to create the final columns in the dataframe
        # harmonic_name_list = ['RoIs']
        # Loop through harmonics & rois
        if t == 0:
            subjects_n = [x[0] for x in [subject_data[x]["data"].shape for x in subject_data.keys()]]
            sub_str = ', '.join(str(x) for x in subjects_n)
            roi_n = [x[1] for x in [subject_data[x]["data"].shape for x in subject_data.keys()]]
            roi_str = ', '.join(str(x) for x in roi_n)
            print_wrap("{0} conditions, {1} ROIs and {2} subjects".format(len(subject_data.keys()), roi_str, sub_str),
                       indent=1)

        assert all([x == 2 for x in roi_n]), "expected exactly two rois (one per hemisphere)"

        harmonic_n = len(harmonic_list)
        for r in range(roi_n[t]):

            for h in range(harmonic_n):
                # current harmonic 
                cur_harm = int(harmonic_list[h])
                xydata = [data.fft.sig_complex[cur_harm - 1] for data in subject_data[task]["data"][:, r]]
                xydata = np.array(xydata, ndmin=2)
                if h == 0 and r == 0:
                    # four values per harmonic: amp, phase, t2 and p-value
                    group_out = np.empty((xydata.shape[1], harmonic_n * 4, roi_n[t]))
                # compute elliptical errors
                split_data = real_imag_split(xydata)

                real_data = np.mean(split_data[:, 0, :], axis=0, keepdims=True)
                imag_data = np.mean(split_data[:, 1, :], axis=0, keepdims=True)

                group_out[:, h * 4, r] = np.abs(real_data + 1j * imag_data)
                phase_mean = np.angle(real_data + 1j * imag_data, not return_rad)
                # unwrap negative phases   
                phase_mean[phase_mean < 0] = phase_mean[phase_mean < 0] + unwrap_factor[int(return_rad)]
                group_out[:, h * 4 + 1, r] = phase_mean

                # compute Hotelling's T-squared:
                cur_hot = [hotelling_t2(xydata[:, x]) for x in range(xydata.shape[1])]
                # t2 value
                group_out[:, h * 4 + 2, r] = [x[0] for x in cur_hot]
                # p-value
                group_out[:, h * 4 + 3, r] = [x[1] for x in cur_hot]

        group_dictionary[task] = group_out
        elapsed = time.time() - start_time
    print_wrap("Group analysis complete, took {0} seconds".format(int(elapsed)))
    return group_dictionary


def roi_group(subject_data, harmonic_list=['1'], output='all', ellipse_type='SEM', make_plot=False, return_rad=True):
    """ Perform group analysis on subject data output from RoiSurfData.
    Parameters
    ------------
    subject_data : numpy array
        create this input using combineHarmonics()
        array dimensions: (roi_number, harmonic_number, subject_number)
    output : string or list or strs, default 'all'
        Can specify what output you would like.
        These are phase difference, amp difference and zSNR
        Options: 'all', 'phase', 'amp', 'zSNR',['phase','amp'] etc
    ellipse_type : string, default 'SEM'
        ellipse type SEM or in format: 'x%CI'
    make_plot : boolean, default False
        If True, will produce a plot for each RoI
    return_rad : boolean, default False
        Specify to return values in radians or degrees
    
    Returns
    ------------
    group_dictionary : dictionary
        broken down by task to include:
            ampPhaseZSNR_df : pd.DataFrame,
                contains RoIs as index, amp difference lower/upper, 
                phase difference lower/upper, zSNR
            error_ellipse_dic : dictionary,
                contains a dictionary of numpy arrays of the 
                error ellipses broken down by RoI.
    """
    start_time = time.time()
    print_wrap("Running group ROI analysis ...")
    group_dictionary = {}
    for t, task in enumerate(subject_data.keys()):
        # dictionary for output of error Ellipse
        # ellipse_dic={}
        # number of rois, harmonics, subjects
        # subjects_n, roi_n = subject_data[task].shape
        # to be used to create the final columns in the dataframe
        # harmonic_name_list = ['RoIs']
        # Loop through harmonics & rois
        if t == 0:
            subjects_n = [x[0] for x in [subject_data[x]["data"].shape for x in subject_data.keys()]]
            sub_str = ', '.join(str(x) for x in subjects_n)
            roi_n = [x[1] for x in [subject_data[x]["data"].shape for x in subject_data.keys()]]
            roi_str = ', '.join(str(x) for x in roi_n);
            print_wrap("{0} conditions, {1} ROIs and {2} subjects".format(len(subject_data.keys()), roi_str, sub_str),
                       indent=1)

        roi_names = subject_data[task]["roi_names"]

        harmonic_n = len(harmonic_list)
        for r in range(roi_n[t]):
            for h in range(harmonic_n):
                # current harmonic 
                cur_harm = int(harmonic_list[h])
                # ee_name = '{0}_harmonic_{1}'.format(roi_names[r],cur_harm)
                xydata = [data.fft.sig_complex[cur_harm - 1] for data in subject_data[task]["data"][:, r]]
                xydata = np.array(xydata, ndmin=2)

                # compute elliptical errors
                amp_mean, amp_diff, phase_mean, phase_diff, zSNR, error_ellipse = fit_error_ellipse(xydata,
                                                                                                    ellipse_type,
                                                                                                    make_plot,
                                                                                                    return_rad)

                assert amp_diff[
                           0, 0] >= 0, "warning: ROI {0} with task {1}: lower error bar is smaller than zero".format(
                    roi_names[r], task)
                assert amp_diff[
                           0, 1] >= 0, "warning: ROI {0} with task {1}: upper error bar is smaller than zero".format(
                    roi_names[r], task)

                # compute Hotelling's T-squared:
                hot_tval, hot_pval, hot_crit = hotelling_t2(xydata)

                if h == 0:  # first harmonic
                    cur_amp = np.concatenate((np.array(amp_mean, ndmin=2), amp_diff), axis=1);
                    cur_phase = np.concatenate((np.array(phase_mean, ndmin=2), phase_diff), axis=1);
                    cur_snr = np.array(zSNR, ndmin=2)
                    cur_hotT2 = np.concatenate((np.array(hot_tval, ndmin=2), np.array(hot_pval, ndmin=2)), axis=1);
                else:
                    cur_amp = np.concatenate((cur_amp, np.concatenate((np.array(amp_mean, ndmin=2), amp_diff), axis=1)),
                                             axis=2);
                    cur_phase = np.concatenate(
                        (cur_phase, np.concatenate((np.array(phase_mean, ndmin=2), phase_diff), axis=1)), axis=2);
                    cur_snr = np.concatenate((cur_snr, np.array(zSNR, ndmin=2)), axis=2)
                    cur_hotT2 = np.concatenate(
                        (cur_hotT2, np.concatenate((np.array(hot_tval, ndmin=2), np.array(hot_pval, ndmin=2)), axis=1)),
                        axis=2);

            # compute cycle average and standard error
            temp_cycle = [data.fft.mean_cycle for data in subject_data[task]["data"][:, r]]
            temp_cycle = np.squeeze(np.asarray(temp_cycle))
            cur_cycle_ave = np.array(np.mean(temp_cycle, axis=0), ndmin=2)
            sub_count = np.count_nonzero(~np.isnan(temp_cycle), axis=0)
            cur_cycle_stderr = np.array(np.divide(np.nanstd(temp_cycle, axis=0), np.sqrt(sub_count)), ndmin=2)
            cur_cycle = np.concatenate((cur_cycle_ave, cur_cycle_stderr))

            # construct the group ROI object            
            cur_obj = groupobject(amp=cur_amp, phase=cur_phase, zSNR=cur_snr, hotT2=cur_hotT2, cycle=cur_cycle,
                                  harmonics=harmonic_list, roi_name=roi_names[r])
            if r == 0:
                group_out = np.array(cur_obj, ndmin=1)
            else:
                group_out = np.concatenate((group_out, np.array(cur_obj, ndmin=1)), axis=0)

        group_dictionary[task] = group_out
        elapsed = time.time() - start_time
    print_wrap("Group analysis complete, took {0} seconds".format(int(elapsed)))
    return group_dictionary
