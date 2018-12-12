import os, subprocess, sys, glob, shutil, tempfile, re, time, textwrap, math
from nilearn import surface as nl_surf
from nilearn import signal as nl_signal
import numpy as np
import pandas as pd
import scipy as scp
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from joblib import Memory

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


def copy_surf_files(fs_dir, tmp_dir, subject, copy="both", suffix="", spec_prefix=""):
    sub_fs = "{0}/{1}{2}".format(fs_dir, subject, suffix)
    assert os.path.isdir(sub_fs), 'no freesurfer dir found'
    sub_suma = "{0}/SUMA".format(sub_fs)
    sub_surf = "{0}/surf".format(sub_fs)
    if copy.lower() == "both":
        copy = ["suma","fs"]
    if "suma" in copy:
        assert os.path.isdir(sub_suma), 'no SUMA dir found'
        suma_list = glob.glob("{0}/*h.smoothwm.gii".format(sub_suma)) + glob.glob("{0}/*h.pial.gii".format(sub_suma))
        suma_format = "gii"
        if len(suma_list) == 0:
            suma_list = glob.glob("{0}/*h.smoothwm.asc".format(sub_suma)) + glob.glob("{0}/*h.pial.asc".format(sub_suma))
            suma_format = "asc"
        suma_list = suma_list + glob.glob("{0}/{1}{2}{3}*.spec".format(sub_suma, spec_prefix, subject, suffix))
        assert len(suma_list) > 0, print_wrap('SUMA error - no .asc or .gii files located')
        # for some reason Vol2Surf needs the a2009 files
        suma_list = suma_list + glob.glob("{0}/*h.aparc.a2009s.annot.niml.dset".format(sub_suma, spec_prefix, subject, suffix))
        for file in suma_list:
            shutil.copy(file, tmp_dir)
    if "fs" in copy:
        assert os.path.isdir(sub_surf), 'no freesurfer surf dir found'
        fs_list = glob.glob("{0}/surf/*h.white".format(sub_fs))
        assert len(fs_list) > 0, 'freesurfer error - no ?h.white files located'
        for file in fs_list:
            shutil.copy(file, tmp_dir)
    return sub_suma, sub_fs, suma_format

def mri_parts(cur_file):
    path = "/".join(cur_file.split("/")[:-1])
    temp_name = cur_file.split("/")[-1]

    suffix_list = [".niml", ".gii", ".nii", "+orig"]
    suffix_idx = [temp_name.find(x) for x in suffix_list if temp_name.find(x) >= 0]

    assert len(suffix_idx) == 1, "more than one suffix matches, not supposed to happen!"

    suffix = temp_name[suffix_idx[0] + 1:]
    name = temp_name[:suffix_idx[0]]

    if suffix == ".niml.roi":
        # convert to niml.dset and return that
        shell_cmd("ROI2dataset -prefix {0}{1}.niml.dset -input {0}{1}.niml.roi".format(path, name))
        suffix = ".niml.dset"

    return path, name, suffix


def rsync(input, output):
    cmd = "rsync -avz --progress --remove-source-files %s/* %s/." % (input, output)
    p = subprocess.Popen(cmd, shell=True)
    stdout, stderr = p.communicate()
    return stderr


def shell_cmd(main_cmd, fs_dir=None, do_print=False):
    if fs_dir is not None:
        main_cmd = "export SUBJECTS_DIR={0}; {1}".format(fs_dir, main_cmd)
    if do_print:
        print(main_cmd + "\n")
    output = subprocess.check_output("{0}".format(main_cmd),
                                     shell=True, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf8')
    return output


def make_temp_dir(top_temp=os.path.expanduser("~/mymri_temp")):
    if not os.path.isdir(top_temp):
        os.makedirs(top_temp)
    tmp_dir = tempfile.mkdtemp("", "tmp_", top_temp)
    return tmp_dir


def fft_offset(complex_in, offset_rad):
    amp = np.absolute(complex_in)
    phase = np.angle(complex_in)
    # subtract offset from phase
    phase = phase - offset_rad
    # let phase vary between -pi and pi
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
    subjects = [x for x in os.listdir(experiment_dir) if 'sub' in x and os.path.isdir(os.path.join(experiment_dir, x))]
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
    def __init__(self, cur_data=np.zeros((120, 1)), cur_object=None, is_time_series=True, roi_names="unknown", tr=99,
                 stim_freq=99, nharm=99, num_vox=0, offset=0):
        if not cur_object:
            self.data = np.zeros((120, 1))
            self.roi_names = roi_names
            self.tr = tr
            self.stim_freq = stim_freq
            self.num_harmonics = nharm
            self.num_vox = num_vox
            self.is_time_series = is_time_series
            self.offset = offset
        else:
            # if curobject provided, inherit all values
            self.data = curobject.data
            self.roi_names = curobject.roi_name
            self.tr = curobject.tr
            self.stim_freq = curobject.stim_freq
            self.num_harmonics = curobject.num_harmonics
            self.num_vox = curobject.num_vox
            self.is_time_series = curobject.is_time_series
            self.offset = curobject.offset
        if cur_data.any():
            if self.data.any():
                self.data = np.dstack((self.data, cur_data))
            else:
                self.data = cur_data

    def mean(self):
        if len(self.data.shape) < 3:
            return self.data
        else:
            return np.mean(self.data, axis=2, keepdims=False)

    def fft(self):
        assert self.is_time_series == True, "not time series, cannot run fft"
        return fft_analysis(self.mean(),
                            tr=self.tr,
                            stimfreq=self.stim_freq,
                            nharm=self.num_harmonics,
                            offset=self.offset)

def mri_frame(cur_data=np.zeros((120, 1)), is_time_series=True, roi_name="whole", tr=99, stim_freq=99, nharm=99, num_vox=0, offset=0, lean=True):
    if "whole" in roi_name:
        idx = ["v-{:06d}".format(x + 1) for x in range(cur_data.shape[0])]
    else:
        idx = roi_name
    fft_keys = ["sig_complex", "mean_cycle"]
    if lean:
        cols = []
    else:
        cols = ["run-{0}".format(x + 1) for x in range(cur_data.shape[2])]
        cols = cols + ['average']
    cols = cols + ['num_vox'] + fft_keys
    num_t = cur_data.shape[1]

    # create data frame
    cur_frame = pd.DataFrame(index=idx, columns=cols)
    if not all(num_vox):
        num_vox = np.ones((cur_data.shape[0], 1))
    mean_data = np.nanmean(cur_data, axis=2)
    fft_data = fft_analysis(mean_data.reshape(num_t, -1), tr=tr, stimfreq=stim_freq, nharm=nharm, offset=offset)
    for x, i in enumerate(idx):
        if not lean:
            for r in range(cur_data.shape[2]):
                cur_frame.at[i, "run-{0}".format(r + 1)] = [cur_data[x, :, r]]
            cur_frame.at[i, "average"] = [mean_data[x,:]]
        for key in fft_keys:
            cur_frame.at[i, key ] = fft_data[key][:,x].tolist()
    cur_frame['num_vox'] = num_vox
    return cur_frame

# used by mrifft
class fftobject:
    def __init__(self, **kwargs):
        allowed_keys = ["spectrum", "frequencies", "mean_cycle", "sig_z", "sig_snr",
                        "sig_amp", "sig_phase", "sig_complex", "noise_complex", "noise_amp", "noise_phase"]
        for key in allowed_keys:
            setattr(self, key, [])
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

## MAIN FUNCTIONS
def mri_spec(task=None, session="01", space="suma_native", detrending=False, scaling=False, smoothing=0):
    out = {}
    out["task"]=task
    out["session"] = session
    out["space"] = space
    out["detrending"] = detrending
    out["smoothing"] = smoothing
    out["scaling"] = scaling
    return out


def write(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, -1)

def read(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def get_file_list(target_folder, type=".gii", spec=mri_spec()):
    file_list = []
    if isinstance(spec, dict):
        # convert dict to mri_spec dictionary
        spec = mri_spec(**spec)
        not_spec = []
        # data_spec is a class specifying bids format
        if spec["space"] in ['suma_std141', 'sumastd141']:
            spec_str = ["bold_space-sumastd141"]
        elif spec["space"] in ['suma_native', 'sumanative']:
            spec_str = ["bold_space-sumanative"]
        elif spec["space"] in ['fs_native', 'fsnative']:
            spec_str = ["bold_space-fsnative"]
        elif spec["space"] in "T1w":
            spec_str = ["bold_space-T1w"]
            # if T1w and BIDS, file name must also contain 'preproc'
            spec_str.append("preproc")
        else:
            print_wrap('unknown space {0} provided'.format(spec["space"]), indent=3)
            print_wrap('... using fsnative', indent=3)
            spec_str = ["bold_space-fsnative"]
        if spec["scaling"]:
            spec_str.append("-sc")
        else:
            not_spec.append("-sc")
        if spec["detrending"]:
            spec_str.append("-dt")
        else:
            not_spec.append("-dt")
        if spec["smoothing"] > 0:
            spec_str.append("{0}fwhm".format(spec["smoothing"]))
        else:
            not_spec.append("fwhm")
        if spec["task"]:
            spec_str.append("task-{0}".format(spec["task"]))

        # data folders for sessions or just current folder
        data_folders = ["{0}/{1}/func".format(target_folder, s) for s in os.listdir(target_folder) if 'ses' in s]
        if data_folders:
            if spec["session"] not in 'all':
                data_folders = ["{0}/ses-{1}/func".format(target_folder, str(spec["session"]).zfill(2))]
    else:
        if isinstance(spec, str):
            spec = [spec]
        spec_str = spec
        data_folders = [target_folder]

    assert type in [".nii.gz",".nii",".niml.dset",".gii.gz",".gii"], "unknown data type {0}!".format(type)

    for cur_dir in data_folders:
        file_list += [cur_dir + "/" + x for x in os.path.os.listdir(cur_dir) if
                      all(y in x for y in spec_str) and
                      all(z not in x for z in not_spec) and
                      x.endswith(type)]
    return file_list, spec


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

    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

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
        shell_cmd("afni -niml & SUMA -spec {0} -sv {1} &".format(spec_file, vol_file))
    else:
        shell_cmd("SUMA -spec {0} &".format(spec_file))


def neuro_to_radio(in_files):
    for scan in in_files:
        path, name, suffix = mri_parts(scan)
        old_orient = shell_cmd("fslorient -getorient {0}".format(scan))
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
        new_orient = shell_cmd("fslorient -getorient {0}".format(scan))
        print_wrap("new orientation: {0}".format(new_orient))


def slice_timing(in_files, ref_file='last', tr_dur=0, pre_tr=0, total_tr=0, slice_time_file=None, pad_ap=0, pad_is=0,
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
        tr_dur = shell_cmd("3dinfo -tr -short {0}".format(ref_file))
        tr_dur = tr_dur.rstrip("\n")
    if total_tr is 0:
        # include all TRs, so get max subbrick value
        total_tr = shell_cmd("3dinfo -nvi -short {0}".format(ref_file))
        total_tr = total_tr.rstrip("\n")
    else:
        # subject has given total number of TRs to include, add preTR to that
        total_tr = eval("pre_tr + total_tr")

    # make temporary, local folder
    cur_dir = os.getcwd()
    tmp_dir = make_temp_dir()
    os.chdir(tmp_dir)

    name_list = []

    for cur_file in in_files:
        file_name, suffix = mri_parts(cur_file)

        # crop and move files
        cur_total_tr = shell_cmd("3dinfo -nvi -short {1}/{0}{2}"
                                               .format(file_name, cur_dir, suffix))
        cur_total_tr = cur_total_tr.rstrip("\n")

        shell_cmd("3dTcat -prefix {0}+orig {1}/{0}{2}''[{3}..{4}]''"
                  .format(file_name, cur_dir, suffix, pre_tr, min(cur_total_tr, total_tr)))

        # slice timing correction
        if slice_time_file is None:
            shell_cmd("3dTshift -quintic -prefix {0}.ts+orig -TR {1}s -tzero 0 -tpattern alt+z {0}+orig"
                      .format(file_name, tr_dur))
        else:
            shell_cmd("3dTshift -quintic -prefix {0}/{1}.ts+orig -TR {2}s -tzero 0 -tpattern @{3} {0}/{1}+orig"
                      .format(tmp_dir, file_name, tr_dur, slice_time_file))

        # deoblique
        shell_cmd("3dWarp -deoblique -prefix {0}.ts.do+orig {0}.ts+orig".format(file_name))

        # pad 
        if pad_ap is not 0 or pad_is is not 0:
            shell_cmd(
                "3dZeropad -A {1} -P {1} -I {2} -S {2} -prefix {0}.ts.do.pad+orig {0}.ts.do+orig"
                    .format(file_name, pad_ap, pad_is))
            shell_cmd("rm {0}.ts.do+orig*".format(file_name))
            shell_cmd("3dRename {0}.ts.do.pad+orig {0}.ts.do+orig".format(file_name))

        if diff_mat:
            # add file_names to list, move later
            name_list.append(file_name)
            if cur_file == ref_file:
                ref_name = file_name
        else:
            shell_cmd("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.nii.gz {0}.ts.do+orig"
                      .format(file_name, cur_dir))

    if diff_mat:
        # take care of different matrices, and move
        for file_name in name_list:
            shell_cmd("@Align_Centers -base {1}.ts.do+orig -dset {0}.ts.do+orig".format(file_name, ref_name))
            shell_cmd("3dresample -master {1}.ts.do+orig -prefix {0}.ts.do.rs+orig -inset {0}.ts.do_shft+orig"
                      .format(file_name, ref_name))
            shell_cmd("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.rs.nii.gz {0}.ts.do.rs+orig"
                      .format(file_name, cur_dir))
    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def vol_reg(in_files, ref_file='last', slow=False, keep_temp=False):
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
    tmp_dir = make_temp_dir()
    os.chdir(tmp_dir)

    for cur_file in in_files:
        temp_path, file_name, suffix = mri_parts(cur_file)

        # move files
        shell_cmd("3dcopy {1}/{0}{2} {0}+orig".format(file_name, cur_dir, suffix))

        # do volume registration
        if slow:
            shell_cmd(
                "3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -heptic -twopass -maxite 50 {0}+orig"
                    .format(file_name, ref_file, cur_dir))
        else:
            shell_cmd(
                "3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -Fourier {0}+orig"
                    .format(file_name, ref_file, cur_dir))

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def scale_detrend(exp_folder, subjects=None, sub_prefix="sub-", tasks=None, pre_tr=0, total_tr=0, scale=True, detrend=True,
                data_spec = {}, bids_regressors="standard", in_format = ".nii.gz", overwrite=False, keep_temp=False):
    """
    Function for third stage of preprocessing: Scaling and Detrending.
    Typically run following mriPre.py and mriVolreg.py 
    Detrending currently requires motion registration parameters
    as .1D files: motparam.xxx.1D 

    Author: pjkohler, Stanford University, 2016
    """

    # figure out subjects
    if not subjects:
        subjects = [x for x in os.listdir(exp_folder) if
                    x.find(sub_prefix) == 0 and os.path.isdir(os.path.join(exp_folder, x))]
    # figure out tasks
    if tasks is None:
        tasks = dict.fromkeys(["no task"], [])
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
                task_list += get_file_list("{0}/{1}".format(exp_folder, sub), type=in_format, spec=data_spec)
            task_list = [re.findall('task-\w+_', x)[0][5:-1] for x in task_list]
            task_list = list(set(task_list))
        # make_dict and assign pre_tr and offset to dict
        tasks = dict.fromkeys(task_list, [])
    for task in tasks.keys():
        # Iterate over subjects
        for sub in subjects:

            # produce list of files
            if task is "no task":
                in_files, out_spec = get_file_list("{0}/{1}".format(exp_folder, sub), type=in_format, spec=data_spec)
                if len(in_files) == 0:
                    continue
                print_wrap("Preprocessing {0} on all tasks".format(sub))
            else:
                data_spec["task"] = task
                in_files, out_spec = get_file_list("{0}/{1}".format(exp_folder, sub), type=in_format, spec=data_spec)
                if len(in_files) == 0:
                    continue
                print_wrap("Preprocessing {0} on task {1}".format(sub, task))

            # make temporary, local folder
            cur_dir = os.getcwd()
            tmp_dir = make_temp_dir()
            os.chdir(tmp_dir)

            for cur_file in in_files:
                cur_dir, cur_name, cur_suffix = mri_parts(cur_file)
                final_suffix = ''
                if scale:
                    final_suffix = "{0}-sc".format(final_suffix)
                if detrend:
                    final_suffix = "{0}-dt".format(final_suffix)
                final_file = "{0}/{1}{2}.{3}".format(cur_dir,cur_name,final_suffix,cur_suffix)
                if os.path.isfile(final_file):
                    if overwrite:
                        os.remove(final_file)
                        print_wrap("Overwriting existing: {0}{1}.{2}".format(cur_name,final_suffix,cur_suffix), indent=1)
                    else:
                        print_wrap("Skipping existing: {0}{1}.{2}".format(cur_name,final_suffix,cur_suffix), indent=1)
                        continue
                else:
                    print_wrap("Creating new: {0}{1}.{2}".format(cur_name, final_suffix, cur_suffix), indent=1)
                # crop and move files
                cur_total_tr = shell_cmd("3dinfo -nv -short {0}/{1}.{2}"
                                                       .format(cur_dir, cur_name, cur_suffix))
                cur_total_tr = int(cur_total_tr.strip("\n"))

                if total_tr > 0:
                    cur_total_tr = min(cur_total_tr, total_tr + pre_tr)

                print_wrap("pre-tr: {0}, total_tr: {1}".format(pre_tr, cur_total_tr-pre_tr), indent=2)

                new_suffix = ''
                ## CONCATENATE
                # subtract one from total tr to account for zero-indexing
                shell_cmd("3dTcat -prefix {0}+orig {1}/{0}.{2}''[{3}..{4}]''"
                          .format(cur_name, cur_dir, cur_suffix, pre_tr, cur_total_tr-1))
                ## SCALE
                if scale:
                    # compute mean
                    shell_cmd("3dTstat -prefix mean_{0}{1}+orig {0}{1}+orig".format(cur_name, new_suffix))
                    shell_cmd("3dcalc -float -a {0}{1}+orig -b mean_{0}{1}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {0}{1}-sc+orig"
                        .format(cur_name, new_suffix))
                    new_suffix = "{0}-sc".format(new_suffix)
                # DETREND
                if detrend:
                    # find confounds
                    bids_confounds = glob.glob(cur_file.split('bold')[0] + '*confounds*.tsv')

                    if len(bids_confounds) > 0:
                        # load in bids data
                        assert len(bids_confounds) == 1, "more than one confound.tsv file found for data file {0}".format(
                            cur_file)
                        try:
                            # load confounds as pandas data frame
                            df = pd.read_csv(bids_confounds[0], '\t', na_values='n/a')
                            # drop pre_trs
                            df = df[pre_tr:cur_total_tr]

                            df_trs = df.values.shape[0]
                            assert (df_trs == cur_total_tr-pre_tr)

                            # select columns to use as nuisance regressors
                            if "standard" in bids_regressors:
                                df = df[['CSF', 'WhiteMatter', 'GlobalSignal', 'FramewiseDisplacement', 'X', 'Y', 'Z', 'RotX',
                                         'RotY', 'RotZ']]
                            elif "all" not in bids_regressors:
                                df = df[bids_regressors]
                            # fill in missing nuisance values with mean for that variable
                            for col in df.columns:
                                if sum(df[col].isnull()) > 0:
                                    # replacing nan values of each column with its average
                                    df[col] = df[col].fillna(np.mean(df[col]))
                                if all([math.isclose(x, 0, abs_tol=1e-09) for x in df[col].values]):
                                    # drop column if all values are the same
                                    df = df.drop([col], axis=1)
                                    print_wrap("Dropping {0}, all values very close to zero".format(col), indent=2)
                            print_wrap("Detrending, using {0} BIDS confound regressors".format(len(df.columns)), indent=2)

                        except:
                            print_wrap("Data file: {0}.{1}, failure in handling confound regressors".format(cur_name,cur_suffix))

                        confounds_file = "{0}/motparam.{1}.1D".format(tmp_dir, cur_name)
                        df.to_csv(confounds_file, sep=' ', index=False, header=False, float_format="%.5f" )
                    else:
                        # not so great way to find afni motparams for now
                        afni_confounds = [x for x in os.path.os.listdir(cur_dir) if
                                         x.endswith(".1D") and "motparam" in x and cur_name in x]
                        assert len(afni_confounds) > 0, "no bids confounds and no matching motparam 1D file found for data file {0}{1}".format(cur_name, cur_suffix)
                        assert len(afni_confounds) == 1, "more than one matching motparam 1D file found for data file {0}{1}".format(
                            cur_name, cur_suffix)

                        confounds_file = afni_confounds[0]
                        shutil.copyfile("{0}/{1}".format(cur_dir, confounds_file),"{0}/{1}".format(tmp_dir,confounds_file))
                        if cur_file == in_files[0]:
                            print_wrap("Detrending, using AFNI confound regressors in file {0}".format(confounds_file), indent=2)

                    shell_cmd("3dDetrend -prefix {0}{1}-dt+orig -polort 2 -vector {2} {0}{1}+orig"
                          .format(cur_name, new_suffix, confounds_file))
                    new_suffix = "{0}-dt".format(new_suffix)

                    # MOVE FILES BACK
                    assert final_file == "{3}/{0}{1}.{2}".format(cur_name, new_suffix, cur_suffix, cur_dir), "something's wrong, final file does not match suffix"
                    shell_cmd("3dAFNItoNIFTI -prefix {0} {1}{2}+orig".format(final_file, cur_name, new_suffix))

            os.chdir(cur_dir)
            if keep_temp is not True:
                # remove temporary directory
                shutil.rmtree(tmp_dir)


def vol_to_surf(experiment_dir, fs_dir=None, subjects=None, sub_prefix="sub-",
                data_spec = {}, in_format = "nii.gz", out_format="gii", overwrite=False,
                std141=False, surf_vol='standard', prefix=None,
                map_func='ave', wm_mod=0.0, gm_mod=0.0, index='voxels', steps=10, mask=None,
                keep_temp=False):
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
    ------------f
    experiment_dir : string
        The directory for the fmri data for the experiment
        Example: '/Volumes/Computer/Users/Username/Experiment/fmriprep'
    fs_dir : string, default os.environ["SUBJECTS_DIR"]
        Freesurfer directory
    subjects : list of strings, Default None
        Optional parameter, if wish to limit function to only certain subjects
        that are to be run. Example : ['sub-0001','sub-0002']
    sub_prefix: str, default: "sub-"
        prefix used for subject data directories
        useful when non-subject folders exist within experiment_dir
    data_spec: dict, list or str
        if dict, the data are stored in bids format, and dict contains
        {'space':'suma_native', 'session':'01', 'detrend': False, smoothing: 0}
        note that default parameters are given above.
        if list or str, non-bids format is assume,
        and data_spec is a str or list of strs that must be present in input files
    in_format : str, Default "".nii.gz"
        file type of input data
    out_format : str, Default "".gii"
        file type of output data
    surf_vol : string, default 'standard'
        File location of volume directory/file
    std141 : Boolean, default False
        Is subject to be run with standard 141?
    keep_temp : Boolean, default False
        Should temporary folder that is set up be kept after function
        runs?
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
        strange option that allows users to specify a prefix 'X' to the name of
        std141 files, such that std141.'X'.SUBNAME_lh.spec is the spec filename
    index : string, default 'voxels'
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_index. Specifies whether to use all seg points or
        unique volume voxels. Options: nodes, voxels
    steps : integer, default 10
        Parameter for AFNI 3dVol2Surf function. Parameter is:
        f_steps. Specify number of evenly spaced points along
        each segment
    mask : string, default None
        Parameter for AFNI 3dVol2Surf function.
        Produces a mask to be applied to input AFNI dataset.
    Returns 
    ------------
    file_list : list of strings
        This is a list of all files created by the function
    """

    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

    file_list = []

    if not subjects:
        subjects = [x for x in os.listdir(experiment_dir) if
                    x.find(sub_prefix) == 0 and os.path.isdir(os.path.join(experiment_dir, x))]

    # Dict to convert between old and new hemisphere notation
    hemi_dic = {'lh': 'L', 'rh': 'R'}

    # Iterate over subjects
    for sub in subjects:
        if std141:
            print_wrap("Running subject {0}, std141 template:".format(sub))
        else:
            print_wrap("Running subject {0}, native:".format(sub))

        in_files, out_spec = get_file_list("{0}/{1}".format(experiment_dir, sub), type=in_format, spec=data_spec)

        output_list = []

        # make temporary, local folder
        tmp_dir = make_temp_dir()

        # check if subjects' SUMA directory exists
        suffix = fs_dir_check(fs_dir, sub)
        suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir, sub, suffix)

        if surf_vol is "standard":
            vol_dir = "{0}/{1}{2}/SUMA".format(fs_dir, sub, suffix)
            vol_file = "{0}{1}_SurfVol.nii".format(sub, suffix)
        else:
            assert os.path.isfile(surf_vol), "surf vol {0} does not exist"
            vol_dir = '/'.join(surf_vol.split('/')[0:-1])
            vol_file = surf_vol.split('/')[-1]

        # make temporary copy of volume file
        shell_cmd("3dcopy {0}/{1} {2}/{1}".format(vol_dir, vol_file, tmp_dir))

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

        if wm_mod is not 0.0 or gm_mod is not 0.0:
            # for gm, positive values makes the distance longer, for wm negative values
            steps = round(steps + steps * gm_mod - steps * wm_mod)

        print_wrap("MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(wm_mod, gm_mod, steps), indent=1)

        os.chdir(tmp_dir)

        suma_copied = False
        for cur_file in in_files:
            cur_path, cur_name, cur_suffix = mri_parts(cur_file)
            if std141 == False:
                new = 'space-sumanative_'
            else:
                new = 'space-sumastd141_'
            if isinstance(data_spec, dict):
                old = re.findall('space-\w+_', cur_name)[0]
                output_name = cur_name.replace(old, new)
            else:
                output_name = '{0}_{1}'.format(cur_name, new)

            if all([os.path.isfile("{0}/{1}.{2}.{3}".format(cur_path, output_name, x, out_format)) for x in ['L','R']]):
                if overwrite:
                    print_wrap("Overwriting already converted file ...", indent=1)
                else:
                    print_wrap("Skipping already converted file ...", indent=1)
                    continue
            if not suma_copied:
                # copy suma files into the temporary directory
                copy_surf_files(fs_dir=fs_dir, tmp_dir=tmp_dir, subject=sub, copy="suma", suffix=suffix, spec_prefix=specprefix)
                suma_copied = True

            shutil.copyfile(cur_file, "{0}.{1}".format(cur_name, cur_suffix))

            if mask is None:
                # no mask
                maskcode = ""
            else:
                if mask is 'data':
                    # mask from input data
                    maskcode = "-cmask '-a {0}[0] -expr notzero(a)' ".format(cur_name)
                else:
                    # mask from distinct dataset, copy mask to folder
                    mask_path, mask_name, mask_suffix = mri_parts(mask)
                    if not mask_path:
                        mask_path = cur_path
                    shell_cmd("3dcopy {0} mask+orig".format(mask_path, mask_name, mask_suffix))
                    maskcode = "-cmask '-a mask+orig[0] -expr notzero(a)' "
            for hemi in ["lh", "rh"]:
                niml_file = "{0}.{1}.niml.dset".format(output_name, hemi_dic[hemi])
                # Converts volume to surface space - output in .niml.dset
                shell_cmd("3dVol2Surf -spec {0}{1}{2}_{3}.spec \
                        -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {5}.{6} -map_func {7} \
                        -f_index {8} -f_p1_fr {9} -f_pn_fr {10} -f_steps {11} \
                        -outcols_NSD_format -oob_value -0 {12}-out_niml {13}/{14}"
                          .format(specprefix, sub, suffix, hemi, vol_file, cur_name, cur_suffix, map_func, index, wm_mod,
                                  gm_mod, steps, maskcode, tmp_dir, niml_file), do_print=False)
                # this section should probably be made into a function, used all over
                if "niml.dset" in out_format:
                    output_file = niml_file
                elif "gii.gz" in out_format:
                    output_file = "{0}.{1}.{2}".format(output_name, hemi_dic[hemi], out_format)
                    shell_cmd("ConvertDset -o_gii_b64gz -input {0} -prefix {1}".format(niml_file, output_file))
                else:
                    output_file = "{0}.{1}.{2}".format(output_name, hemi_dic[hemi], out_format)
                    shell_cmd("ConvertDset -o_gii_b64 -input {0} -prefix {1}".format(niml_file, output_file))
                output_list.append(output_file)

        for o in output_list:
            print_wrap("... moving files ...", indent=2)
            # Removes output gii file if it exists
            o_path = "{0}/{1}".format(cur_path, o)
            if os.path.isfile(o_path):
                os.remove(o_path)

            # copy output file
            shutil.copyfile("{0}/{1}".format(tmp_dir, o), o_path)
            file_list.append(o_path)

        os.chdir(cur_path)
        if keep_temp is not True:
            # remove temporary directory
            shutil.rmtree(tmp_dir)
    print_wrap('Vol2Surf run complete')
    return file_list


def surf_smooth(experiment_dir, fs_dir=None, subjects=None, sub_prefix="sub-",
                data_spec = {"space": "suma_std141"}, in_format="gii", out_format = "gii", overwrite=False,
                blur_size=3.0, detrend_smooth=False, prefix=None, out_dir=None, keep_temp=False):
    """
    Function for smoothing surface MRI data
    Supports suma surfaces in native and std141 space,
    and can handle gifti or niml.dset format
    Wraps afnis SurfSmooth function:
    see: https://afni.nimh.nih.gov/pub/dist/doc/program_help/SurfSmooth.html

    Author: pjkohler, Stanford University, 2016
    Updated : fhethomas, 2018

    Parameters
    ------------
    experiment_dir : string
        The directory for the fmri data for the experiment
        Example: '/Volumes/Computer/Users/Username/Experiment/fmriprep'
    fs_dir : string, default os.environ["SUBJECTS_DIR"]
        Freesurfer directory
    subjects : list of strings, Default None
        Optional parameter, if wish to limit function to only certain subjects
        that are to be run. Example : ['sub-0001','sub-0002']
    sub_prefix : str, default: "sub-"
        prefix used for subject data directories
        useful when non-subject folders exist within experiment_dir
    data_spec: dict, list or str
        if dict, the data are stored in bids format, and dict contains
        {'space':'suma_std141', 'task':None, 'session':'01', 'detrend': False, smoothing: 0}
        note that default parameters are given above.
        if list or str, non-bids format is assume,
        and data_spec is a str or list of strs that must be present in input files
    in_format : str, Default "".gii"
        file format of input data
    out_format : str, Default "".gii"
        file format of output data
    blur_size: int, default 3
        value input to SurfSmooth's '-target_fwhm' parameter
    detrend_smooth : Boolean, default True
        if true, '-detrend_in 2' is added to the SurfSmooth cmd
    out_dir : str, path to output directory, default None
        if not given, smoothed files will be placed in same directory as input files
    prefix : string, default None
        strange option that allows users to specify a prefix 'X' to the name of
        std141 files, such that std141.'X'.SUBNAME_lh.spec is the spec filename
        no sure why this was ever necessary, but leaving it in for now
    keep_temp : Boolean, default False
        Should temporary folder that is set up be kept after function
        runs?
    """

    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

    blur_size = int(blur_size)
    assert blur_size > 0, " blur size {0} must be bigger than zero".format(blur_size)
    file_list = []

    if not subjects:
        subjects = [x for x in os.listdir(experiment_dir) if
                    x.find(sub_prefix) == 0 and os.path.isdir(os.path.join(experiment_dir, x))]

    # Dict to convert between old and new hemisphere notation
    hemi_dic = {'lh': 'L', 'rh': 'R', 'L': 'lh', 'R': 'rh'}

    # is it standard or std141
    std141 = False
    if isinstance(data_spec, dict):
        # convert dict to mri_spec object
        data_spec = mri_spec(**data_spec)
        if data_spec["space"] in ['suma_std141', 'sumastd141']:
            std141 = True
        bids_format = True
    else:
        if isinstance(data_spec, str):
            spec = [data_spec]
        if any(["std141" in x for x in data_spec]):
            std141 = True
        bids_format = False
    if detrend_smooth:
        detrend_cmd = "-detrend_in 2 "
    else:
        detrend_cmd = ""

    # Iterate over subjects
    for sub in subjects:
        if std141:
            print_wrap("Smoothing subject {0}, sumastd141 space, with a {1} fwhm kernel".format(sub, blur_size))
        else:
            print_wrap("Smoothing subject {0}, native space, with a {1} fwhm kernel".format(sub, blur_size))

        in_files, out_spec = get_file_list("{0}/{1}".format(experiment_dir, sub), spec=data_spec, type=in_format)

        suma_copied = False
        for cur_file in in_files:
            cur_dir, cur_name, cur_suffix = mri_parts(cur_file)

            if bids_format:
                processing = cur_name.split("_")[-1]
                hemi = processing.split(".")[-1]
                processing = processing.split(".")[0]
                new_name = cur_name.replace(processing, processing + "-{0}fwhm".format(blur_size))
            else:
                if any([x in cur_name for x in ["lh", ".L"]]):
                    hemi = "L"
                elif any([x in cur_name for x in ["rh", ".R"]]):
                    hemi = "R"
                else:
                    print_wrap("hemisphere could not be determined for file {0}".format(cur_file))
                    return
                new_name = '{0}-{1}fwhm'.format(cur_name, blur_size)

            if out_dir:
                cur_dir = out_dir

            if os.path.isfile("{0}/{1}.{2}".format(cur_dir, new_name, out_format)):
                if overwrite:
                    print_wrap("Overwriting already converted file ...", indent=1)
                else:
                    print_wrap("Skipping already converted file ...", indent=1)
                    continue
            if not suma_copied:
                # make temporary, local folder
                tmp_dir = make_temp_dir()

                # check if subjects' SUMA directory exists
                suffix = fs_dir_check(fs_dir, sub)
                suma_dir = "{0}/{1}{2}/SUMA".format(fs_dir, sub, suffix)

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
                copy_surf_files(fs_dir=fs_dir, tmp_dir=tmp_dir, subject=sub, copy="suma", suffix=suffix, spec_prefix=specprefix)

                os.chdir(tmp_dir)
                suma_copied = True

            # copy input file
            shutil.copyfile(cur_file, "{0}.{1}".format(cur_name, cur_suffix))

            # run smoothing
            shell_cmd("SurfSmooth -spec {0}{1}{2}_{3}.spec \
                                -surf_A smoothwm -met HEAT_07 -target_fwhm {4} -input {5}.{6} \
                                -cmask '-a {5}.{6}[0] -expr bool(a)' {7}-output {8}.{6}"
                      .format(specprefix, sub, suffix, hemi_dic[hemi], blur_size, cur_name, out_format, detrend_cmd,
                              new_name))

            # Removes output gii file if it exists
            out_path = "{0}/{1}.{2}".format(cur_dir, new_name, out_format)
            if os.path.isfile(out_path):
                os.remove(out_path)

            # copy output file
            shutil.copyfile("{0}/{1}.{2}".format(tmp_dir, new_name, out_format), out_path)
            # copy smoothing record file, give more reasonable name
            shutil.copyfile("{0}/{1}.{2}.1D.smrec".format(tmp_dir, new_name, out_format),
                            "{0}/{1}.smrec.1D".format(cur_dir, new_name))

        os.chdir(cur_dir)
        if keep_temp is not True:
            # remove temporary directory
            shutil.rmtree(tmp_dir)


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
    tmp_dir = tmp_dir = make_temp_dir()

    # check if subjects' freesurfer directory exists
    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

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
    shell_cmd("3dcopy {0}/{1} {2}/{1}".format(vol_dir, vol_file, tmp_dir, vol_file))

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

    copy_surf_files(fs_dir=fs_dir, tmp_dir=tmp_dir, subject=subject, copy="suma", suffix=suffix, spec_prefix=specprefix)

    os.chdir(tmp_dir)
    for cur_file in in_files:
        shutil.copy("{0}/{1}".format(cur_dir, cur_file), tmp_dir)
        file_path, file_name, file_suffix = mri_parts(cur_file)

        if 'lh' in file_name.split('.'):
            hemi = 'lh'
        elif 'rh' in file_name.split('.'):
            hemi = 'rh'
        else:
            sys.exit("ERROR! Hemisphere could not be deduced from: '{0}'."
                     .format(cur_file))

        shell_cmd("3dSurf2Vol -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {4} \
                    -sdata {5}.niml.dset -map_func {6} -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                    -prefix {11}/{5}"
                  .format(specprefix, subject, suffix, hemi, vol_file, file_name, map_func, index, wm_mod, gm_mod,
                          steps, tmp_dir))

        shell_cmd("3dcopy {2}/{0}+orig {1}/{0}.nii.gz".format(file_name, out_dir, tmp_dir))

    os.chdir(cur_dir)
    if keep_temp is not True:
        # remove temporary directory
        shutil.rmtree(tmp_dir)


def roi_templates(subjects, roi_type="all", atlasdir=None, fs_dir=None, out_dir="standard", forcex=False,
                  separate_out=False, keep_temp=False, skipclust=False, intertype="NearestNode",
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
    fs_dir : string, default None
            The freesurfer directory of subjects
    out_dir : string, default "standard"
            Output directory
    forcex : boolean, default False
            If there is no xhemi directiory, then as part of Benson;
            register lh to fsaverage sym & mirror-reverse subject rh 
            and register to lh fsaverage_sym
    separate_out : boolean, default False
            Can choose to separate out as part of Benson into ["angle", "eccen", 
            "areas", "all"]
    keep_temp : boolean, default False
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
    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")
        old_subject_dir = fs_dir
    else:
        old_subject_dir = os.getenv("SUBJECTS_DIR")
        os.environ["SUBJECTS_DIR"] = fs_dir
    if atlasdir is None:
        atlasdir = "{0}/ROI_TEMPLATES".format(fs_dir)
    # is a userinput output directory used?
    if out_dir != 'standard':
        outdir_flag = 'custom'
    else:
        outdir_flag = out_dir

    # get current directory    
    cur_dir = os.getcwd()

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
        suffix = fs_dir_check(fs_dir, sub)
        # check the voxel size is even and res is 1x1x1
        vox_hdr = nib.load("{0}/{1}{2}/mri/orig.mgz".format(fs_dir, sub, suffix)).header
        vox_shape = vox_hdr.get_data_shape()
        assert len([shape for shape in vox_shape if shape % 2 != 0]) == 0, 'Voxel Shape incorrect {0}'.format(vox_shape)
        vox_res = vox_hdr.get_zooms()
        assert vox_res == (1.0, 1.0, 1.0), 'Voxel Resolution incorrect: {0}'.format(vox_res)

        if outdir_flag != "custom":
            out_dir = "{0}/{1}{2}/TEMPLATE_ROIS".format(fs_dir, sub, suffix)
        else:
            out_dir = "{0}/{1}/TEMPLATE_ROIS".format(out_dir, sub, suffix)  # force sub in name, in case multiple subjects

        # make temporary, local folder
        tmp_dir = make_temp_dir()
        # and subfolders
        os.mkdir(tmp_dir + "/TEMPLATE_ROIS")

        # copy SUMA and freesurfer files
        sub_suma, sub_fs, suma_format = copy_surf_files(fs_dir, tmp_dir, sub, copy="both", suffix=suffix)

        # Copy existing mapping files
        mapfiles = [0, 0]
        for file in glob.glob("{0}/{1}{2}.std141_to_native.*.niml.M2M".format(sub_suma, sub, suffix)):
            shutil.copy(file, tmp_dir)
            if 'lh' in file:
                mapfiles[0] = file
            else:
                mapfiles[1] = file

        os.chdir(tmp_dir)

        # BENSON ROIS *******************************************************************
        if run_benson == True:
            outname = 'Benson2014'
            print_wrap("running Benson2014:", indent=1)
            if os.path.isdir(sub_fs + "/xhemi") is False or forcex is True:
                print_wrap("doing fsaverage_sym registration", indent=2)
                # Invert the right hemisphere - currently removed as believed not needed
                # shell_cmd("xhemireg --s {0}{1}".format(sub,suffix), fs_dir,do_print=True)
                # register lh to fsaverage sym
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh".format(sub, suffix), fs_dir)
                # mirror-reverse subject rh and register to lh fsaverage_sym
                # though the right hemisphere is not explicitly listed below, it is implied by --lh --xhemi
                shell_cmd("surfreg --s {0}{1} --t fsaverage_sym --lh --xhemi".format(sub, suffix), fs_dir)
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
                          .format(sub, suffix, "fsaverage_sym", atlasdir, bdata, outname, tmp_dir), fs_dir)
                shell_cmd("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1}/xhemi --trgsurfreg {2}.sphere.reg \
                    --hemi lh --sval {3}/{5}/{4}-template-2.5.sym.mgh --tval ./TEMPLATE_ROIS/rh.{5}.{4}.mgh"
                          .format(sub, suffix, "fsaverage_sym", atlasdir, bdata, outname, tmp_dir), fs_dir)
                # convert to suma
                for hemi in ["lh", "rh"]:
                    shell_cmd(
                        "mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.{2}.mgh {0}.white ./TEMPLATE_ROIS/{0}.{1}.{2}.gii".format(
                            hemi, outname, bdata, tmp_dir))
                    shell_cmd(
                        "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.{2}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.{2}.niml.dset".format(
                            hemi, outname, bdata, tmp_dir))

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
                        .format(hemi, outname, sub, suffix), fs_dir)
                # convert mgz to gii
                shell_cmd("mris_convert -f ./TEMPLATE_ROIS/{0}.{1}.mgz {0}.white ./TEMPLATE_ROIS/{0}.{1}.gii"
                          .format(hemi, outname))
                # convert gii to niml.dset
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{0}.{1}.gii -prefix ./TEMPLATE_ROIS/{0}.{1}.niml.dset"
                        .format(hemi, outname))

        ## WANG ROIS *******************************************************************
        if run_wang == True:

            outname = 'Wang2015'

            for file in glob.glob("{0}/Wang2015/subj_surf_all/maxprob_surf_*.1D.dset".format(atlasdir)):
                shutil.copy(file, tmp_dir + "/.")
            surf_to_surf_i = 'fs' if suma_format == 'asc' else 'gii'
            print_wrap("running Wang", indent=1)
            if all([ os.path.isfile(x) for x in mapfiles ]) and not force_new_mapping:
                print_wrap("using existing mapping file from SUMA dir", indent=2)
                newmap = False
            else:
                print_wrap("generating new mapping file", indent=2)
                newmap = True
            if not skipclust:  # do optional surface-based clustering
                print_wrap("doing clustering", indent=2)

            for h, hemi in enumerate(["lh", "rh"]):
                # if you have a mapping file, this is much faster.  see SurfToSurf -help
                # you can still run without a mapping file, but it is generated on-the-fly (slow!)
                # mapping file may have already been generated - option 2 maybe generated
                if newmap:
                    shell_cmd(
                        "SurfToSurf -i_{3} {0}.smoothwm.{2} -i_{3} std.141.{0}.smoothwm.{2} -output_params {1} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                            .format(hemi, intertype, suma_format, surf_to_surf_i))
                    # update M2M file name to be more informative and not conflict across hemispheres
                    os.rename("./SurfToSurf.niml.M2M".format(outname, hemi),
                              "{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi))
                else:
                    shell_cmd(
                        "SurfToSurf -i_{4} {0}.smoothwm.{3} -i_{4} std.141.{0}.smoothwm.{3} -output_params {1} -mapfile {2} -dset maxprob_surf_{0}.1D.dset'[1..$]'"
                            .format(hemi, intertype, mapfiles[h], suma_format, surf_to_surf_i))

                # give output file a more informative name
                os.rename("./SurfToSurf.maxprob_surf_{0}.niml.dset".format(hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))

                # we don't need this and it conflicts across hemisphere                    
                os.remove("./SurfToSurf.1D".format(outname, hemi))

                # make a 1D.dset copy using the naming conventions of other rois,
                # so we can utilize some other script more easily (e.g., roi1_copy_surfrois_locally.sh)
                # mainly for Kastner lab usage
                shell_cmd(
                    "ConvertDset -o_1D -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prepend_node_index_1D -prefix ./TEMPLATE_ROIS/{0}.{1}.1D.dset"
                        .format(hemi, outname))

                if not skipclust:  # do optional surface-based clustering
                    for idx in range(1, 26):
                        # clustering steps
                        specfile = "{0}{1}_{2}.spec".format(sub, suffix, hemi)
                        surffile = "{0}.smoothwm.{1}".format(hemi, suma_format)

                        # isolate ROI
                        shell_cmd(
                            "3dcalc -a ./TEMPLATE_ROIS/{2}.{0}.niml.dset -expr 'iszero(a-{1})' -prefix {2}.temp.niml.dset"
                                .format(outname, idx, hemi))
                        # do clustering, only consider cluster if they are 1 edge apart
                        shell_cmd(
                            "SurfClust -spec {0} -surf_A {1} -input {2}.temp.niml.dset 0 -rmm -1 -prefix {2}.temp2 -out_fulllist -out_roidset"
                                .format(specfile, surffile, hemi))

                        # pick only biggest cluster
                        if idx is 1:
                            # create new cluster file, or overwrite existing (should never happen)
                            if os.path.isfile("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi)):
                                print_wrap(
                                    "removing existing file ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname,
                                                                                                              hemi),
                                    indent=2)
                                os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                            shell_cmd(
                                "3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)*{2}' -prefix {1}.{0}_cluster.niml.dset"
                                    .format(outname, hemi, idx))
                        else:
                            # add to existing cluster file
                            shell_cmd(
                                "3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -b {1}.{0}_cluster.niml.dset -expr 'b+iszero(a-1)*{2}' -prefix {1}.temp3.niml.dset"
                                    .format(outname, hemi, idx))
                            os.rename("{0}.temp3.niml.dset".format(hemi),
                                      "{1}.{0}_cluster.niml.dset".format(outname, hemi))

                        for file in glob.glob("./*temp*"):
                            os.remove(file)
                # move clustered file into template folder
                os.rename("{1}.{0}_cluster.niml.dset".format(outname, hemi),
                          "./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                # copy mapping file to subjects' home SUMA directory
                if newmap:
                    shutil.move("{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi),
                                "{3}/{0}{1}.std141_to_native.{2}.niml.M2M".format(sub, suffix, hemi, sub_suma))
                # convert wang roi file to asc
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(
                        outname, hemi))
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname, hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                # convert  wang roi file to gii
                shell_cmd(
                    "ConvertDset -o_gii_b64 -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.gii".format(
                        outname, hemi))
                if not skipclust:
                    shell_cmd(
                        "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(
                            outname, hemi))
                    os.remove("./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    os.rename("./TEMPLATE_ROIS/{1}.{0}_cluster.temp.niml.dset".format(outname, hemi),
                              "./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset".format(outname, hemi))
                    # convert clustered file to gii
                    shell_cmd(
                        "ConvertDset -o_gii_b64 -input ./TEMPLATE_ROIS/{1}.{0}_cluster.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}_cluster.gii".format(
                            outname, hemi))
        ##KGS ROIs *********************************************************************
        if run_kgs == True:
            outname = 'KGS2016'
            print_wrap("running KGS2016:", indent=1)
            os.chdir(tmp_dir)
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
                    shell_cmd("mri_label2label --srcsubject fsaverage --trgsubject {2}{3} --regmethod surface --hemi {0} \
                        --srclabel {1}/{5}/{0}.MPM_{4}.label --trglabel ./{0}.{4}_TEMP.label"
                              .format(hemi, atlasdir, sub, suffix, roi, outname))

                    # convert to gifti
                    shell_cmd(
                        "mris_convert --label {0}.{1}_TEMP.label {1} {0}.white {0}.{1}_TEMP.gii"
                            .format(hemi, roi))

                    # convert to .niml.dset
                    shell_cmd(
                        "ConvertDset -o_niml_asc -input {0}.{1}_TEMP.gii -prefix {0}.{1}_TEMP.niml.dset"
                            .format(hemi, roi))

                    # isolate roi of interest
                    # do clustering, only consider cluster if they are 1 edge apart
                    shell_cmd("SurfClust -spec {2}{3}_{0}.spec -surf_A {0}.smoothwm.{4} -input {0}.{1}_TEMP.niml.dset 0 \
                        -rmm -1 -prefix {0}.{1}_TEMP2.niml.dset -out_fulllist -out_roidset"
                              .format(hemi, roi, sub, suffix, suma_format))

                    # create mask, pick only biggest cluster
                    shell_cmd(
                        "3dcalc -a {0}.{1}_TEMP2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)' -prefix {0}.{1}_TEMP3.niml.dset"
                            .format(hemi, roi))

                    # dilate mask
                    shell_cmd(
                        "ROIgrow -spec {2}{3}_{0}.spec -surf_A {0}.smoothwm.{4} -roi_labels {0}.{1}_TEMP3.niml.dset -lim 1 -prefix {0}.{1}_TEMP4"
                            .format(hemi, roi, sub, suffix, suma_format))

                    numnodes = shell_cmd("3dinfo -ni {0}.{1}_TEMP3.niml.dset".format(hemi, roi))
                    numnodes = int(numnodes.rstrip("\n"))
                    numnodes = numnodes - 1
                    shell_cmd(
                        "ConvertDset -o_niml_asc -i_1D -input {0}.{1}_TEMP4.1.1D -prefix {0}.{1}_TEMP4.niml.dset -pad_to_node {2} -node_index_1D {0}.{1}_TEMP4.1.1D[0]"
                            .format(hemi, roi, numnodes))

                    if idx == 1:
                        shell_cmd(
                            "3dcalc -a {0}.{1}_TEMP4.niml.dset -expr 'notzero(a)' -prefix {0}.{2}.niml.dset".format(
                                hemi, roi, outname))
                    else:
                        shell_cmd("3dcalc -a {0}.{1}_TEMP4.niml.dset -b {0}.{2}.niml.dset \
                            -expr '(b+notzero(a)*{3})*iszero(and(notzero(b),notzero(a)))' -prefix {0}.{1}_TEMP5.niml.dset".format(
                            hemi, roi, outname, idx))
                        shutil.move("{0}.{1}_TEMP5.niml.dset".format(hemi, roi),
                                    "{0}.{1}.niml.dset".format(hemi, outname))
                shutil.move("{0}.{1}.niml.dset".format(hemi, outname),
                            "./TEMPLATE_ROIS/{0}.{1}.niml.dset".format(hemi, outname))
                # convert from niml.dset to gii
                shell_cmd(
                    "ConvertDset -o_gii_b64 -input ./TEMPLATE_ROIS/{0}.{1}.niml.dset -prefix ./TEMPLATE_ROIS/{0}.{1}.gii".format(
                        hemi, outname))
                shell_cmd(
                    "ConvertDset -o_niml_asc -input ./TEMPLATE_ROIS/{1}.{0}.niml.dset -prefix ./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(
                        outname, hemi))
                os.remove("./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
                os.rename("./TEMPLATE_ROIS/{1}.{0}.temp.niml.dset".format(outname, hemi),
                          "./TEMPLATE_ROIS/{1}.{0}.niml.dset".format(outname, hemi))
        os.chdir(cur_dir)

        if os.path.isdir(out_dir):
            print_wrap("ROI output directory ""TEMPLATE_ROIS"" exists, adding '_new'", indent=1)
            shutil.move("{0}/TEMPLATE_ROIS".format(tmp_dir), "{0}_new".format(out_dir))
        else:
            shutil.move("{0}/TEMPLATE_ROIS".format(tmp_dir), "{0}".format(out_dir))
        if keep_temp is not True:
            # remove temporary directory
            shutil.rmtree(tmp_dir)
    # reset the subjects dir
    os.environ["SUBJECTS_DIR"] = old_subject_dir


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

    fft_dict = {
        "sig_complex": sig_complex, "sig_amp":sig_amp, "sig_phase":sig_phase,
        "noise_complex": noise_complex, "noise_amp": noise_amp, "noise_phase": noise_phase,
        "sig_z":sig_z, "sig_snr":sig_snr, "mean_cycle":mean_cycle,
        "spectrum":spectrum,"frequencies":frequencies }
    return fft_dict

def roi_get_data(surf_files, roi_type="wang", sub=False, do_scale=False, do_detrend=False,
                 use_regressors='standard', TR=2.0, roilabel=None, fs_dir=None, report_timing=False):
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
    TR : float, default 2.0
            Repetition Time
    roilabel : Roi label, default None
            Region of Interest label if required
    fs_dir : directory, default os.environ["SUBJECTS_DIR"]
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

    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

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
        assert s_2 in surf_files,  "File {0} does not have a matching file from the other hemisphere".format(s)

    task_list = [re.search('_task-(.+)_run', x)[1] for x in l_files]
    assert task_list == [re.search('_task-(.+)_run', x)[1] for x in r_files]
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
                   'benson': ["V1", "V2", "V3"]}
        # define roi files
        roi_file = [None, None]
        if roi_type == "wang":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fs_dir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            roi_label = roi_dic[roi_type] + ["V1", "V2", "V3"]
        elif roi_type == "benson":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fs_dir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            roi_label = roi_dic['benson']
        elif roi_type == "wang+benson":
            roi_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Wang2015.gii".format(fs_dir, sub)
            roi_file[1] = roi_file[0].replace("lh", "rh")
            eccen_file = [None, None]
            eccen_file[0] = "{0}/sub-{1}/TEMPLATE_ROIS/lh.Benson2014.all.gii".format(fs_dir, sub)
            eccen_file[1] = eccen_file[0].replace("lh", "rh")
            # define roilabel based on ring centers
            ring_incr = 0.25
            ring_size = .5
            ring_max = 6
            ring_min = 1
            ring_centers = np.arange(ring_min, ring_max, ring_incr)  # list of ring extents
            ring_extents = [(x - ring_size / 2, x + ring_size / 2) for x in ring_centers]
            roi_label = [y + "_{:0.2f}".format(x) for y in roi_dic['benson'] for x in ring_centers]
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
            roi_label = ["roi_{:02.0f}".format(x) for x in range(1, max_idx)]

    for h, hemi in enumerate(["L", "R"]):
        smooth_list = [x[x.find('fwhm') - 1] for x in data_files[h] if x.find('fwhm') >= 0]
        if len(smooth_list) == 0:
            smooth_msg = "unsmoothed"
        else:
            assert all(x == smooth_list[0] for x in smooth_list), "error, files have different levels of smooothing"
            smooth_msg = "{0}fwhm smoothed".format(smooth_list[0])
        data_n = None
        for r, run_file in enumerate(data_files[h]):
            try:
                cur_data = nl_surf.load_surf_data(run_file)
            except:
                print_wrap("Data file: {0} could not be opened".format(run_file))

            if do_scale:
                try:
                    cur_mean = np.mean(cur_data, axis=1, keepdims=True)
                    # repeat mean along second dim to match data
                    cur_mean = np.tile(cur_mean, [1, cur_data.shape[1]])

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
                    if r == 0:
                        print_wrap("using {0} confound regressors".format(len(df.columns)), indent=3)

                    new_data = nl_signal.clean(cur_data, detrend=True, standardize=False, confounds=df.values,
                                               low_pass=None, high_pass=None, t_r=TR, ensure_finite=False)
                    cur_data = np.transpose(new_data)
                except:
                    print_wrap("Data file: {0}, detrending failure".format(run_file.split('/')[-1]))

            if data_n:
                assert data_n == cur_data.shape[
                    0], "two runs from {0}H have different number of surface vertices".format(hemi)
            else:
                data_n = cur_data.shape[0]

            if r == 0:
                run_data = cur_data
            else:
                run_data = np.dstack((run_data, cur_data))

        if roi_type == "whole":
            vox_count = np.ones((run_data.shape[0], 1))
            cur_label = ["v-{:06d}".format(x + 1) for x in range(run_data.shape[0])]
        else:
            # uses surface module of nilearn to import data in .gii format
            roi_data = nl_surf.load_surf_data(roi_file[h])
            # Benson should just use ROIs
            if roi_type == "benson":
                roi_data = roi_data[:, 2]
                roi_n = roi_data.shape[0]
            # wang+benson-specific code begins here
            elif roi_type == "wang+benson":
                eccen_data = nl_surf.load_surf_data(eccen_file[h])
                # select eccen data from Benson
                eccen_data = eccen_data[:, 1]
                assert eccen_data.shape[0] == roi_data.shape[0], "ROIs and Template Eccen have different number of surface vertices"
                ring_data = np.zeros_like(roi_data)
                for r, evc in enumerate(["V1", "V2", "V3"]):
                    # find early visual cortex rois in wang rois
                    roi_set = [i + 1 for i, s in enumerate(roi_dic['wang']) if evc+'d' in s or evc+'v' in s]
                    evc_idx = np.any(np.array([ roi_data == x for x in roi_set ]),axis=0)
                    # define indices based on ring extents
                    for e, extent in enumerate(ring_extents):
                        # get indexes that are populated in both i.e. between lower and higher extent
                        eccen_idx = np.all(np.vstack(((eccen_data > extent[0]), (eccen_data < extent[1]), evc_idx)), axis=0)
                        idx_val = e + (r * len(ring_centers))
                        ring_data[eccen_idx] = idx_val + 1
                # now set ring values as new roi values
                roi_data = ring_data
            assert run_data.shape[0] == roi_data.shape[0], "Data and ROI have different number of surface vertices"
            run_t = np.array([np.mean(run_data[roi_data == x], axis=0) for x in np.unique(roi_data) if x > 0])
            vox_count = np.reshape(np.array([np.count_nonzero(roi_data == x) for x in np.unique(roi_data) if x > 0]), (-1, 1))

            if roi_type == "wang":
                # combine dorsal and ventral
                for r, evc in enumerate(["V1", "V2", "V3"]):
                    # find early visual cortex rois in wang rois
                    roi_set = [i + 1 for i, s in enumerate(roi_dic['wang']) if evc + 'd' in s or evc + 'v' in s]
                    evc_idx = np.any(np.array([roi_data == x for x in roi_set]), axis=0)
                    run_t = np.vstack((run_t, np.mean(run_data[evc_idx], axis=0, keepdims=True)))
                    vox_count =  np.vstack((vox_count, np.count_nonzero(evc_idx)))

            run_data = run_t
            cur_label = roi_label
            assert run_data.shape[0] == len(cur_label), "ROI data and output data are mismatched"

        if "L" in hemi:
            left_data = run_data
            left_vox = vox_count
            left_label = ["{0}-{1}".format(label, hemi) for label in cur_label]
        else:
            right_data = run_data
            right_vox = vox_count
            right_label = ["{0}-{1}".format(label, hemi) for label in cur_label]

            if roi_type not in "whole":
                # only do bilateral data for ROIs
                bl_data = np.divide(left_data + right_data, 2)
                bl_vox = [sum(x) for x in zip(left_vox, right_vox)]

                all_data = np.vstack((left_data,right_data, bl_data))
                all_vox = np.vstack((left_vox, right_vox, bl_vox))
                cur_label = left_label + right_label + ["{0}-BL".format(label) for label in cur_label]
            else:
                # combine left and right
                all_data = np.vstack((left_data, right_data))
                all_vox = np.vstack((left_vox, right_vox))
                cur_label = left_label + right_label

    all_data = all_data.transpose(1, 0, 2) # move time into first dimension
    if report_timing:
        elapsed = time.time() - t
        print_wrap("roi_get_data complete, took {:3.2f} seconds".format(elapsed), indent=1)

    return all_data, all_vox, cur_label, task_list

def hotelling_t2(dset1=np.zeros((15,3,4), dtype=np.complex), dset2=np.zeros((1, 1), dtype=np.complex), test_type="hot_dep", test_mode=False,report_timing=True):
    t = time.time()
    assert test_type != "tcirc", "tcirc not currently supported"
    if not dset1.any():
        dset1 = simulate_complex(dset1.shape , a_mu=1.1, ph_mu=np.pi, assign_nans=True )
        dset2 = simulate_complex(dset1.shape , a_mu=1, ph_mu=np.pi, assign_nans=True )
    else:
        assert np.iscomplexobj(dset1), "non-complex values given as 'dset1' input"
        assert np.iscomplexobj(dset2), "non-complex values given as 'dset2' input"

    # compare against zero?
    if dset2.size == 1:
        num_cond = 1
        dset2 = np.full_like(dset1, dset2, dtype=np.complex)
        assert test_type in ["hot_dep", "tcirc"], "second dset is single vector, two full datasets required for independent test"
    else:
        if test_type in ["hot_dep", "tcirc"]:
            assert dset2.shape == dset1.shape, "'dset2' input must either be a single complex value or a nparray same shape as 'dset1'"
        else:
            assert dset2.shape[1:] == dset1.shape[1:], "'dset2' input must have same shape as 'dset1' on all dimensions but the first"
        num_cond = 2

    out_dfs = np.full((2,) + dset1.shape[1:], np.nan)
    out_fapprox = np.full(dset1.shape[1:], np.nan)
    out_p = np.full(dset1.shape[1:], np.nan)

    # PREP R
    icsnp = importr('ICSNP')
    pandas2ri.activate()

    if test_type.lower() in ["hot_dep"]:
        # mask out NaNs
        is_nan = (np.isnan(dset1) + np.isnan(dset2)) > 0
        # mask real and imaginary separately
        dset1_ma = np.ma.masked_where(is_nan, dset1)
        dset2_ma = np.ma.masked_where(is_nan, dset2)

        # determine number of trials
        p = np.float(2.0)  # number of variables
        df1 = p  # numerator degrees of freedom.
        # subtract conditions
        combo_data = np.subtract(dset1_ma, dset2_ma)
        real_data = np.real(combo_data)
        imag_data = np.imag(combo_data)
        test_mu = [0.0, 0.0]
        for idx in np.ndindex(dset1_ma.shape[1:]):
            cur_idx = (slice(None),) + idx 

            # USE R
            frame_data = pd.DataFrame(np.stack((real_data[cur_idx],imag_data[cur_idx]), axis=1))
            os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
            hot_r = icsnp.HotellingsT2(frame_data.dropna(), mu=np.asarray(test_mu))
            r_tsqrdf = np.array(hot_r.rx2('statistic'))
            r_dfs = np.array(hot_r.rx2('parameter'))
            r_p = np.array(hot_r.rx2('p.value'))

            # make sure that one-sample test was done
            assert "Hotelling's one sample T2-test" in hot_r.rx2('method')

            if test_mode:
                # COMPUTE VALUES
                M = sum(is_nan[cur_idx] == 0)
                df2 = M - p  # denominator degrees of freedom.
                samp_cov_mat = np.ma.cov(real_data[cur_idx], imag_data[cur_idx])

                # Eqn. 2 in Sec. 5.3 of Anderson (1984), multiply by inverse of fraction below::
                t_crit = (((M - 1) * p) / df2) * scp.stats.f.ppf(1 - 0.05, df1, df2)

                # Eqn. 2 of Sec. 5.1 of Anderson (1984):
                inv_cov_mat = np.linalg.inv(samp_cov_mat)
                samp_mu = np.array([np.ma.mean(real_data[cur_idx],axis=0), np.ma.mean(imag_data[cur_idx],axis=0)]) 
                c_tsqrd = np.float(
                    np.matmul(np.matmul(M * np.reshape(samp_mu - test_mu, (1, 2)), inv_cov_mat), samp_mu - test_mu))

                # F approximation, inverse of fraction above
                c_tsqrdf = ( ( df2 / ((M - 1) * p) ) ) * c_tsqrd
                # use scipys F cumulative distribution function.
                c_p = 1.0 - scp.stats.f.cdf(c_tsqrdf, df1, df2)  

                assert bool(np.isclose(r_p,c_p, rtol=1e-09)) and bool(np.isclose(c_tsqrdf, r_tsqrdf, rtol=1e-09)) and r_dfs[0] == df1 and r_dfs[1] == df2, "test mode failed: computed and r-values are different"

            # assign output data
            out_dfs[(slice(None),) + idx] = r_dfs
            out_fapprox[idx] = r_tsqrdf
            out_p[idx] = r_p

    elif test_type.lower() in ["hot_ind"]:
        # mask out NaNs
        nan_dset1 = np.isnan(dset1) > 0
        nan_dset2 = np.isnan(dset2) > 0
        # mask real and imaginary separately
        dset1_ma = np.ma.masked_where(nan_dset1, dset1)
        dset2_ma = np.ma.masked_where(nan_dset2, dset2)
        dset1_real = np.real(dset1_ma); dset1_imag = np.imag(dset1_ma)
        dset2_real = np.real(dset2_ma); dset2_imag = np.imag(dset2_ma)

        p = np.float(2.0)  # number of variables
        df1 = p  # numerator degrees of freedom
        test_mu = [0.0, 0.0]
        for idx in np.ndindex(dset1_ma.shape[1:]):
            cur_idx = (slice(None),) + idx

            # USE R
            frame_data1 = pd.DataFrame(np.stack((dset1_real[cur_idx],dset1_imag[cur_idx]), axis=1))
            frame_data2 = pd.DataFrame(np.stack((dset2_real[cur_idx],dset2_imag[cur_idx]), axis=1))
            os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
            hot_r = icsnp.HotellingsT2(frame_data1.dropna(), frame_data2.dropna(), mu=np.asarray(test_mu), test='f')
            r_tsqrdf = np.array(hot_r.rx2('statistic'))
            r_dfs = np.array(hot_r.rx2('parameter'))
            r_p = np.array(hot_r.rx2('p.value'))

            # make sure that two-sample test was done
            assert "Hotelling's two sample T2-test" in hot_r.rx2('method')

            if test_mode:
                # COMPUTE VALUES
                M_1 = sum(nan_dset1[cur_idx] == 0)
                M_2 = sum(nan_dset2[cur_idx] == 0)
                df2 = M_1 + M_2 - 1 - p # denominator degrees of freedom.

                # Eqn. 18 in Sec. 5.17 of Anderson (1984), multiply by inverse of fraction below:
                t_crit = ( ( ( M_1 + M_2 - 2 ) * p ) / df2 ) * scp.stats.f.ppf(1 - 0.05, df1, df2)

                scov_1 = np.ma.cov(dset1_real[cur_idx], dset1_imag[cur_idx])
                scov_2 = np.ma.cov(dset2_real[cur_idx], dset2_imag[cur_idx])

                S_pool = ( (M_1 - 1) * scov_1 + (M_2 - 1) * scov_2 ) / (M_1 + M_2 - 2)
                diff_S_pool = S_pool * (1 / M_1 + 1 / M_2)

                diff_mu = np.array([np.ma.mean(dset1_real[cur_idx], axis=0), np.ma.mean(dset1_imag[cur_idx],axis=0)]) - np.array([np.ma.mean(dset2_real[cur_idx],axis=0), np.ma.mean(dset2_imag[cur_idx],axis=0)])

                inv_cov_mat = np.linalg.inv(diff_S_pool)

                c_tsqrd = np.float(
                    np.matmul(np.matmul(np.reshape(diff_mu - test_mu, (1, 2)), inv_cov_mat), diff_mu - test_mu))
                c_tsqrdf = c_tsqrd * ( df2 / (p * (M_1 + M_2 - 2)) )
                c_p = 1.0 - scp.stats.f.cdf(c_tsqrdf, df1, df2)

                assert bool(np.isclose(r_p,c_p, rtol=1e-09)) and bool(np.isclose(c_tsqrdf, r_tsqrdf, rtol=1e-09)) and r_dfs[0] == df1 and r_dfs[1] == df2, "test mode failed: computed and r-values are different"

                # # Eqn. 16 in Sec. 5.3.4 of Anderson (1984)
                # FOR SOME REASON THIS APPROACH PRODUCES OUTRAGEOUSLY LOW P-VALUES, MUST BE SOME BUG!
                # diff_S_pool2 = ( 1 / ( M_1 + M_2 - 2 ) ) * (scov_1 + scov_2)
                # # Eqn. 17 in Sec. 5.3.4 of Anderson (1984)
                # tsqrd2 = np.float(
                #     np.matmul(np.matmul( ( ( M_1 * M_2 ) / ( M_1 + M_2) ) * np.reshape(diff_mu - test_mu, (1, 2)), np.linalg.inv(diff_S_pool2)), diff_mu - test_mu))
                # tsqrdf2 = tsqrd2 * ( df2 / (p * (M_1 + M_2 - 2)) )
                # p_val2 = 1.0 - scp.stats.f.cdf(tsqrdf2, df1, df2)

            # assign output data
            out_dfs[(slice(None),) + idx] = r_dfs
            out_fapprox[idx] = r_tsqrdf
            out_p[idx] = r_p
    #else:
        # # note, if two experiment conditions are to be compared, we assume that the number of samples is equal
        # df2 = num_cond * (2.0 * M - p)  # denominator degrees of freedom.
        #
        # # compute estimate of sample mean(s) from V & M 1991
        # samp_mu = np.mean(in_vals, 0.0)
        #
        # # compute estimate of population variance, based on individual estimates
        # v_indiv = 1 / df2 * (np.sum(np.square(np.abs(in_vals[:, 0] - samp_mu[0])))
        #                      + np.sum(np.square(np.abs(in_vals[:, 1] - samp_mu[1]))))
        #
        # if num_cond == 1:
        #     # comparing against zero
        #     v_group = M / p * np.square(np.abs(samp_mu[0] - samp_mu[1]))
        #     # note, distinct multiplication factor
        #     mult_factor = 1 / M
        # else:
        #     # comparing two conditions
        #     v_group = (np.square(M)) / (2 * (M * 2)) * np.square(np.abs(samp_mu[0] - samp_mu[1]))
        #     # note, distinct multiplication factor
        #     mult_factor = (M * 2) / (np.square(M))
        #
        # # Find critical F for corresponding alpha level drawn from F-distribution F(2,2M-2)
        # # Use scipys percent point function (inverse of `cdf`) for f
        # # multiply by inverse of multiplication factor to get critical t_circ
        # t_crit = scp.stats.f.ppf(1 - alpha, df1, df2) * (1 / mult_factor)
        # # compute the tcirc-statistic
        # tsqrd = (v_group / v_indiv) * mult_factor;
        # # M x T2Circ ( or (M1xM2/M1+M2)xT2circ with 2 conditions)
        # # is distributed according to F(2,2M-2)
        # # use scipys F probability density function
        # p_val = 1 - scp.stats.f.cdf(tsqrd * (1 / mult_factor), df1, df2);
    
    return (out_fapprox, out_p, out_dfs)

def simulate_complex(test_shape, a_mu=None, ph_mu=None, assign_nans=False):
    # phase
    if ph_mu:
        # use specified mean
        test_ph = np.full(test_shape,ph_mu)
    else:
        # use random values between -pi and pi
        test_ph = np.repeat(np.random.random_sample((1,) + test_shape[1:]) * 2 * np.pi - np.pi, test_shape[0], 0)
    # add 1/4 pi noise around mean
    test_ph = test_ph + (np.random.random_sample(test_shape) * 1 / 4 * np.pi) - 1 / 8 * np.pi
    # amplitude
    if a_mu:
        test_amp = np.full(test_shape, a_mu)
    else:
        # use random values between 0 and 1
        test_amp = np.repeat(np.random.random_sample((1,) + test_shape[1:]), test_shape[0], 0)
    # add 1/4 noise around mean
    test_amp = test_amp * 7 / 8 + test_amp * (np.random.random_sample(test_shape) * 1 / 4)
    complex_test = test_amp * np.exp(1j * test_ph)
    real_test = np.real(complex_test)
    imag_test = np.imag(complex_test)
    if assign_nans:
        for idx in np.ndindex(test_shape[1:]):
            # assign one NaN per 'condition' to the real and imaginary values
            real_nan = np.random.randint(0, test_shape[0], 1)
            real_idx = (real_nan,) + idx
            imag_nan = np.random.randint(0, test_shape[0], 1)
            imag_idx = (imag_nan,) + idx
            real_test[real_idx] = np.nan
            imag_test[imag_idx] = np.nan
    complex_test = imag_test + 1j * real_test
    return complex_test

def vector_projection(complex_in=np.zeros((15,3,4), dtype="complex128"), test_fig=False):
    # project_amp, project_err, project_t, project_p, project_real, project_imag = vector_projection(complex_in, test_fig=False)
    if not complex_in.any():
        test_shape = complex_in.shape
        complex_in = simulate_complex(test_shape)
        test_fig = True
    else:
        assert np.iscomplexobj(complex_in), "input values must be complex"
    real_in = np.real(complex_in)
    imag_in = np.imag(complex_in)
    assert real_in.shape == imag_in.shape, "real and imaginary must have same shape"

    is_nan = (np.isnan(real_in) + np.isnan(imag_in)) > 0
    # mask real and imaginary separately
    real_ma = np.ma.masked_where(is_nan, real_in)
    imag_ma = np.ma.masked_where(is_nan, imag_in)
    c_data = np.ma.stack((real_ma, imag_ma), -1)
    c_data = np.moveaxis(c_data, -1, 0)
    c_mean = np.repeat(np.ma.mean(c_data, 1, keepdims=True), c_data.shape[1], 1)

    len_c = np.divide(np.sum(np.multiply(c_data, c_mean), 0), np.sum(np.multiply(c_mean, c_mean), 0))
    c_out = len_c * c_mean

    project_amp = np.multiply(np.sqrt(np.square(c_out[0, :]) + np.square(c_out[1, :])), np.sign(len_c))

    # compute t-values
    project_t, project_p = stats.ttest_1samp(project_amp, 0.0)
    project_p = project_p / 2  # one-tailed, so divide by p-vals by two

    # and standard error
    project_err = np.divide(np.nanstd(project_amp, axis=0), np.sqrt(sum(is_nan==0)))

    # fill nans back in
    project_amp = np.ma.filled(project_amp)
    project_real = np.ma.filled(c_out[0, :], np.nan)
    project_imag = np.ma.filled(c_out[1, :], np.nan)

    out_mean = np.ma.mean(c_out, 1, keepdims=False)
    assert np.all(
        np.asarray([np.isclose(np.ma.max(c_mean, axis=1)[x], out_mean[x, :], atol=1e-09) for x in [0, 1]])
    ), "mean of projected should be the same as mean of unprojected"

    if test_fig:
        for idx in np.ndindex(c_data.shape[2:]):
            cur_idx = (slice(None),) + (slice(None),) + idx
            fig_in = c_data[cur_idx]
            fig_out = c_out[cur_idx]
            fig_mean = c_mean[cur_idx]
            fig_mean = fig_mean[:,0] # same value for all subs
            ax_max = math.ceil(np.max(np.abs(fig_in) * 5)) / 5
            plt.figure(figsize=(5, 5))
            plt.plot([0, fig_mean[0]], [0, fig_mean[1]], '-', c="k")
            plt.plot(np.zeros((2, 1)), [-ax_max, ax_max], '-', c="k")
            plt.plot([-ax_max, ax_max], np.zeros((2, 1)), '-', c="k")
            plt.plot([fig_in[0, :], fig_out[0, :]], [fig_in[1, :], fig_out[1, :]], '-',c="gray")
            plt.plot(fig_in[0, :], fig_in[1, :], 'o', c="g", markerfacecolor="none", markersize = 10)
            plt.plot(fig_out[0, :], fig_out[1, :], 'o', c="b", markerfacecolor="none", markersize = 10)
            plt.plot(fig_mean[0], fig_mean[1], 'o', c="r", markerfacecolor="none", markersize=20)
            axes = plt.gca()
            axes.set_xlim([-ax_max, ax_max])
            axes.set_ylim([-ax_max, ax_max])
            axes.set_aspect('equal', 'box')
            plt.show()
    return project_amp, project_err, project_t, project_p, project_real, project_imag

def fit_error_ellipse(complex_in=np.zeros((15,100000), dtype="complex128"), ellipse_type='SEM', make_plot=True, return_rad=True):
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

    if not complex_in.any():
        test_shape = complex_in.shape
        complex_in = simulate_complex(test_shape)
        make_plot = False
    else:
        assert np.iscomplexobj(complex_in), "input values must be complex"

    # convert return_rad to an integer for indexing purposes later
    return_rad = int(return_rad)
    conv_factor = np.array([180 / np.pi, 1])
    unwrap_factor = np.array([360, 2 * np.pi])

    # mask out NaNs
    is_nan = np.isnan(complex_in)
    # mask real and imaginary separately
    complex_ma = np.ma.masked_where(is_nan, complex_in)
    # now loop over data sets
    amp_out = np.full((3,) + complex_in.shape[1:], np.nan)
    phase_out = np.full((3,) + complex_in.shape[1:], np.nan)
    snr_out = np.full(complex_in.shape[1:], np.nan)
    count = 0
    print_proc = -1
    for idx in np.ndindex(complex_ma.shape[1:]):
        count += 1
        cur_idx = (slice(None),) + idx
        n = sum(is_nan[cur_idx]==0)
        real_data = np.real(complex_ma[cur_idx])
        imag_data = np.imag(complex_ma[cur_idx])
        # mean and covariance
        mean_data = np.array([np.ma.mean(real_data), np.ma.mean(imag_data)])
        try:
            samp_covmat = np.ma.cov(np.ma.array([real_data, imag_data]))
            # calc eigenvalues, eigenvectors
            eigenval, eigenvec = np.linalg.eigh(samp_covmat)
            # sort the eigenvector by the eigenvalues
            ordered_eig = np.sort(eigenval)
            idx_eig = np.argsort(eigenval)
            smaller_eigenvec = eigenvec[:, idx_eig[0]]
            larger_eigenvec = eigenvec[:, idx_eig[1]]
            smaller_eigenval = ordered_eig[0]
            larger_eigenval = ordered_eig[1]
            phi = np.arctan2(larger_eigenvec[1], larger_eigenvec[0])
            # this angle is between -pi & pi, shift to 0 and 2pi
            if phi < 0:
                phi = phi + 2 * np.pi
        except:
            print('Unable to run eigen value decomposition. Probably data have only 1 sample')
            return None
        theta_grid = np.linspace(0, 2 * np.pi, num=100)
        # what type of ellipse?
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
                crit_val = float(ellipse_type[:-3]) / 100
            except:
                print('ellipse_type incorrectly formatted, please see docstring')
                return None
            assert crit_val < 1.0 and crit_val > 0.0, 'ellipse_type CI range must be between 0 & 100'
            t0_sqrd = ((n - 1) * 2) / (n * (n - 2)) * stats.f.ppf(crit_val, 2, n - 2)
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
        error_ellipse = np.add(error_ellipse, mean_data)

        # find vector length of each point on ellipse
        norms = np.linalg.norm(error_ellipse, axis=1)
        amp_bounds = np.array([np.min(norms), np.max(norms)])
        amp_idx = np.array([np.argmin(norms), np.argmax(norms)])
        amp_mean = np.linalg.norm(mean_data)

        # calculate phase angles &
        phase_angles = np.arctan2(error_ellipse[:, 1], error_ellipse[:, 0])
        phase_angles[phase_angles < 0] = phase_angles[phase_angles < 0] + 2 * np.pi

        # if ellipse overlaps with origin, defined by whether phase angles in all 4 quadrants
        quad = []
        quad.append( len(phase_angles[(phase_angles > 0) & (phase_angles < np.pi / 2)]) > 0 )
        quad.append( len(phase_angles[(phase_angles > np.pi / 2) & (phase_angles < np.pi)]) > 0 )
        quad.append( len(phase_angles[(phase_angles > np.pi / 2) & (phase_angles < 3 * np.pi / 2)]) > 0 )
        quad.append( len(phase_angles[(phase_angles > 3 * np.pi / 2) & (phase_angles < 2 * np.pi)]) > 0 )

        phase_mean = np.arctan2(mean_data[1], mean_data[0]) * conv_factor[return_rad]
        # invert phase mean, so that positive values indicate rightward shift,
        # and negative indicate leftward shift, relative to the cosine
        phase_mean = phase_mean * -1
        # unwrap negative phases
        if phase_mean < 0:
            phase_mean = phase_mean + unwrap_factor[return_rad]

        if all(quad):
            amp_bounds = np.array([0, amp_bounds[1]])
            phase_bounds = np.array([0, unwrap_factor[return_rad]])
        else:
            if quad[3]:
                # wrap to -pi - pi
                phase_angles = (phase_angles + np.pi) % (2 * np.pi) - np.pi
                diff_phase = np.max(phase_angles) - np.min(phase_angles)
                phase_idx = np.array([np.argmin(phase_angles), np.argmax(phase_angles)])
                phase_angles[phase_angles < 0] = phase_angles[phase_angles < 0] + 2 * np.pi
            else:
                diff_phase = np.max(phase_angles) - np.min(phase_angles)
                phase_idx = np.array([np.argmin(phase_angles), np.argmax(phase_angles)])

            # invert phase bounds, and convert to 0 to 2*pi or 0 to 360,
            phase_bounds = np.array([ phase_angles[x] * conv_factor[return_rad] for x in phase_idx ]) * -1
            phase_bounds[phase_bounds < 0] = phase_bounds[phase_bounds < 0] + unwrap_factor[return_rad]
            phase_bounds = np.sort(phase_bounds)

        phase_diff = np.array([np.absolute(phase_bounds[0] - phase_mean),
                               np.absolute(phase_bounds[1] - phase_mean)], ndmin=2)

        # unwrap phase diff for any ellipse that overlaps with positive x axis
        phase_diff[phase_diff > unwrap_factor[return_rad] / 2] = unwrap_factor[return_rad] - phase_diff[phase_diff > unwrap_factor[return_rad] / 2]
        amp_diff = np.array([amp_mean - amp_bounds[0], amp_bounds[1] - amp_mean], ndmin=2)
        z_snr = amp_mean / np.mean(np.array([amp_mean - amp_bounds[0], amp_bounds[1] - amp_mean]))

        amp_out[(slice(None),) + idx] = np.insert(amp_diff.tolist(),0, amp_mean)
        phase_out[(slice(None),) + idx] = np.insert(phase_diff.tolist(), 0, phase_mean)
        snr_out[idx] = z_snr

        # Data plot
        if make_plot:
            # Below makes 2 subplots
            plt.figure(figsize=(9, 9))
            font = {'size': 16, 'color': 'k', 'weight': 'light'}
            # Figure 1 - eigen vector & SEM ellipse
            plt.subplot(1, 2, 1)
            plt.plot(real_data, imag_data, 'ko', markerfacecolor='k')
            plt.plot([0, mean_data[0]], [0, mean_data[1]], linestyle='solid', color='k', linewidth=1)
            # plot ellipse
            plt.plot(error_ellipse[:, 0], error_ellipse[:, 1], 'b-', linewidth=1, label=ellipse_type + ' ellipse')
            # plot smaller eigen vec
            small_eigen_mean = [np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[0]) + mean_data[0],
                                np.multiply(np.sqrt(smaller_eigenval), smaller_eigenvec[1]) + mean_data[1]]
            plt.plot([mean_data[0], small_eigen_mean[0]], [mean_data[1], small_eigen_mean[1]], 'g-', linewidth=1, label='smaller eigen vec')
            # plot larger eigen vec
            large_eigen_mean = [np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[0]) + mean_data[0],
                                np.multiply(np.sqrt(larger_eigenval), larger_eigenvec[1]) + mean_data[1]]
            plt.plot([mean_data[0], large_eigen_mean[0]], [mean_data[1], large_eigen_mean[1]], 'm-',
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
            plt.plot([0, error_ellipse[amp_idx[0], 0]], [0, error_ellipse[amp_idx[0], 1]],
                     color='r', linestyle='--')
            plt.plot([0, error_ellipse[amp_idx[1], 0]], [0, error_ellipse[amp_idx[1], 1]],
                     color='r', label='amp bounds: {:2.2f}-{:2.2f}'.format(amp_bounds[0],amp_bounds[1]), linestyle='--')
            font['color'] = 'r'

            # plot phase bounds
            plt.plot([0, error_ellipse[phase_idx[0], 0]], [0, error_ellipse[phase_idx[0], 1]],
                     color='b', linewidth=1)
            plt.plot([0, error_ellipse[phase_idx[1], 0]], [0, error_ellipse[phase_idx[1], 1]],
                     color='b', linewidth=1, label='phase bounds: {:2.2f}-{:2.2f}'.format(phase_bounds[0],phase_bounds[1]))
            font['color'] = 'b'

            # plot mean vector
            plt.plot([0, mean_data[0]], [0, mean_data[1]], color='k', linewidth=1, label='mean ampl.')
            font['color'] = 'k'
            plt.text(mean_data[0], mean_data[1], "({:2.2f},{:2.2f})".format(amp_mean,phase_mean), fontdict=font)

            # plot major/minor axis
            plt.plot([mean_data[0], a * larger_eigenvec[0] + mean_data[0]],
                     [mean_data[1], a * larger_eigenvec[1] + mean_data[1]],
                     color='m', linewidth=1)
            plt.plot([mean_data[0], -a * larger_eigenvec[0] + mean_data[0]],
                     [mean_data[1], -a * larger_eigenvec[1] + mean_data[1]],
                     color='m', linewidth=1)
            plt.plot([mean_data[0], b * smaller_eigenvec[0] + mean_data[0]],
                     [mean_data[1], b * smaller_eigenvec[1] + mean_data[1]],
                     color='g', linewidth=1)
            plt.plot([mean_data[0], -b * smaller_eigenvec[0] + mean_data[0]],
                     [mean_data[1], -b * smaller_eigenvec[1] + mean_data[1]],
                     color='g', linewidth=1)
            plt.axhline(color='k', linewidth=1)
            plt.axvline(color='k', linewidth=1)
            plt.legend(loc=3, frameon=False)
            plt.axis('equal')
            plt.show()
    return amp_out, phase_out, snr_out


def subset_rois(in_file, roi_selection=["evc"], out_file=None, roi_labels="wang", fs_dir=None):
    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")
    # roi_labels = ["V1d", "V1v", "V2d", "V2v", "V3A", "V3B", "V3d", "LO1", "TO1", "V3v", "VO1", "hV4"]
    if roi_labels == "wang":
        label_path = "{0}/ROI_TEMPLATES/Wang2015/ROIfiles_Labeling.txt".format(fs_dir)
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
        tmp_dir = tmp_dir = make_temp_dir()
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

def subject_analysis(exp_folder, fs_dir=None, subjects='All', roi_type='wang+benson',
                 data_spec={}, in_format=".gii", tasks='All', offset=0, pre_tr=0, report_timing=False, overwrite=False):
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
    fs_dir : string/os file location, default to SUBJECTS_DIR
        Freesurfer folder directory
    subjects : string/list, Default 'All'
        Options:
            - 'All' - identifies and runs all subjects
            - ['sub-0001','sub-0002'] - runs only a list of subjects
    pre_tr : int, default 0
        input for RoiSurfData - please see for more info
    roi_type : str, default 'wang+benson'
        other options as per RoiSurfData function
    data_spec: dict, list or str
        if dict, the data are stored in bids format, and dict contains
        {'space':'suma_native', 'task': None, 'session':'01', 'detrend': False, smoothing: 0}
        note that default parameters are given above.
        if list or str, non-bids format is assume,
        and data_spec is a str or list of strs that must be present in input files
    in_format : str, Default "".nii.gz"
        file type of input data
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
    start_t = time.time()

    if not fs_dir:
        assert os.getenv("SUBJECTS_DIR"), "fs_dir not provided and 'SUBJECTS_DIR' environment variable not set"
        fs_dir = os.getenv("SUBJECTS_DIR")

    if type(subjects) is str:
        # make it a list
        subjects = [subjects]
    if 'all' in [x.lower() for x in subjects]:
        subjects = [x for x in os.listdir(exp_folder) if 'sub' in x and os.path.isdir(os.path.join(exp_folder, x))]

    # if 'all' is in list, do all tasks
    if tasks == None:
        tasks = dict.fromkeys(["no task"], [])
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
                task_list += get_data_files("{0}/{1}".format(exp_folder, sub), type=in_format, spec=data_spec)
            task_list = [re.findall('task-\w+_', x)[0][5:-1] for x in task_list]
            task_list = list(set(task_list))
        # make_dict and assign pre_tr and offset to dict
        tasks = dict.fromkeys(task_list, [])

    first_loop = True
    out_dict = {}
    for s, sub in enumerate(subjects):
        # produce list of files
        surf_files, cur_spec = get_file_list("{0}/{1}".format(exp_folder, sub), type=in_format, spec=data_spec)

        if len(surf_files) > 0:
            # set up cache
            cache_dir = "{0}/__cache__".format(exp_folder)
            memory = Memory(cache_dir, verbose=0, mmap_mode='r')
            roi_get_cached = memory.cache(roi_get_data, ignore=['report_timing'])
            if overwrite:
                roi_get_cached.clear(warn=False)

            # run roi_get_data
            cur_t = time.time()
            sub_data, sub_vox, out_label, task_list = roi_get_cached(surf_files, roi_type=roi_type, fs_dir=fs_dir)
            print_wrap(time.time()- cur_t)
            for t, task in enumerate(tasks.keys()):
                # dictionary takes precedence
                if "offset" in tasks[task].keys():
                    offset = tasks[task]["offset"]
                if "pre_tr" in tasks[task].keys():
                    pre_tr = tasks[task]["pre_tr"]

                if task is "no task":
                    task_idx = range(len(task_list))
                    print_wrap("analyzing {0} without considering task, pre-tr: {1}, offset: {2}".format(sub, pre_tr, offset), indent=1)
                else:
                    task_idx = [idx for idx, name in enumerate(task_list) if task == name]
                    if task_idx:
                        print_wrap("analyzing {0} on task {1}, pre-tr: {2}, offset: {3}".format(sub, task, pre_tr, offset), indent=1)

                if task_idx:
                    out_obj = roiobject(cur_data=sub_data[pre_tr:,:,task_idx], cur_object=None, roi_names=out_label, tr=2.0, stim_freq=10, nharm=5, num_vox=sub_vox, is_time_series=True, offset=offset)
                    sub_dict = {"sig_complex": out_obj.fft()["sig_complex"], "mean_cycle": out_obj.fft()["mean_cycle"]}

                    if first_loop:
                        out_spec = cur_spec
                        out_names = out_obj.roi_names
                        first_loop = False
                    else:
                        assert out_names == out_obj.roi_names, "roi names do not match across subjects"
                        assert {**out_spec} == {**cur_spec}, "specs do not match across subjects"

                    if task in out_dict:
                        out_dict[task]["data"].append(sub_dict)
                        out_dict[task]["num_vox"].append(sub_vox)
                    else:
                        out_dict[task] = {"data": [sub_dict], "num_vox": [sub_vox], "roi_names": out_names, **out_spec}

    if report_timing:
        elapsed = time.time() - start_t
        print_wrap("subject_analysis complete, took {:02.2f} minutes".format(elapsed/60))
    return out_dict

def group_compare(exp_dir, tasks, fs_dir=None, subjects='All', data_spec={}, roi_type="wang", harmonic_list=[1], test_type = None, report_timing=True, overwrite=False):

    start_time = time.time()

    print_wrap("running group {0} analysis ...".format(roi_type))
    group_dictionary = {}
    roi_n = []
    make_plot = False

    if type(exp_dir) is str:
        # make it a list
        exp_dir = [exp_dir]
        if not test_type:
            test_type = "hot_dep"
    else:
        assert type(exp_dir), "exp_dir must be a string or a list"
        # if test_type not given, assume independent test
        if not test_type:
            test_type = "hot_ind"

    comp_dict = {}
    for  cur_exp in exp_dir:
        out_dict = subject_analysis(exp_folder=exp_dir, fs_dir=fs_dir, tasks=tasks, roi_type=roi_type, data_spec=data_spec, in_format=".gii", overwrite=overwrite)

def group_analyze(exp_dir, tasks, fs_dir=None, subjects='All', data_spec={}, roi_type="whole", harmonic_list=["1"], output='all', ellipse_type='SEM', return_rad=True, report_timing=True, overwrite=False):
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

    print_wrap("running group {0} analysis ...".format(roi_type))
    group_dictionary = {}
    roi_n = []
    make_plot = False

    if any(isinstance(x, int) for x in harmonic_list):
        harmonic_list = [str(x) for x in harmonic_list]
    assert all(float(x) > 0 for x in harmonic_list), "harmonic_list must be a list of named harmonics, not zero-based indices"

    out_dict = subject_analysis(exp_folder=exp_dir, fs_dir=fs_dir, tasks=tasks, roi_type=roi_type, data_spec=data_spec, in_format=".gii", overwrite=overwrite)
    for t, task in enumerate(out_dict.keys()):
        for s, sub_data in enumerate(out_dict[task]["data"]):
            # get complex values
            temp_data = sub_data["sig_complex"]
            if s == 0:
                all_data = np.zeros((len(out_dict[task]["data"]),temp_data.shape[0],temp_data.shape[1]),dtype="complex128")
                if t == 0:
                    roi_names = out_dict[task]["roi_names"]
                else:
                    assert roi_names == out_dict[task]["roi_names"], "rois not matched across tasks"
            all_data[s, :, :] = temp_data

        # only plot harmonic of interest
        harmonic_idx = [int(float(x) - 1) for x in harmonic_list]
        all_data = all_data[:, harmonic_idx, :]

        if roi_type not in ["whole"]:
            hot_t, hot_p, hot_crit = hotelling_t2(all_data)

            project_amp, project_err, project_t, project_p, project_real, project_imag = vector_projection(all_data)

            amp_out, phase_out, snr_out = fit_error_ellipse(all_data, ellipse_type, make_plot, return_rad)
            assert np.all(amp_out[1,] >= 0), "warning: ROI {0} with task {1}: lower error bar is smaller than zero".format(roi, task)
            assert np.all(amp_out[2,] >= 0), "warning: ROI {0} with task {1}: upper error bar is smaller than zero".format(roi, task)

            for h, harm in enumerate(harmonic_list):
                stats_df = pd.DataFrame(index=roi_names,columns=['amp_mu', 'ph_mu', 'z_snr', 'el_err_amp', 'el_err_ph', "proj_err",
                                                 "hotT2", "ttest_1s", "project_sub_amps"])
                stats_df.at[roi_names, 'amp_mu'] = amp_out[0, h, :]
                stats_df.at[roi_names, 'ph_mu'] = phase_out[0, h, :]
                stats_df.at[roi_names, 'z_snr'] = snr_out[h, :]
                stats_df.at[roi_names, 'el_err_amp'] = [(amp_out[1:, h, x]) for x in range(all_data.shape[2])]
                stats_df.at[roi_names, 'el_err_ph'] = [(phase_out[1:, h, x]) for x in range(all_data.shape[2])]
                stats_df.at[roi_names, 'proj_err'] = project_err[h, :]
                stats_df.at[roi_names, 'hotT2'] = [(hot_t[h, x], hot_p[h, x]) for x in range(all_data.shape[2])]
                stats_df.at[roi_names, 'ttest_1s'] = [(project_t[h, x], project_p[h, x]) for x in range(all_data.shape[2])]
                stats_df.at[roi_names, 'project_sub_amps'] = project_amp[h, :]

                # compute cycle average and standard error, only once
                if h == 0:

                    all_cycle = np.array([x["mean_cycle"] for x in out_dict[task]["data"]])
                    cycle_ave = np.mean(all_cycle, axis=0, keepdims=True)
                    sub_count = np.count_nonzero(~np.isnan(all_cycle), axis=0)
                    cycle_stderr = np.divide(np.nanstd(all_cycle, axis=0, keepdims=True), np.sqrt(sub_count))
                    cycle_df = pd.DataFrame(index=roi_names, columns=['ave', 'stderr'])
                    cycle_df.at[roi_names, 'ave'] = [(cycle_ave[:, :, x]) for x in range(all_data.shape[2])]
                    cycle_df.at[roi_names, 'stderr'] = [(cycle_stderr[:, :, x]) for x in range(all_data.shape[2])]
                    group_out = {}
                    group_out[harm] = {"stats": stats_df, "cycle": cycle_df}
                else:
                    group_out[harm] = {"stats": stats_df}

        else:
            real_mean = np.mean(np.real(all_data), axis=0)
            imag_mean = np.mean(np.imag(all_data), axis=0)
            amp_out = np.abs(real_mean + 1j * imag_mean)
            phase_out = np.angle(real_mean + 1j * imag_mean, not return_rad)

            # invert phase, so that positive values indicate rightward shift,
            # and negative indicate leftward shift, relative to the cosine
            phase_out = phase_out * -1

            # unwrap negative phases
            unwrap_factor = np.array([360, 2 * np.pi])
            phase_out[phase_out < 0] = phase_out[phase_out < 0] + unwrap_factor[int(return_rad)]

            # four values per harmonic: amp, phase, t2 and p-value
            group_out = np.empty((len(harmonic_list) * 4, len(roi_names)))
            group_out[np.arange(0,len(harmonic_list)*4, 4), :] = amp_out
            group_out[np.arange(1,len(harmonic_list)*4, 4), :] = phase_out

            if "projected" in roi_type:
                # do vector projection
                project_amp, project_err, t_val, p_val, project_real, project_imag = vector_projection(all_data, test_fig=False)
            else:
                # compute Hotelling's T-squared:
                t_val, p_val, hot_crit = hotelling_t2(all_data)
            group_out[np.arange(2, len(harmonic_list) * 4, 4), :] = t_val
            group_out[np.arange(3, len(harmonic_list) * 4, 4), :] = p_val

        group_dictionary[task] = group_out
    if report_timing:
        elapsed = time.time() - start_time
        print_wrap("group analysis complete, took {:02d} seconds".format(round(elapsed)))
    return group_dictionary


def whole_group(exp_folder, harmonic_list=['1'], return_rad=True):
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

                # invert phase, so that positive values indicate rightward shift,
                # and negative indicate leftward shift, relative to the cosine
                phase_mean = phase_mean * -1

                # unwrap negative phases
                phase_mean[phase_mean < 0] = phase_mean[phase_mean < 0] + unwrap_factor[int(return_rad)]

                group_out[:, h * 4 + 1, r] = phase_mean

                project_amp, project_err, project_test, project_real, project_imag = vector_projection(xydata,
                                                                                                       test_fig=False)
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