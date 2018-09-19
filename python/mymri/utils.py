import os, subprocess, sys, glob, shutil, tempfile, re, stat
from os.path import join, abspath
import argparse
import json
import nibabel as nib
import nipype.interfaces.fsl as fsl
import numpy as np
import tarfile


# *****************************
# *** HELPER FUNCTIONS
# *****************************
def bids_anat(sub_id, src_dir, dest_dir, deface=True, update_metadata=False):
    """
    Moves and converts a anatomical folder associated with a T1 or T2
    to BIDS format. Assumes files are named appropriately
    (e.g. "[examinfo]_anat_[T1w,T2w]")
    """

    if update_metadata:
        # no metadata for anats, so return immediately
        return

    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    print('BIDSifying ANATOMY ...')
    counter = 0
    for cur_src in src_dir:
        if corr_name is None:
            #T1 or T2, figure out from file name
            anat_type = cur_src[(cur_src.index('anat_')+5):]
            anat_type = anat_type.replace('run_','run-')
            if "run" in anat_type:
                run = '_' + anat_type.split('_')[1]
            else:
                run = ''
            anat_type = '_'.join(anat_type.split('_')[:1])
            if 'inplane' in cur_src:
                anat_type = 'inplane'+anat_type.replace('w','')
            anat_dest = os.path.join(dest_dir, sub_id + run + '_' + anat_type + '.nii.gz')
        else:
            anat_dest = corr_name[counter]
            counter += 1
        
        anat_src = glob.glob(os.path.join(cur_src,'*nii.gz'))
        assert len(anat_src) == 1, "More than one anat file found in directory %s" % anat_src
        anat_src = anat_src[0]
        
        print('\t' +  "/".join(anat_src.split('/')[-4:]) + ' to\n\t' +  "/".join(anat_dest.split('/')[-5:]) )

        if not os.path.exists(anat_dest):
            shutil.copyfile(anat_src, anat_dest)
            if deface:
                # deface
                print('\tDefacing...')
                subprocess.call("pydeface %s" % anat_dest, shell=True)
                # cleanup
                os.remove(anat_dest)
                defaced_file = anat_dest.replace('.nii','_defaced.nii')
                os.rename(defaced_file,anat_dest)
        else:
            print('Did not save anat because %s already exists!' % anat_dest)

def bids_fmap(sub_id, src_dir, dest_dir, func_dir, update_metadata=False):
    """
    Moves and converts an epi folder associated with a fieldmap
    to BIDS format. Assumes files are named appropriately
    (e.g. "[examinfo]_fmap_fieldmap")
    """
    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    print('BIDSifying FIELDMAP ...')
    counter = 0
    for cur_src in src_dir:
        fmap_files = glob.glob(os.path.join(cur_src,'*.nii.gz'))
        assert len(fmap_files) == 2, "Didn't find the correct number of files in %s" % cur_src
        fmap_index = [0,1]['fieldmap.nii.gz' in fmap_files[1]]
        fmap_src = fmap_files[fmap_index]
        mag_src = fmap_files[1-fmap_index]
        if corr_name is None:
            fmap_dest = os.path.join(dest_dir, sub_id + '_' + 'fieldmap.nii.gz')
            mag_dest = os.path.join(dest_dir, sub_id + '_' + 'magnitude.nii.gz')
        else:
            fmap_dest = corr_name[counter]
            mag_dest = fmap_dest.replace("fieldmap","magnitude")
            counter += 1

        print('\t' +  "/".join(fmap_src.split('/')[-4:]) + ' to\n\t' +  "/".join(fmap_dest.split('/')[-5:]) )

        if not update_metadata:
            shutil.copyfile(fmap_src, fmap_dest)
            shutil.copyfile(mag_src, mag_dest)

        # get metadata
        fmap_meta_dest = fmap_dest.replace('.nii.gz', '.json')  
        if not os.path.exists(fmap_meta_dest):
            try:
                fmap_meta_src = [x for x in glob.glob(os.path.join(cur_src,'*.json')) if 'qa' not in x]
                if not fmap_meta_src:
                    fmap_meta_src = fmap_src
                else:
                    fmap_meta_src = dti_meta_src[0]
                func_runs = [os.sep.join(os.path.normpath(f).split(os.sep)[-3:]) for f in glob.glob(os.path.join(func_dir,'*task*bold.nii.gz'))]
                fmap_meta = get_meta(fmap_meta_src, "fieldmap",'',func_runs)
                json.dump(fmap_meta,open(fmap_meta_dest,'w'))
            except IndexError:
                print("Metadata couldn't be created for %s" % fmap_dest)

def bids_dti(sub_id, src_dir, dest_dir, update_metadata=False):
    """
    Moves and converts an epi folder associated with a fieldmap
    to BIDS format. Assumes files are named appropriately
    (e.g. "[examinfo]_fmap_fieldmap")
    """
    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    counter = 0
    print('BIDSifying DTI ...')
    for cur_src in src_dir:
        dti_files = glob.glob(os.path.join(cur_src,'*.nii.gz'))
        assert len(dti_files) <= 1, "More than one nifti file found in directory %s" % cur_src
        if len(dti_files) == 0:
            print('Skipping %s, no nii.gz file found' % cur_src)
            return
        dti_src = dti_files[0]
        bval_src = dti_src.replace('.nii.gz','.bval')
        bvec_src = dti_src.replace('.nii.gz','.bvec')

        if corr_name is None:
            dti_index = cur_src.index('run')
            dtiname = cur_src[dti_index:]
            # bring to subject directory and divide into sbref and bold
            dti_dest = os.path.join(dest_dir, sub_id + '_' + filename + '.nii.gz')
            dti_dest = dti_dest.replace('run_','run-').replace('dti','dwi')
        else:
            dti_dest = corr_name[counter]
            counter += 1

        bval_dest = dti_dest.replace('.nii.gz','.bval')
        bvec_dest = dti_dest.replace('.nii.gz','.bvec')
        
        print('\t' +  "/".join(dti_src.split('/')[-4:]) + ' to\n\t' +  "/".join(dti_dest.split('/')[-5:]) )

        if not update_metadata:
            shutil.copyfile(dti_src, dti_dest)
            shutil.copyfile(bval_src, bval_dest)
            shutil.copyfile(bvec_src, bvec_dest)

        # get metadata
        dti_meta_dest = dti_dest.replace('.nii.gz', '.json')    
        if not os.path.exists(dti_meta_dest):
            try:
                dti_meta_src = [x for x in glob.glob(os.path.join(cur_src,'*.json')) if 'qa' not in x]
                if not dti_meta_src:
                    dti_meta_src = dti_src
                else:
                    dti_meta_src = dti_meta_src[0]
                dti_meta = get_meta(dti_meta_src, "dti")
                json.dump(dti_meta,open(dti_meta_dest,'w'))
            except IndexError:
                print("Metadata couldn't be created for %s" % dti_dest)
        
        
def bids_pe(sub_id, src_dir, dest_dir, func_dir, update_metadata=False):
    """
    Moves and converts an epi folder associated with an 
    alternative phase encoding direction to BIDS format. 
    Assumes files are named "[examinfo]_pe1"
    """
    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    counter = 0
    print('BIDSifying PE ...')
    for cur_src in src_dir:
        pe_files = glob.glob(os.path.join(cur_src,'*.nii.gz'))
        # remove files that are sometimes added, but are of no interest
        pe_files = [i for i in pe_files if 'phase' not in i.split('/')[-1]]
        assert len(pe_files) <= 1, "More than one pe file found in directory %s" % cur_src
        if len(pe_files) == 0:
            print('Skipping %s, no nii.gz file found' % cur_src)
            return
        pe_src = pe_files[0]
        if corr_name is None:
            pe_index = cur_src.index('run')
            filename = cur_src[pe_index:]
            # bring to subject directory and divide into sbref and bold
            pe_dest = os.path.join(dest_dir, sub_id + '_' + filename + '.nii.gz')
            pe_dest = pe_dest.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_epi').replace('_pe1','').replace('_pe','')
        else:
            pe_dest = corr_name[counter]
            filename = pe_dest[pe_dest.index('run'):]
            filename = filename.replace('.nii.gz','')
            counter += 1

        print('\t' +  "/".join(pe_src.split('/')[-4:]) + ' to\n\t' +  "/".join(pe_dest.split('/')[-5:]) )

        if not update_metadata:
            # check if file exists. If it does, check if the saved file has more time points
            if os.path.exists(pe_dest):
                print('%s already exists!' % pe_dest)
                saved_shape = nib.load(pe_src).shape
                current_shape = nib.load(pe_dest).shape
                print('Dimensions of saved image: %s' % list(saved_shape))
                print('Dimensions of current image: %s' % list(current_shape))
                if (current_shape[-1] <= saved_shape[-1]):
                    print('Current image has fewer or equivalent time points than saved image. Exiting...')
                    return
                else:
                    print('Current image has more time points than saved image. Overwriting...')
        
            # save pe image to bids directory
            shutil.copyfile(pe_src, pe_dest)

        # get pe metadata
        pe_meta_dest = pe_dest.replace('.nii.gz', '.json')  
        if not os.path.exists(pe_meta_dest):
            try:
                pe_meta_src = [x for x in glob.glob(os.path.join(cur_src,'*.json')) if 'qa' not in x]
                if not pe_meta_src:
                    pe_meta_src = pe_src
                else:
                    pe_meta_src = pe_meta_src[0]
                func_runs = [os.sep.join(os.path.normpath(f).split(os.sep)[-3:]) for f in glob.glob(os.path.join(func_dir,'*task*bold.nii.gz'))]
                pe_meta = get_meta(pe_meta_src, "pe", filename, func_runs)
                json.dump(pe_meta,open(pe_meta_dest,'w'))
            except IndexError:
                print("Metadata couldn't be created for %s" % pe_dest)

def bids_sbref(sub_id, src_dir, dest_dir, update_metadata=False):
    """
    Moves and converts an epi folder associated with a sbref
    calibration scan to BIDS format. Assumes tasks are named appropriately
    (e.g. "[examinfo]_task_[task]_run_[run_number]_sbref")
    """

    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    print('BIDSifying SBREF ...')
    counter = 0
    for cur_src in src_dir:
        sbref_files = glob.glob(os.path.join(cur_src,'*.nii.gz'))
        # remove files that are sometimes added, but are of no interest
        sbref_files = [i for i in sbref_files if 'phase' not in i.split('/')[-1]]
        assert len(sbref_files) <= 1, "More than one func file found in directory %s" % cur_src
        if len(sbref_files) == 0:
            print('Skipping %s, no nii.gz file found' % cur_src)
            return
        sbref_src = sbref_files[0]

        if corr_name is None:
            task_index = cur_src.index('task')
            filename = cur_src[task_index:]
            # bring to subject directory and divide into sbref and bold
            sbref_dest = os.path.join(dest_dir, sub_id + '_' + filename + '.nii.gz')
            sbref_dest = sbref_dest.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_sbref')
        else:
            sbref_dest = corr_name[counter]
            filename = sbref_dest[sbref_dest.index('task'):]
            filename = filename.replace('.nii.gz','')
            counter += 1

        print('\t' +  "/".join(sbref_src.split('/')[-4:]) + ' to\n\t' +  "/".join(sbref_dest.split('/')[-5:]) )

        if not update_metadata:
            # check if file exists. If it does, check if the saved file has more time points
            if os.path.exists(sbref_dest):
                print('%s already exists!' % sbref_dest)
                saved_shape = nib.load(sbref_dest).shape
                current_shape = nib.load(sbref_src).shape
                print('Dimensions of saved image: %s' % list(saved_shape))
                print('Dimensions of current image: %s' % list(current_shape))
                if (current_shape[-1] <= saved_shape[-1]):
                    print('Current image has fewer or equivalent time points than saved image. Exiting...')
                    return
                else:
                    print('Current image has more time points than saved image. Overwriting...')

            # save sbref image to bids directory
            shutil.copyfile(sbref_src, sbref_dest)
        
        # get sbref metadata
        sbref_meta_dest = sbref_dest.replace('.nii.gz', '.json')    
        if not os.path.exists(sbref_meta_dest):
            try:
                sbref_meta_src = [x for x in glob.glob(os.path.join(cur_src,'*.json')) if 'qa' not in x]
                if not sbref_meta_src:
                    sbref_meta_src = sbref_src
                else:
                    sbref_meta_src = sbref_meta_src[0]
                sbref_meta = get_meta(sbref_meta_src, 'sbref', filename)
                json.dump(sbref_meta,open(sbref_meta_dest,'w'))
            except IndexError:
                print("Metadata couldn't be created for %s" % sbref_file)

def bids_task(sub_id, src_dir, dest_dir, update_metadata=False):
    """
    Moves and converts an epi folder associated with a task
    to BIDS format. Assumes tasks are named appropriately
    (e.g. "[examinfo]_task_[task]_run_[run_number]_ssg")
    """

    if isinstance(src_dir, dict):
        # use dictionary, define list of destination files and list of source folders
        corr_name = [os.path.join(dest_dir, sub_id + '_' + x + '.nii.gz') for x in src_dir.values()]
        src_dir = src_dir.keys() 
    else:
        corr_name = None

    # loop over source folders
    print('BIDSifying TASK ...')
    counter = 0
    for cur_src in src_dir:
        task_file = glob.glob(os.path.join(cur_src,'*.nii.gz'))
        task_file = [i for i in task_file if 'phase' not in i.split('/')[-1]]
        assert len(task_file) <= 1, "More than one func file found in directory %s" % cur_src
        if len(task_file) == 0:
            print('Skipping %s, no nii.gz file found' % cur_src)
            return
        task_src = task_file[0]

        if corr_name is None:
            task_index = cur_src.index('task')
            taskname = cur_src[task_index:]
            # bring to subject directory and divide into sbref and bold
            task_dest = os.path.join(dest_dir, sub_id + '_' + taskname + '.nii.gz')
            task_dest = task_dest.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_bold')
        else:
            task_dest = corr_name[counter]
            taskname = task_dest[task_dest.index('task'):]
            taskname = taskname.replace('.nii.gz','')
            counter += 1

        print('\t' +  "/".join(task_src.split('/')[-4:]) + ' to\n\t' +  "/".join(task_dest.split('/')[-5:]) )

        if not update_metadata:
            # check if file exists. If it does, check if the saved file has more time points
            if os.path.exists(task_dest):
                print('%s already exists!' % task_dest)
                saved_shape = nib.load(task_src).shape
                current_shape = nib.load(task_dest).shape
                print('Dimensions of saved image: %s' % list(saved_shape))
                print('Dimensions of current image: %s' % list(current_shape))
                if (current_shape[-1] <= saved_shape[-1]):
                        print('Current image has fewer or equal time points than saved image. Exiting...')
                        return
                else:
                        print('Current image has more time points than saved image. Overwriting...')
            # save bold image to bids directory
            shutil.copyfile(task_src, task_dest)

            # get physio if it exists
            physio_file = glob.glob(os.path.join(cur_src, '*physio.tgz'))
            if len(physio_file)>0:
                assert len(physio_file)==1, ("More than one physio file found in directory %s" % cur_src)
                tar = tarfile.open(physio_file[0])
                tar.extractall(dest_dir)
                # extract the actual filename of the physio data
                physio_file = os.path.basename(physio_file[0])[:-4]
                for pfile in glob.iglob(os.path.join(dest_dir, physio_file, '*Data*')):
                        pname = 'respiratory' if 'RESP' in pfile else 'cardiac'
                        new_physio_file = task_dest.replace('_bold.nii.gz', 
                                                                '_recording-' + pname + '_physio.tsv.gz')
                        f = np.loadtxt(pfile)
                        np.savetxt(new_physio_file, f, delimiter = '\t')
                shutil.rmtree(os.path.join(dest_dir, physio_file))
        
        # get task metadata
        task_meta_dest = task_dest.replace('.nii.gz', '.json')  
        if not os.path.exists(task_meta_dest):
            task_meta_src = [x for x in glob.glob(os.path.join(cur_src,'*.json')) if 'qa' not in x]
            if not task_meta_src:
                task_meta_src = task_src
            else:
                task_meta_src = task_meta_src[0]
            task_meta = get_meta(task_meta_src, "task", taskname)
            json.dump(task_meta,open(task_meta_dest,'w'))

def cleanup(path):
    for f in glob.glob(os.path.join(path, '*')):
        new_name = f.replace('task_', 'task-').replace('run_','run-').replace('_ssg','')
        if 'run' in new_name:
            new_name = new_name.split('_')
            # put leading zeros in run numbers
            run_no = [x.replace(x[4:],x[4:].zfill(2)) for x in new_name if 'run' in x]
            index = [i for i, s in enumerate(new_name) if 'run' in s]
            new_name[index[0]] = run_no[0]
            new_name = '_'.join(new_name)
            os.rename(f,new_name)

def get_meta(meta_file, scan_type, taskname=None, intended_list=None):
    """
    Returns BIDS meta data for bold 
    """
    if '.nii' in meta_file:
        descrip = str(nib.load(meta_file).header['descrip'])
        if 'mux' in descrip:
            mux = int(descrip[descrip.index('mux=')+4:].split(';')[0])
        elif 'mux' in meta_file:
            mux = int(meta_file[meta_file.index('mux')+3])
        else:
            mux = int(1)
        n_echoes = descrip[descrip.index('acq=')+4:].split(';')[0]
        n_echoes = int(n_echoes.split(',')[1].replace(']',''))
        # echo_time and echo spacing in nifti header is expressed in ms
        echo_spacing = float(descrip[descrip.index('ec=')+3:].split(';')[0]) / 1000
        echo_time = float(descrip[descrip.index('te=')+3:].split(';')[0]) / 1000
        flip_angle = float(descrip[descrip.index('fa=')+3:].split(';')[0])
        # note, mux factor has already been incorporated in slice count
        n_slices = int(nib.load(meta_file).header['dim'][3])
        tr = float(nib.load(meta_file).header['pixdim'][4])
        phase_dir = bin(nib.load(meta_file).header['dim_info'])
        phase_dir = int(phase_dir[int(len(phase_dir)/2) : int(len(phase_dir)/2+2)],2)
        # make phase_dir zero-indexed
        phase_dir = phase_dir-1
        if 'pe' in descrip:
            pe_polar = descrip[descrip.index('pe=')+3:].split(';')[0]
            if '0' in pe_polar:
                phase_sign = '-'
            else:
                phase_sign = ''
        else:
            # if no pe field, assume pe=0
            phase_sign = '-'
    else:
        meta_in = json.load(open(meta_file,'r'))
        mux = meta_in['num_bands']
        n_echoes = meta_in['acquisition_matrix_y']
        echo_spacing = meta_in['effective_echo_spacing']
        echo_time =  meta_in['te']
        flip_angle = meta_in['flip_angle']
        # note, mux factor has not been incorporated in slice count
        n_slices = meta_in['num_slices'] * mux
        tr = meta_in['tr']
        # note, phase_dir is already zero-indexed
        phase_dir = meta_in['phase_encode']
        phase_sign = '-'

    meta_out = {}
    total_time = (n_echoes-1)*echo_spacing
    # fill in metadata
    meta_out['SliceTiming'] = get_slice_timing(n_slices, tr, mux = mux) 
    meta_out['FlipAngle'] = flip_angle
    if (scan_type in ['task', 'sbref']):
        meta_out['TaskName'] = taskname.split('_')[0]
        meta_out['RepetitionTime'] = tr
    elif (scan_type == 'fieldmap'):
        meta_out['IntendedFor'] = intended_list
        meta_out['Units'] = 'Hz'
    elif (scan_type == 'pe'):
        meta_out['RepetitionTime'] = tr
        meta_out['IntendedFor'] = intended_list
        meta_out['Units'] = 'Hz'
        phase_sign = ''
        # assume phase sign is flipped
    elif (scan_type == 'dti'):
        meta_out = {} # clear meta data, dti only needs some of these
    else:
        raise ValueError("get_meta: unknown type %s." % scan_type)
    # meta data for all types 
    meta_out['PhaseEncodingDirection'] = ['i','j','k'][phase_dir] + phase_sign   
    meta_out['EffectiveEchoSpacing'] = echo_spacing
    meta_out['EchoTime'] = echo_time
    meta_out['TotalReadoutTime'] = total_time
    return meta_out

def get_slice_timing(nslices, tr, mux = 1, order = 'ascending'):
    """
    nslices: int, total number of slices
    tr: float, repetition total_time
    mux: int, optional mux factor
    """
    if mux is not 1:
        assert nslices%mux == 0
        nslices = nslices//mux
        mux_slice_acq_order = list(range(0,nslices,2)) + list(range(1,nslices,2))
        mux_slice_acq_time = [float(s)/nslices*tr for s in range(nslices)]
        unmux_slice_acq_order = [nslices*m+s for m in range(mux) for s in mux_slice_acq_order]
        unmux_slice_acq_time = mux_slice_acq_time * mux
        slicetimes = list(zip(unmux_slice_acq_time,unmux_slice_acq_order))
    else:
        slice_acq_order = list(range(0,nslices,2)) + list(range(1,nslices,2))
        slice_acq_time = [float(s)/nslices*tr for s in range(nslices)]
        slicetimes = list(zip(slice_acq_time,slice_acq_order))
    #reorder slicetimes by slice number
    sort_index = sorted(enumerate([x[1] for x in slicetimes]), key= lambda x: x[1])
    sorted_slicetimes = [slicetimes[i[0]][0] for i in sort_index]
    return sorted_slicetimes

def get_subj_path(nims_file, data_dir, id_correction_dict=None):
    """
    Takes a path to a nims_file and returns a subject id
    If a dictionary specifying id corrections is provided
    (in the form of a json file with exam numbers as keys and
    ids as values), the function will return the corrected id number
    """
    exam_number = nims_file.split('_')[-1]
    # correct session if provided
    if id_correction_dict:
        sub_session = id_correction_dict.get(exam_number,'skip')
    else:
        meta_json = glob.glob(os.path.join(nims_file,'*','*1.json'))[0]
        meta_file = json.load(open(meta_json,'r'))
        sub_session = str(meta_file['patient_id'].split('@')[0])
    if 'skip' in sub_session:
        subj_dir = None
    else:
        sub_id = sub_session.split('_')[0]
        session = '1'
        if '_' in sub_session:
            session = sub_session.split('_')[1]
        session = session.zfill(2)
        subj_dir = os.path.join(data_dir, 'sub-'+sub_id, 'ses-'+session)
    return subj_dir

def mkdir(path):
    try:
            os.mkdir(path)
    except OSError:
            print('Directory %s already existed' % path)
    return path

def rsync(input, output):
    #if ':' not in output:
    #       try: 
    #               os.makedirs(output)
    #       except OSError:
    #               if not os.path.isdir(output):
    #                       raise
    #else:
    #       remote, path = output.split(':')
    #       print(remote,path)
    #       subprocess.Popen("ssh %s mkdir -p %s" % (remote, path), shell=True).wait()
    cmd = "rsync -avz --progress --remove-source-files %s/* %s/." % (input, output)
    p = subprocess.Popen(cmd, shell=True)
    stdout, stderr = p.communicate()
    return stderr

# *****************************
# *** Main BIDS function
# *****************************
def bids_subj(orig_dir, temp_dir, out_dir, deface=True, run_correction=None, update_metadata=False):
    """
    Takes 
        orig_dir (the path to the subject's data directory in the original format, on nims or elsewhere) 
        temp_dir (the path to the temporary directory),
        and out_dir (the path to the BIDS output directory)
    Moves/converts that subject's data to BIDS
    """
    # extract subject and session ID to use for output
    split_dir = os.path.normpath(temp_dir).split(os.sep)
    sub_id = [x for x in split_dir if 'sub' in x][0]
    ses_id = [x for x in split_dir if 'ses' in x][0]
    out_dir = os.path.join(out_dir,sub_id,ses_id)

    if os.path.exists(out_dir) and not update_metadata:
        print("Path %s already exists. Skipping." % out_dir)
        success = False
    else:
        success = True
        print("BIDSifying %s" % orig_dir)
        print("Using temp path %s" % temp_dir)
        print("Using bids path: %s" % out_dir)
        base_file_id = sub_id + '_' + ses_id
        # split subject path into a super subject path and a session path
        os.makedirs(temp_dir)
        anat_temp = mkdir(os.path.join(temp_dir,'anat'))
        func_temp = mkdir(os.path.join(temp_dir,'func'))
        fmap_temp = mkdir(os.path.join(temp_dir,'fmap'))
        dti_temp = mkdir(os.path.join(temp_dir,'dwi'))

        if run_correction is None:
            task_origs = sorted(glob.glob(os.path.join(orig_dir,'*task*')))[::-1]
            task_origs = [x for x in task_origs if 'sbref' not in x and 'pe' not in x]
            anat_origs = sorted(glob.glob(os.path.join(orig_dir,'*anat*')))[::-1]
            pe_origs = sorted(glob.glob(os.path.join(orig_dir,'*pe*')))[::-1]
            fmap_origs = sorted(glob.glob(os.path.join(orig_dir,'*fieldmap*')))[::-1]
            sbref_origs = sorted(glob.glob(os.path.join(orig_dir,'*sbref*')))[::-1]
            dti_origs = sorted(glob.glob(os.path.join(orig_dir,'*dti*')))[::-1]
        else:
            run_filtered = {k:v for k,v in run_correction.items() if orig_dir.split("_")[-1] in k}
            task_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'task' in v and 'sbref' not in v and 'pe' not in v}
            anat_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'T1' in v or 'T2' in v}
            pe_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'epi' in v}
            fmap_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'fieldmap' in v}
            sbref_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'sbref' in v}
            dti_origs = {os.path.join(orig_dir,k):v for k,v in run_filtered.items() if 'dwi' in v}

        # anat files
        if anat_origs: bids_anat(base_file_id, anat_origs, anat_temp, deface, update_metadata)

        # task files
        if task_origs: bids_task(base_file_id, task_origs, func_temp, update_metadata)

        # sbref files
        if sbref_origs: bids_sbref(base_file_id, sbref_origs, func_temp, update_metadata)
            
        cleanup(func_temp)
        cleanup(anat_temp)

        # pe files. note, using fmap path
        if pe_origs: bids_pe(base_file_id, pe_origs, fmap_temp, func_temp, update_metadata)

        # fmap files
        if fmap_origs: bids_fmap(base_file_id, fmap_origs, fmap_temp, func_temp, update_metadata)

        cleanup(fmap_temp)

        # fmap files
        if dti_origs: bids_dti(base_file_id, dti_origs, dti_temp, update_metadata)

        return success
def get_subdir(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

# *****************************
# *** UTILITY FUNCTIONS 
# *****************************

def BidsOrganizer(
    study_dir,
    id_correction=None, 
    run_correction=None, 
    record=None, 
    study_id=None,
    bids_dir='/Volumes/svndl/RAW_DATA/MRI_RAW', 
    temp_dir='/Volumes/Denali_DATA1/TEMP',
    run_all=False, 
    deface=False,
    update_metadata=False):
    # *****************************
    # *** Constant Variables
    # *****************************
    cardiac_bids = {
        "StartTime": -30.0,
        "SamplingFrequency": .01,
        "Columns": ["cardiac"]
    }

    respiratory_bids = {
        "StartTime": -30.0,
        "SamplingFrequency":  .04,
        "Columns": ["respiratory"]
    }

    # check if BIDS path exists
    if not os.path.isdir(bids_dir):
        print('BIDS directory %s does not exist!' % bids_dir)
        return
    
    #study name
    if study_id is None:
        study_id = study_dir.split('/')[-1]
    # output directory
    out_dir = os.path.join(bids_dir,study_id)
    mkdir(out_dir)
    # directories to BIDSify
    nims_dirs = sorted(get_subdir(study_dir),reverse=True)
    if run_all is False:
        nims_dirs = [nims_dirs[0]]
    # get temporary directory to save bids in
    temp_dir = os.path.join(temp_dir,study_id)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    mkdir(temp_dir)
    
    # set id_correction dict if provided
    if id_correction is not None and os.path.isfile(id_correction): 
        print('Using ID correction json file: %s' % id_correction)
        id_correction = json.load(open(id_correction,'r'))
    else:
        print('No ID correction')

    # set run_correction dict if provided
    if run_correction is not None and os.path.isfile(run_correction): 
        print('Using run correction json file: %s' % run_correction)
        run_correction = json.load(open(run_correction,'r'))
    else:
        print('No run name correction')

    # record file
    if record is None: 
        record = record
    else:
        record = '/Volumes/svndl/RAW_DATA/completed_files.txt'
        print('Using record file: %s' % record)

    #header file
    header = {'Name': study_id, 'BIDSVersion': '1.51-rc1'}
    json.dump(header,open(os.path.join(temp_dir, 'dataset_description.json'),'w'))

    # bidsify all subjects in path
    for nims_file in nims_dirs:
        temp_subj  = get_subj_path(nims_file, temp_dir, id_correction)
        print("*******************************************************************")
        if temp_subj == None:
            print("Skipping %s" % nims_file)
            continue
        success = bids_subj(nims_file, temp_subj, out_dir, deface, run_correction, update_metadata)
        # move files
        if success: 
            err = rsync(temp_dir, out_dir)
            if err != None: success = False
            if success: 
                print('Successfully transferred %s' % nims_file)
                if record != None:
                    with open(record, 'a') as f:
                        f.write(nims_file)  
                        f.write('\n')
    # remove temp folder
    shutil.rmtree(temp_dir)


def BidsLinks(anat_session,exp_session,remove_only=False):
    sublist = [x.split("/")[-1] for x in glob.glob(exp_session+"/sub*")]
    for sub in sublist:
        for ses in glob.glob(exp_session+"/"+sub+"/ses*"): 
            # usually only one session, but allow for more than one
            target_anat = ses+"/anat"
            generating_files = True
            while generating_files:
                target_files = [x for x in glob.glob(target_anat+"/*") if "inplane" not in x ] 
                if not target_files:
                    try:
                        source_anat = glob.glob(anat_session+"/"+sub+"/ses*")[0]+"/anat"
                        source_files = [x for x in glob.glob(source_anat+"/*") if "inplane" not in x ]
                    except:
                        print("ERROR: Source anatomies missing for subject {0} in anat session {1}".format(sub,anat_session))
                    target_files = [x.replace(source_anat,target_anat) for x in source_files]
                    if ses.split("/")[-1] not in "ses-01":
                        # if not session 1, replace session id
                        target_files = [x.replace("ses-01",ses.split("/")[-1]) for x in target_files]
                    for i, item in enumerate(target_files):
                        os.symlink(source_files[i], target_files[i])
                    generating_files = False
                    print("Made symlinks to source anatomies for subject {0} in anat session {1}".format(sub,anat_session))
                else:
                    for target in target_files:
                        if os.path.islink(target):
                            os.remove(target)
                    if remove_only:
                    # don't jump back into file creation loop
                        generating_files = False

def HardCreate(bids_dir,experiment,subjects="all",fs_dir="main"):
    """Function creates hardlinks from freesurfer directory to the experiment folder

    Parameters
    ------------
    bids_dir : string
        The directory for BIDS Analysis. Should contain the freesurfer folder and experiment folder.
    experiment : string
        Used for location of the experiment folder within the BIDS directory
    subjects : list of strings
        This is a list of the subjects that require hardlinks to be made
    fs_dir : string
        The freesurfer directory
    Returns
    ------------
    checking_dic : dictionary
        Contains the source and destination of the files. Used for checking that the new directory 
        is actually a hard link of the old one.
    """
    checking_dic = {}
    if fs_dir.lower() in "main":
        fs_dir = "{0}/freesurfer".format(bids_dir)
        if "all" in subjects:
            subjects = [x for x in os.path.os.listdir(fs_dir) if "sub-" in x[0:4]]
        for sub in subjects:
            src = "{0}/{1}".format(fs_dir,sub)
            if not os.path.isdir(src):
                print("... freesurfer directory {0} does not exist, skipping".format(src))
                continue
            else:
                dst = "{0}/{1}/freesurfer/{2}".format(bids_dir,experiment,sub)
                if os.path.isdir(dst):
                    # remove dst directory, before making link
                    shutil.rmtree(dst)
                HardCopy(src,dst)
                checking_dic[sub] = [src,dst]
        HardCheck(checking_dic)
    return checking_dic

def HardCheck(checking_dic):
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

def HardCopy(src, dst):
    working_dir = os.getcwd()
    dest = abspath(dst)
    os.mkdir(dst)
    os.chdir(src)
    for root, dirs, files in os.walk('.'):
        cur_dst = join(dst, root)
        for d in dirs:
            os.mkdir(join(cur_dst, d))
        for f in files:
            fromfile = join(root, f)
            to = join(cur_dst, f)
            os.link(fromfile, to)
    os.chdir(working_dir)
