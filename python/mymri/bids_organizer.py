import argparse
import glob
import json
import nibabel as nib
import nipype.interfaces.fsl as fsl
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tarfile

# *****************************
# *** HELPER FUNCTIONS
# *****************************
def bids_anat(sub_id, src_dir, dest_dir, deface=True):
	"""
	Moves and converts a anatomical folder associated with a T1 or T2
	to BIDS format. Assumes files are named appropriately
	(e.g. "[examinfo]_anat_[T1w,T2w]")
	"""
	#T1 or T2?
	anat_type = src_dir[(src_dir.index('anat_')+5):]
	anat_type = anat_type.replace('run_','run-')
	if "run" in anat_type:
	    run = '_' + anat_type.split('_')[1]
	else:
	    run = ''
	anat_type = '_'.join(anat_type.split('_')[:1])
	if 'inplane' in src_dir:
		anat_type = 'inplane'+anat_type.replace('w','')
	anat_file = glob.glob(os.path.join(src_dir,'*nii.gz'))
	assert len(anat_file) == 1, "More than one anat file found in directory %s" % src_dir
	new_file = os.path.join(dest_dir, sub_id + run + '_' + anat_type + '.nii.gz')
	if not os.path.exists(new_file):
			shutil.copyfile(anat_file[0], new_file)
			if deface:
				# deface
				print('\tDefacing...')
				subprocess.call("pydeface %s" % new_file, shell=True)
				# cleanup
				os.remove(new_file)
				defaced_file = new_file.replace('.nii','_defaced.nii')
				os.rename(defaced_file,new_file)
	else:
			print('Did not save anat because %s already exists!' % new_file)

def bids_fmap(sub_id, src_dir, dest_dir, func_dir):
	"""
	Moves and converts an epi folder associated with a fieldmap
	to BIDS format. Assumes files are named appropriately
	(e.g. "[examinfo]_fmap_fieldmap")
	"""
	fmap_files = glob.glob(os.path.join(src_dir,'*.nii.gz'))
	assert len(fmap_files) == 2, "Didn't find the correct number of files in %s" % src_dir
	fmap_index = [0,1]['fieldmap.nii.gz' in fmap_files[1]]
	fmap_file = fmap_files[fmap_index]
	mag_file = fmap_files[1-fmap_index]
	fmap_name = sub_id + '_' + 'fieldmap.nii.gz'
	mag_name = sub_id + '_' + 'magnitude.nii.gz'
	if not os.path.exists(os.path.join(dest_dir, fmap_name)):
			shutil.copyfile(fmap_file, os.path.join(dest_dir, fmap_name))
			shutil.copyfile(mag_file, os.path.join(dest_dir, mag_name))
			func_runs = [os.sep.join(os.path.normpath(f).split(os.sep)[-3:]) for f in glob.glob(os.path.join(func_dir,'*task*bold.nii.gz'))]
			fieldmap_meta = {'Units': 'Hz', 'IntendedFor': func_runs}
			json.dump(fieldmap_meta,open(os.path.join(dest_dir, sub_id + '_fieldmap.json'),'w'))
	else:
			print('Did not save fmap_epi because %s already exists!' % os.path.join(dest_dir, fmap_name))

def bids_pe(sub_id, src_dir, dest_dir, func_dir):
	"""
	Moves and converts an epi folder associated with an 
	alternative phase encoding direction to BIDS format. 
	Assumes files are named "[examinfo]_pe1"
	"""
	pe_index = src_dir.index('run')
	filename = src_dir[pe_index:]
	pe_files = glob.glob(os.path.join(src_dir,'*.nii.gz'))
	# remove files that are sometimes added, but are of no interest
	pe_files = [i for i in pe_files if 'phase' not in i]
	assert len(pe_files) <= 1, "More than one pe file found in directory %s" % pe_files
	if len(pe_files) == 0:
			print('Skipping %s, no nii.gz file found' % src_dir)
			return
	# bring to subject directory and divide into sbref and bold
	pe_file = os.path.join(dest_dir, sub_id + '_' + filename + '.nii.gz')
	pe_file = pe_file.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_epi').replace('_pe1','').replace('_pe','')
	# check if file exists. If it does, check if the saved file has more time points
	if os.path.exists(pe_file):
			print('%s already exists!' % pe_file)
			saved_shape = nib.load(pe_file).shape
			current_shape = nib.load(pe_file[0]).shape
			print('Dimensions of saved image: %s' % list(saved_shape))
			print('Dimensions of current image: %s' % list(current_shape))
			if (current_shape[-1] <= saved_shape[-1]):
					print('Current image has fewer or equivalent time points than saved image. Exiting...')
					return
			else:
					print('Current image has more time points than saved image. Overwriting...')
	# save sbref image to bids directory
	shutil.copyfile(pe_files[0], pe_file)
	# get metadata
	pe_meta_file = pe_file.replace('.nii.gz', '.json')	
	if not os.path.exists(pe_meta_file):
		try:
				meta_file = [x for x in glob.glob(os.path.join(src_dir,'*.json')) 
					if 'qa' not in x][0]
				pe_meta = get_functional_meta(meta_file, filename)
				pe_meta['Units'] = 'Hz' 
				func_runs = [os.sep.join(os.path.normpath(f).split(os.sep)[-3:]) for f in glob.glob(os.path.join(func_dir,'*task*bold.nii.gz'))]
				pe_meta['IntendedFor'] = func_runs
				json.dump(pe_meta,open(pe_meta_file,'w'))
		except IndexError:
				print("Metadata couldn't be created for %s" % sbref_file)

def bids_sbref(sub_id, src_dir, dest_dir):
	"""
	Moves and converts an epi folder associated with a sbref
	calibration scan to BIDS format. Assumes tasks are named appropriately
	(e.g. "[examinfo]_task_[task]_run_[run_number]_sbref")
	"""
	task_index = src_dir.index('task')
	filename = src_dir[task_index:]
	sbref_files = glob.glob(os.path.join(src_dir,'*.nii.gz'))
	# remove files that are sometimes added, but are of no interest
	sbref_files = [i for i in sbref_files if 'phase' not in i]
	assert len(sbref_files) <= 1, "More than one func file found in directory %s" % src_dir
	if len(sbref_files) == 0:
			print('Skipping %s, no nii.gz file found' % src_dir)
			return

	# bring to subject directory and divide into sbref and bold
	sbref_file = os.path.join(dest_dir, sub_id + '_' + filename + '.nii.gz')
	sbref_file = sbref_file.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_sbref')

	# check if file exists. If it does, check if the saved file has more time points
	if os.path.exists(sbref_file):
			print('%s already exists!' % sbref_file)
			saved_shape = nib.load(sbref_file).shape
			current_shape = nib.load(sbref_files[0]).shape
			print('Dimensions of saved image: %s' % list(saved_shape))
			print('Dimensions of current image: %s' % list(current_shape))
			if (current_shape[-1] <= saved_shape[-1]):
					print('Current image has fewer or equivalent time points than saved image. Exiting...')
					return
			else:
					print('Current image has more time points than saved image. Overwriting...')
	# save sbref image to bids directory
	shutil.copyfile(sbref_files[0], sbref_file)
	# get metadata
	upper_dir = "/".join(dest_dir.split('/')[:-1])
	# remove run number from meta file
	sbref_meta_file = os.path.join(upper_dir, sub_id + '_' + filename[:(filename.index('run')-1)] + '_sbref.json')
	sbref_meta_file = sbref_meta_file.replace('task_', 'task-')

	if not os.path.exists(sbref_meta_file):
		try:
			meta_file = [x for x in glob.glob(os.path.join(src_dir,'*.json')) 
				if 'qa' not in x][0]
			func_meta = get_functional_meta(meta_file, filename)
			json.dump(func_meta,open(sbref_meta_file,'w'))
		except IndexError:
			print("Metadata couldn't be created for %s" % sbref_file)

def bids_task(sub_id, src_dir, dest_dir):
	"""
	Moves and converts an epi folder associated with a task
	to BIDS format. Assumes tasks are named appropriately
	(e.g. "[examinfo]_task_[task]_run_[run_number]_ssg")
	"""
	task_index = src_dir.index('task')
	taskname = src_dir[task_index:]
	task_file = glob.glob(os.path.join(src_dir,'*.nii.gz'))
	task_file = [i for i in task_file if 'phase' not in i]
	assert len(task_file) <= 1, "More than one func file found in directory %s" % src_dir
	if len(task_file) == 0:
			print('Skipping %s, no nii.gz file found' % src_dir)
			return

	# bring to subject directory and divide into sbref and bold
	bold_file = os.path.join(dest_dir, sub_id + '_' + taskname + '.nii.gz')
	bold_file = bold_file.replace('task_', 'task-').replace('run_','run-').replace('_ssg','_bold')
	# check if file exists. If it does, check if the saved file has more time points
	if os.path.exists(bold_file):
			print('%s already exists!' % bold_file)
			saved_shape = nib.load(bold_file).shape
			current_shape = nib.load(task_file[0]).shape
			print('Dimensions of saved image: %s' % list(saved_shape))
			print('Dimensions of current image: %s' % list(current_shape))
			if (current_shape[-1] <= saved_shape[-1]):
					print('Current image has fewer or equal time points than saved image. Exiting...')
					return
			else:
					print('Current image has more time points than saved image. Overwriting...')
	# save bold image to bids directory
	shutil.copyfile(task_file[0], bold_file)
	# get epi metadata
	upper_dir = "/".join(dest_dir.split('/')[:-1])
	# remove run number from meta file
	bold_meta_file = os.path.join(upper_dir, sub_id + '_' + taskname[:(taskname.index('run')-1)] + '_bold.json')
	bold_meta_file = bold_meta_file.replace('task_', 'task-')

	if not os.path.exists(bold_meta_file):
		meta_file = [x for x in glob.glob(os.path.join(src_dir,'*.json')) if 'qa' not in x][0]
		func_meta = get_functional_meta(meta_file, taskname)
		json.dump(func_meta,open(bold_meta_file,'w'))
	# get physio if it exists
	physio_file = glob.glob(os.path.join(src_dir, '*physio.tgz'))
	if len(physio_file)>0:
			assert len(physio_file)==1, ("More than one physio file found in directory %s" % src_dir)
			tar = tarfile.open(physio_file[0])
			tar.extractall(dest_dir)
			# extract the actual filename of the physio data
			physio_file = os.path.basename(physio_file[0])[:-4]
			for pfile in glob.iglob(os.path.join(dest_dir, physio_file, '*Data*')):
					pname = 'respiratory' if 'RESP' in pfile else 'cardiac'
					new_physio_file = bold_file.replace('_bold.nii.gz', 
															'_recording-' + pname + '_physio.tsv.gz')
					f = np.loadtxt(pfile)
					np.savetxt(new_physio_file, f, delimiter = '\t')
			shutil.rmtree(os.path.join(dest_dir, physio_file))

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

def get_functional_meta(json_file, taskname):
    """
    Returns BIDS meta data for bold 
    """
    meta_file = json.load(open(json_file,'r'))
    meta_data = {}
    mux = meta_file['num_bands']
    nslices = meta_file['num_slices'] * mux
    tr = meta_file['tr']
    n_echoes = meta_file['acquisition_matrix_y'] 
    # fill in metadata
    meta_data['TaskName'] = taskname.split('_')[1]
    meta_data['EffectiveEchoSpacing'] = meta_file['effective_echo_spacing']
    meta_data['EchoTime'] = meta_file['te']
    meta_data['FlipAngle'] = meta_file['flip_angle']
    meta_data['RepetitionTime'] = tr
    # slice timing
    meta_data['SliceTiming'] = get_slice_timing(nslices, tr, mux = mux)
    total_time = (n_echoes-1)*meta_data['EffectiveEchoSpacing']
    meta_data['TotalReadoutTime'] = total_time
    meta_data['PhaseEncodingDirection'] = ['i','j','k'][meta_file['phase_encode']] + '-'        
    return meta_data

def get_fmap_epi_meta(json_file, intended_list):
    """
    Returns BIDS meta data for epi fieldmaps
    """
    meta_file = json.load(open(json_file,'r'))
    meta_data = {}
    mux = meta_file['num_bands']
    nslices = meta_file['num_slices'] * mux
    n_echoes = meta_file['acquisition_matrix_y'] 
    # fill in metadata
    meta_data['EffectiveEchoSpacing'] = meta_file['effective_echo_spacing']
    total_time = (n_echoes-1)*meta_data['EffectiveEchoSpacing']
    meta_data['TotalReadoutTime'] = total_time
    meta_data['PhaseEncodingDirection'] = ['i','j','k'][meta_file['phase_encode']]    
    meta_data['IntendedFor'] = intended_list        
    return meta_data

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
	meta_json = glob.glob(os.path.join(nims_file,'*','*1.json'))[0]
	meta_file = json.load(open(meta_json,'r'))
	exam_number = str(meta_file['exam_number'])
	sub_session = str(meta_file['patient_id'].split('@')[0])
	# correct session if provided
	if id_correction_dict:
			sub_session = id_correction_dict.get(exam_number,sub_session)
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
	#		try: 
	#				os.makedirs(output)
	#		except OSError:
	#				if not os.path.isdir(output):
	#						raise
	#else:
	#		remote, path = output.split(':')
	#		print(remote,path)
	#		subprocess.Popen("ssh %s mkdir -p %s" % (remote, path), shell=True).wait()
	cmd = "rsync -avz --progress --remove-source-files %s/* %s/." % (input, output)
	p = subprocess.Popen(cmd, shell=True)
	stdout, stderr = p.communicate()
	return stderr

# *****************************
# *** Main BIDS function
# *****************************
def bids_subj(orig_dir, temp_dir, out_dir, deface=True):
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

	if os.path.exists(out_dir):
		print("Path %s already exists. Skipping." % out_dir)
		success = False
	else:
		success = True
		print("********************************************")
		print("BIDSifying %s" % orig_dir)
		print("Using temp path %s" % temp_dir)
		print("Using nims path: %s" % out_dir)
		print("********************************************")
		base_file_id = sub_id + '_' + ses_id
		# split subject path into a super subject path and a session path
		os.makedirs(temp_dir)
		anat_temp = mkdir(os.path.join(temp_dir,'anat'))
		func_temp = mkdir(os.path.join(temp_dir,'func'))
		fmap_temp = mkdir(os.path.join(temp_dir,'fmap'))

		# task files
		task_origs = sorted(glob.glob(os.path.join(orig_dir,'*task*')))[::-1]
		task_origs = [x for x in task_origs if 'sbref' not in x and 'pe' not in x]
		if task_origs:
			print('BIDSifying task...')
			for task_orig in task_origs:
				print('\t' + task_orig)
				bids_task(base_file_id, task_orig, func_temp)

		# anat files
		anat_origs = sorted(glob.glob(os.path.join(orig_dir,'*anat*')))[::-1]
		if anat_origs:
			print('BIDSifying anatomy...')
			for anat_orig in anat_origs:
				print('\t' + anat_orig)
				bids_anat(base_file_id, anat_orig, anat_temp, deface)

		# sbref files
		sbref_origs = sorted(glob.glob(os.path.join(orig_dir,'*sbref*')))[::-1]
		if sbref_origs:
			print('BIDSifying sbref...')
			for sbref_orig in sbref_origs:
				print('\t' + sbref_orig)
				bids_sbref(base_file_id, sbref_orig, func_temp)

		cleanup(anat_temp)
		cleanup(func_temp)
		
		# pe files
		pe_origs = sorted(glob.glob(os.path.join(orig_dir,'*pe*')))[::-1]
		if pe_origs:
			print('BIDSifying pe...')
			for pe_orig in pe_origs:
				print('\t' + pe_orig)
				# note, still using fmap path
				bids_pe(base_file_id, pe_orig, fmap_temp, func_temp)

		# fmap files
		fmap_origs = sorted(glob.glob(os.path.join(orig_dir,'*fieldmap*')))[::-1]
		if fmap_origs:
			print('BIDSifying fmap...')
			for fmap_orig in fmap_origs:
				print('\t' + fmap_orig)
				bids_fmap(base_file_id, fmap_orig, fmap_temp, func_temp)
		
		cleanup(fmap_temp)

		return success
def get_subdir(a_dir):
	return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
		if os.path.isdir(os.path.join(a_dir, name))]

def bids_organizer(
	study_dir, 
	id_correction=None, 
	record=None, 
	bids_dir='/Volumes/svndl/RAW_DATA/MRI_RAW', 
	temp_dir='/Volumes/Denali_4D2/TEMP',
	run_all=False, 
	deface=False):
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
	study_id = study_dir.split('/')[-1]
	# output directory
	out_dir = os.path.join(bids_dir,study_id)
	mkdir(out_dir)
	# directories to BIDSify
	nims_dirs = get_subdir(study_dir)
	if run_all is False:
		nims_dirs = nims_dirs[-1:]
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
	for nims_file in sorted(nims_dirs):
		temp_subj  = get_subj_path(nims_file, temp_dir, id_correction)
		if temp_subj == None:
			print("Couldn't find subj_path for %s" % nims_file)
			continue
		success = bids_subj(nims_file, temp_subj, out_dir, deface)
		# move files
		if success:
			err = rsync(temp_dir, out_dir)
			if err != None:
				success = False
		if success == True and record != None:
			with open(record, 'a') as f:
				f.write(nims_file)  
				f.write('\n')
				print('Successfully transferred %s' % nims_file)
	# remove temp folder
	shutil.rmtree(temp_dir)

	# add physio metadata
	#if not os.path.exists(os.path.join(temp_path, 'recording-cardiac_physio.json')):
	#    if len(glob.glob(os.path.join(temp_path, 'sub-*', 'func', '*cardiac*'))) > 0:
	#        json.dump(cardiac_bids,open(os.path.join(temp_path, 'recording-cardiac_physio.json'),'w'))
	#if not os.path.exists(os.path.join(temp_path, 'recording-respiratory_physio.json')):
	#    if len(glob.glob(os.path.join(temp_path, 'sub-*', 'func', '*respiratory*'))) > 0:
	#        json.dump(respiratory_bids,open(os.path.join(temp_path, 'recording-respiratory_physio.json'),'w'))
	# *****************************
	# *** Cleanup
	# *****************************
	#cleanup(temp_path)
