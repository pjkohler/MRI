#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil, glob,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
    # get current directory    
    curdir = os.getcwd()

    for sub in args.subjects: # loop over list of subjects

        # check if subjects' freesurfer directory exists
        if os.path.isdir("{0}/{1}".format(args.fsdir,sub)):
            # no suffix needed
            suffix=""
        else:
            # suffix needed
            suffix="_fs4"
            if not os.path.isdir("{0}/{1}{2}".format(args.fsdir,sub,suffix)):
                sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
                    .format(args.fsdir,sub,suffix))        
        
        if args.outdir in "standard":
            outdir = "{0}/{1}{2}/{3}".format(args.fsdir,sub,suffix,args.outname)
        else:
            outdir = "{0}_{1}".format(args.outdir,sub) # force sub in name, in case multiple subjects
    
        # make temporary, local folder
        tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
        # and subfoldes
        os.mkdir(tmpdir+"/surf")
        os.mkdir(tmpdir+"/SUMA")
        os.mkdir(tmpdir+"/"+args.outname)
        
        # copy relevant freesurfer files
        surfdir = "{0}/{1}{2}/surf".format(args.fsdir,sub,suffix)
        for file in glob.glob(surfdir+"/*h.white"):
            shutil.copy(file,tmpdir+"/surf")
        
        # copy relevant SUMA files
        sumadir = "{0}/{1}{2}/SUMA".format(args.fsdir,sub,suffix)
        for file in glob.glob(sumadir+"/*h.smoothwm.asc"):
            shutil.copy(file,tmpdir+"/SUMA")
        for file in glob.glob("{0}/{1}_*.spec".format(sumadir,sub)):
            shutil.copy(file,tmpdir+"/SUMA")
        
        os.chdir(tmpdir)
        
        for hemi in ["lh","rh"]:
            idx = 0
            for roi in ["IOG","OTS","mFUS","pFUS","PPA","VWFA1","VWFA2"]:
                idx += 1
                if not os.path.isfile("{1}/{0}.MPM_{2}.label".format(hemi,args.atlasdir,roi)):
                    # if label file does not exist, skip it
                    continue
                # Make the intermediate (subject-native) surface:
                #   --srcsubject is always fsaverage since we assume the input file is an fsaverage file
                #   --trgsubject is the subject we want to convert to
                #   --sval is the file containing the surface data
                #   --hemi is just the hemisphere we want to surf-over
                #   --tval is the output file            
                subprocess.call("mri_label2label --srcsubject fsaverage --trgsubject {2}{3} --regmethod surface --hemi {0} \
                    --srclabel {1}/{0}.MPM_{4}.label --trglabel ./{0}.{4}_TEMP.label".format(hemi,args.atlasdir,sub,suffix,roi), shell=True)
                    
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
                print numnodes
                numnodes = numnodes - 1
                subprocess.call("ConvertDset -o_niml_asc -i_1D -input {0}.{1}_TEMP4.1.1D -prefix {0}.{1}_TEMP4.niml.dset -pad_to_node {2} -node_index_1D {0}.{1}_TEMP4.1.1D[0]"
                    .format(hemi,roi,numnodes), shell=True)

                if idx is 1:
                    subprocess.call("3dcalc -a {0}.{1}_TEMP4.niml.dset -expr 'notzero(a)' -prefix {0}.{2}.niml.dset".format(hemi,roi,args.outname), shell=True)
                    shutil.move("{0}.{1}.niml.dset".format(hemi,args.outname), "./{1}/{0}.{1}.niml.dset".format(hemi,args.outname))
                else:
                    subprocess.call("3dcalc -a {0}.{1}_TEMP4.niml.dset -b ./{2}/{0}.{2}.niml.dset \
                        -expr '(b+notzero(a)*{3})*iszero(and(b,notzero(a)))' -prefix {0}.{1}_TEMP5.niml.dset".format(hemi,roi,args.outname,idx), shell=True)
                    shutil.move("{0}.{1}_TEMP5.niml.dset".format(hemi,roi), "./{1}/{0}.{1}.niml.dset".format(hemi,args.outname))
        
        os.chdir(curdir)
        if os.path.isdir(outdir):
            print "Output directory {0} exists, adding '_new'".format(outdir) 
            shutil.move("{0}/{1}".format(tmpdir,args.outname), "{0}_new".format(outdir)) 
        else:
            shutil.move("{0}/{1}".format(tmpdir,args.outname), "{0}".format(outdir)) 
        if args.keeptemp is not True:
            # remove temporary directory
            shutil.rmtree(tmpdir) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "#############################################################\n"
        "Function for converting atlas ROIs on the ventral surface \n"
        "created by Kevin Weiner and Kalinit Grill-Spector (in press), \n"
        "from freesurfers's fsaverage\n"
        "into subjects' native surface space. \n"
        "For a different atlas, see function mriGlasserConvert.py.\n"
        "\n"
        "Author: pjkohler, Stanford University, 2016     "
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subjects",type=str,nargs="+", help="One or more subject IDs (without '_fs4')") 
    parser.add_argument(
        "-v", "--verbose", action="store_true",help="increase output verbosity")    
    parser.add_argument(
        "--atlasdir", metavar="str", type=str,default="{0}/ROI_TEMPLATES/KGS2016".format(os.environ["SUBJECTS_DIR"]),
         nargs="?", help="Full path to atlas directory \n(default: {SUBJECTS_DIR}/ROI_TEMPLATES/KGS2016)")
    parser.add_argument(
        "--outname", metavar="str", type=str,default="kgs_atlas",
         nargs="?", help="Output file name \n(default: kgs_atlas)")    
    parser.add_argument(
        "--outdir", metavar="str", type=str,default="standard",
         nargs="?", help="Full path to output directory  \n(default: {fsdir}/{subject ID}/{outname}/)")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],
         nargs="?", help="Full path to freesurfer directory  \n(default: as set in environment variable from bash)")
    parser.add_argument(
        "--forcex", help="Force xhemi registration  \n(default: off)",
        action="store_true")
    parser.add_argument(
        "--separate_out", help="Make separate output files  \n(default: off)",
        action="store_true")
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder? (default: off)",
        action="store_true")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

main(args, loglevel)
    