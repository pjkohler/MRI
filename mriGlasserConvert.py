#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil, glob,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
    # assign remaining defaults
    if args.no_fs4:
        suffix=""
    else:
        suffix="_fs4"
    for sub in args.subjects: # loop over list of subjects
        
        if args.outdir in "standard":
            outdir = "{0}/{1}{2}/{3}".format(args.fsdir,sub,suffix,args.outname)
        else:
            outdir = "{0}_{1}".format(args.outdir,sub) # force sub in name, in case multiple subjects
    
        # make temporary, local folder
        curdir = os.getcwd()
        tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
        # and subfoldes
        os.mkdir(tmpdir+"/surf")
        os.mkdir(tmpdir+"/"+args.outname)
        
        # copy relevant freesurfer files
        surfdir = "{0}/{1}{2}/surf".format(args.fsdir,sub,suffix)
        for file in glob.glob(surfdir+"/*h.white"):
            shutil.copy(file,tmpdir+"/surf")
            
        os.chdir(tmpdir)

        for hemi in ["lh","rh"]:
            # convert from .annot to mgz
            subprocess.call("mri_annotation2label --subject fsaverage --hemi {0} --annotation {1}/{0}.HCPMMP1.annot --seg {0}.glassertemp1.mgz"
                .format(hemi,args.atlasdir), shell=True)
            # convert to subjects native space
            subprocess.call("mri_surf2surf --srcsubject fsaverage --trgsubject {2}{3} --sval {0}.glassertemp1.mgz --hemi {0} --tval ./{1}/{0}.{1}.mgz"
                .format(hemi,args.outname,sub,suffix), shell=True)
            # convert mgz to gii
            subprocess.call("mris_convert -f ./{1}/{0}.{1}.mgz ./surf/{0}.white ./{1}/{0}.{1}.gii"
                .format(hemi,args.outname), shell=True)
            # convert gii to niml.dset
            subprocess.call("ConvertDset -o_niml_asc -input ./{1}/{0}.{1}.gii -prefix ./{1}/{0}.{1}.niml.dset"
                .format(hemi,args.outname), shell=True)
            
        os.chdir(curdir)
        if os.path.isdir("{0}".format(outdir)):
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
        "############################################################\n"
        "Script for converting atlas ROIs from freesurfers's fsaverage\n"
        "into subjects' native surface space. \n"
        "Current implementation uses ROIs atlas from Glasser et al. \n"
        "(Nature, 2016).\n"
        "Requires atlas template, which can be downloaded at:\n"
        "https://balsa.wustl.edu/study/show/RVVG\n"
        "############################################################"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subjects",type=str,nargs="+", help="One or more subject IDs (without '_fs4')") 
    parser.add_argument(
        "-v", "--verbose", action="store_true",help="increase output verbosity")    
    parser.add_argument(
        "--atlasdir", metavar="str", type=str,default="{0}/ROI_TEMPLATES/Glasser2016".format(os.environ["SUBJECTS_DIR"]),
         nargs="?", help="Full path to atlas directory \n(default: {SUBJECTS_DIR}/ROI_TEMPLATES/Glasser2016)")
    parser.add_argument(
        "--outname", metavar="str", type=str,default="glass_atlas",
         nargs="?", help="Output file name \n(default: glass_atlas)")    
    parser.add_argument(
        "--outdir", metavar="str", type=str,default="standard",
         nargs="?", help="Full path to output directory  \n(default: {fsdir}/{subject ID}/{outname}/)")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],
         nargs="?", help="Full path to output directory  \n(default: as set in environment variable from bash)")
    parser.add_argument(
        "--intertype", metavar="str", type=str,default='NearestNode',
         nargs="?", help="Interpolation type?  \n(default: NearestNode)")
    parser.add_argument(
        "--do_clust", help="Do optional surface-based clustering of each ROI?  \n(default: on)",
        action="store_true")    
    parser.add_argument(
        "--no_fs4", help="Auto add SVNDL-style '_fs4' suffix to subject ID?  \n(default: on)",
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
    
    