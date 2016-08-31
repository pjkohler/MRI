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
        os.mkdir(tmpdir+"/"+args.outname)
        
        # copy relevant freesurfer files
        surfdir = "{0}/{1}{2}/surf".format(args.fsdir,sub,suffix)
        for file in glob.glob(surfdir+"/*h.white"):
            shutil.copy(file,tmpdir+"/surf")
        
        os.chdir(tmpdir)
        
        if os.path.isdir(surfdir+"/../xhemi") is False or args.forcex is True:
            # register lh to fsaverage sym
            subprocess.call("surfreg --s {0}{1} --t fsaverage_sym --lh"
                .format(sub,suffix), shell=True)
            
            # mirror-reverse subject rh and register to lh fsaverage_sym
            # though the right hemisphere is not explicitly listed above, it is implied by --lh --xhemi
            subprocess.call("surfreg --s {0}{1} --t fsaverage_sym --lh --xhemi"
                .format(sub,suffix), shell=True)
        else:
            print "Skipping registration"
                    

        if args.separate_out:
            datalist = ["angle", "eccen", "areas", "all"]
        else:
            datalist = ["all"]
            
        
        for bdata in datalist:

            # resample right and left hemisphere data to symmetric hemisphere
            subprocess.call("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1} --trgsurfreg {2}.sphere.reg \
                --hemi lh --sval {3}/{4}-template-2.5.sym.mgh --tval ./{5}/lh.{4}.{5}.mgh"
                .format(sub,suffix,"fsaverage_sym",args.atlasdir,bdata,args.outname), shell=True)
            subprocess.call("mri_surf2surf --srcsubject {2} --srcsurfreg sphere.reg --trgsubject {0}{1}/xhemi --trgsurfreg {2}.sphere.reg \
                --hemi lh --sval {3}/{4}-template-2.5.sym.mgh --tval ./{5}/rh.{4}.{5}.mgh"                
                .format(sub,suffix,"fsaverage_sym",args.atlasdir,bdata,args.outname), shell=True)
        	
            # convert to suma
            for hemi in ["lh","rh"]:
                subprocess.call("mris_convert -f ./{0}/{1}.{2}.{0}.mgh ./surf/{1}.white ./{0}/{1}.{2}.{0}.gii"
                    .format(args.outname,hemi,bdata,suffix), shell=True)
                subprocess.call("ConvertDset -o_niml_asc -input ./{0}/{1}.{2}.{0}.gii -prefix ./{0}/{1}.{2}.{0}..niml.dset"
                    .format(args.outname,hemi,bdata,suffix), shell=True)
        
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
        "Script for generating V1-V3 ROIs in subject's native space,  \n" 
        "predicted from the cortical surface anatomy\n"
        "as described in Benson et al. (PLoS Comput Biol., 2014).\n"
        "Requires template data, which can be downloaded at:\n"
        "https://cfn.upenn.edu/aguirre/wiki/public:retinotopy_template\n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                  \n"
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subjects",type=str,nargs="+", help="One or more subject IDs (without '_fs4')") 
    parser.add_argument(
        "-v", "--verbose", action="store_true",help="increase output verbosity")    
    parser.add_argument(
        "--atlasdir", metavar="str", type=str,default="{0}/ROI_TEMPLATES/Benson2014".format(os.environ["SUBJECTS_DIR"]),
         nargs="?", help="Full path to atlas directory \n(default: {SUBJECTS_DIR}/ROI_TEMPLATES/Benson2014)")
    parser.add_argument(
        "--outname", metavar="str", type=str,default="benson_atlas",
         nargs="?", help="Output file name \n(default: benson_atlas)")    
    parser.add_argument(
        "--outdir", metavar="str", type=str,default="standard",
         nargs="?", help="Full path to output directory  \n(default: {fsdir}/{subject ID}/{outname}/)")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],
         nargs="?", help="Full path to output directory  \n(default: as set in environment variable from bash)")
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
    