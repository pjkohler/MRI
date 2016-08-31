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
        
    if args.outdir in "standard":
        args.outdir = "{0}/{1}{2}/{3}".format(args.fsdir,args.subject,suffix,args.outname)
    
    # make temporary, local folder
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    # and subfoldes
    os.mkdir(tmpdir+"/SUMA")
    os.mkdir(tmpdir+"/"+args.outname)
    
    # copy relevant SUMA files
    sumadir = "{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix)
    for file in glob.glob(sumadir+"/*h.smoothwm.asc"):
        shutil.copy(file,tmpdir+"/SUMA")
    for file in glob.glob("{0}/{1}_*.spec".format(sumadir,args.subject)):
        shutil.copy(file,tmpdir+"/SUMA")
    for file in glob.glob("{0}/{1}{2}.std141_to_native.*.niml.M2M".format(sumadir,args.subject,suffix)):
        shutil.copy(file,tmpdir+"/SUMA")
            
    os.chdir(tmpdir)
    for hemi in ["lh","rh"]:
        # if you have a mapping file, this is much faster.  see SurfToSurf -help
        # you can still run without a mapping file, but it is generated on-the-fly (slow!)
        mapfile = "./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(args.subject,suffix,hemi)
        if os.path.isfile(mapfile):
            print("Using existing mapping file {0}".format(mapfile))
            subprocess.call("SurfToSurf -i_fs ./SUMA/{0}.smoothwm.asc -i_fs ./SUMA/std.141.{0}.smoothwm.asc -output_params {1} -mapfile {2} -dset {3}/maxprob_surf_{0}.1D.dset'[1..$]'"
                .format(hemi,args.intertype,mapfile,args.atlasdir), shell=True)
            newmap = False
        else:
            print "Generating new mapping file"
            newmap = True
            subprocess.call("SurfToSurf -i_fs ./SUMA/{0}.smoothwm.asc -i_fs ./SUMA/std.141.{0}.smoothwm.asc -output_params {1} -dset {2}/maxprob_surf_{0}.1D.dset'[1..$]'"
                .format(hemi,args.intertype,args.atlasdir), shell=True)       
            # update M2M file name to be more informative and not conflict across hemispheres
            os.rename("./SurfToSurf.niml.M2M".format(args.outname, hemi), "./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(args.subject,suffix,hemi))
        
        # give output file a more informative name
        os.rename("./SurfToSurf.maxprob_surf_{0}.niml.dset".format(hemi),"./{0}/{1}.{0}.niml.dset".format(args.outname,hemi))
        # we don't need this and it conflicts across hemisphere                    
        os.remove("./SurfToSurf.1D".format(args.outname, hemi))
        
        # make a 1D.dset copy using the naming conventions of other rois,
        # so we can utilize some other script more easily (e.g., roi1_copy_surfrois_locally.sh)
        # mainly for Kastner lab usage
        subprocess.call("ConvertDset -o_1D -input ./{1}/{0}.{1}.niml.dset -prepend_node_index_1D -prefix ./{1}/{0}.{1}.1D.dset"
            .format(hemi, args.outname), shell=True)
        
        if args.doclust: # do optional surface-based clustering
            for idx in range(1,26):
                # clustering steps
                specfile="./SUMA/{0}{1}_{2}.spec".format(args.subject,suffix,hemi)  
                surffile="./SUMA/{1}.smoothwm.asc".format(suffix,hemi)
    
                # isolate ROI
                subprocess.call("3dcalc -a ./{0}/{2}.{0}.niml.dset -expr 'iszero(a-{1})' -prefix {2}.temp.niml.dset"
                    .format(args.outname, idx,hemi), shell=True)
                
                # do clustering, only consider cluster if they are 1 edge apart
                subprocess.call("SurfClust -spec {0} -surf_A {1} -input {2}.temp.niml.dset 0 -rmm -1 -prefix {2}.temp2 -out_fulllist -out_roidset"
                    .format(specfile,surffile,hemi), shell=True)
                    
                # pick only biggest cluster
                if idx is 1:
                    if os.path.isfile("./{0}/{1}.{0}_cluster.niml.dset".format(args.outname,hemi)):
                        print("Removing existing file ./{0}/{1}.{0}_cluster.niml.dset".format(args.outname,hemi)) 
                        os.remove("./{0}/{1}.{0}_cluster.niml.dset".format(args.outname,hemi))
                    subprocess.call("3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -expr 'iszero(a-1)*{2}' -prefix {1}.{0}_cluster.niml.dset"
                        .format(args.outname,hemi,idx), shell=True)
                else:
                    subprocess.call("3dcalc -a {1}.temp2_ClstMsk_e1.niml.dset -b {1}.{0}_cluster.niml.dset -expr 'b+iszero(a-1)*{2}' -prefix {1}.temp3.niml.dset"
                        .format(args.outname,hemi,idx), shell=True)
                    #os.remove("./{0}/{1}.{0}_cluster.niml.dset".format(args.outname, hemi))
                    os.rename("{0}.temp3.niml.dset".format(hemi), "{1}.{0}_cluster.niml.dset".format(args.outname, hemi))
                        
                for file in glob.glob("./*temp*"):
                    os.remove(file)
            # is this step necessary?
            subprocess.call("ConvertDset -input {1}.{0}_cluster.niml.dset -o_niml_asc -prefix {1}.temp4.niml.dset"
                .format(args.outname,hemi,idx), shell=True)
            os.remove("{1}.{0}_cluster.niml.dset".format(args.outname, hemi))
            os.rename("{0}.temp4.niml.dset".format(hemi), "./{0}/{1}.{0}_cluster.niml.dset".format(args.outname, hemi))
        
        # copy mapping file to subjects' home SUMA directory
        if newmap:            
            shutil.move("./SUMA/{0}{1}.std141_to_native.{2}.niml.M2M".format(args.subject,suffix,hemi),
                        "{3}/{0}{1}.std141_to_native.{2}.niml.M2M".format(args.subject,suffix,hemi,sumadir))
    
    os.chdir(curdir)
    if os.path.isdir("{0}".format(args.outdir)):
        print "Output directory {0} exists, adding '_new'".format(args.outdir) 
        shutil.move("{0}/{1}".format(tmpdir,args.outname), "{0}_new".format(args.outdir)) 
    else:
        shutil.move("{0}/{1}".format(tmpdir,args.outname), "{0}".format(args.outdir)) 
    if args.keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir) 
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "############################################################\n"
        "Script for converting atlas ROIs from AFNI/SUMA std141\n"
        "into subjects' native surface space. \n"
        "Current implementation uses ROIs atlas from \n"
        "Wang, Mruczek, Arcaro & Kastner (Cerebral Cortex 2014).\n"
        "For a different atlas, see function mriGlasserConvert.py.\n"
        "Script heavily inspired by shell script from Ryan Mruczek. \n"
        "Requires atlas template, which can be downloaded at: \n"
        "http://www.princeton.edu/~napl/vtpm.htm\n"
        "############################################################"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')") 
    parser.add_argument("-v", "--verbose", action="store_true",
        help="increase output verbosity")    
    parser.add_argument(
        "--atlasdir", metavar="str", type=str,default="{0}/ROI_TEMPLATES/Wang2015/ProbAtlas_v4/subj_surf_all".format(os.environ["SUBJECTS_DIR"]),
         nargs="?", help="Full path to atlas directory \n(default: {fsdir}/ROI_TEMPLATES/Wang2015/ProbAtlas_v4/subj_surf_all)")
    parser.add_argument(
        "--outname", metavar="str", type=str,default="wangatlas",
         nargs="?", help="Output file name \n(default: wangatlas)")    
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
        "--doclust", help="Do optional surface-based clustering of each ROI?  \n(default: on)",
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

