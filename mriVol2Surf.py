#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil, glob,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
    # get current directory    
    curdir = os.getcwd()

    # make temporary, local folder
    tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))   
    
    # check if subjects' freesurfer directory exists
    if os.path.isdir("{0}/{1}".format(args.fsdir,args.subject)):
        # no suffix needed
        suffix=""
    else:
        # suffix needed
        suffix="_fs4"
        if not os.path.isdir("{0}/{1}{2}".format(args.fsdir,args.subject,suffix)):
            sys.exit("ERROR!\nSubject folder {0}/{1} \ndoes not exist, without or with suffix '{2}'."
            .format(args.fsdir,args.subject,suffix))
    
    if args.wm_mod is not 0.0 or args.gm_mod is not 0.0:
        # for gm, positive values makes the distance longer, for wm negative values
        args.steps = round(args.steps + args.steps * args.gm_mod - args.steps * args.wm_mod)
    
    print "MAPPING: WMOD: {0} GMOD: {1} STEPS: {2}".format(args.wm_mod,args.gm_mod,args.steps)

    if args.surfvol is "standard":
        voldir = "{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix) 
        volfile = "{0}{1}_SurfVol+orig".format(args.subject,suffix)
    else:
        voldir = '/'.join(args.surfvol.split('/')[0:-1])
        volfile = args.surfvol.split('/')[-1]
        if not voldir: # if volume directory is empty
            voldir = curdir
    
    # make temporary copy of volume file     
    subprocess.call("3dcopy {0}/{1} {2}/{1}"
        .format(voldir,volfile,tmpdir,volfile), shell=True)
    
    # now get specfiles
    if args.prefix is not ".":
        args.prefix = ".{0}.".format(args.prefix)    
    
    if args.std141:
        specprefix = "std.141."
        args.prefix = ".std.141{0}".format(args.prefix)
    else:
        specprefix = ""    
    
    sumadir = "{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix)
    for file in glob.glob(sumadir+"/*h.smoothwm.asc"):
            shutil.copy(file,tmpdir)
    for file in glob.glob(sumadir+"/*h.pial.asc"):
            shutil.copy(file,tmpdir)
    for file in glob.glob("{0}/{1}{2}{3}*.spec".format(sumadir,specprefix,args.subject,suffix)):
        shutil.copy(file,tmpdir)
    for file in glob.glob(sumadir+"/*aparc.a2009s.annot.niml.dset"):
        shutil.copy(file,tmpdir)
    
    os.chdir(tmpdir)
    for curfile in args.infiles:
        if ".nii.gz" in curfile:
            filename = curfile[:-7]
            filesuffix = ".nii.gz"
        elif ".nii" in curfile:
            filename = curfile[:-4]
            filesuffix = ".nii"
        elif "+orig" in curfile:
            split = curfile.rpartition("+")            
            filename = split[0]
            filesuffix = "".join(split[1:])
        
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(filename,curdir,filesuffix), shell=True)
            
        for hemi in ["lh","rh"]:
            if args.mask is 'data':
                if ".nii.gz" in args.mask:
                    maskname = args.mask[:-7]
                    masksuffix = ".nii.gz"
                elif ".nii" in args.mask:
                    maskname = args.mask[:-4]
                    masksuffix = ".nii"
                elif "+orig" in args.mask:
                    split = args.mask.rpartition("+")            
                    maskname = split[0]
                    masksuffix = "".join(split[1:])
                # copy mask
                subprocess.call("3dcopy {1}/{0}{2} mask+orig".format(maskname,curdir,masksuffix), shell=True)
                maskcode = "-cmask '-a mask+orig[0] -expr notzero(a)' "
            else:
                maskcode = ""
            subprocess.call("3dVol2Surf -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -surf_B pial -sv {4} -grid_parent {5}+orig -map_func {6} \
                    -f_index {7} -f_p1_fr {8} -f_pn_fr {9} -f_steps {10} \
                    -outcols_NSD_format -oob_value -0 {13}-out_niml {11}/{3}{12}{5}.niml.dset"
                    .format(specprefix,args.subject,suffix,hemi,volfile,filename,args.mapfunc,args.index,args.wm_mod,args.gm_mod,args.steps,curdir,args.prefix,maskcode), shell=True)
    
    os.chdir(curdir)    
    #if args.keeptemp is not True:
    #    # remove temporary directory
    #    shutil.rmtree(tmpdir) 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "#############################################################\n"
        "Function for converting from volume to surface space.  \n" 
        "Supports suma surfaces both in native and std141 space.\n"
        "Surfave volume can be given using the --surf_vol argument.\n"
        "Various other options from 3dVol2Surf are implemented, \n"
        "sometimes names that are more meaningful (to me).\n"
        "'data' option for mask still needs to be implemented. \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                  \n"
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')")
    parser.add_argument(
        "infiles",type=str,nargs="+", help="List of EPI files") 
    parser.add_argument(
        "-v", "--verbose", action="store_true",help="increase output verbosity")    
    parser.add_argument(
        "--mapfunc", metavar="str", type=str,default="ave",nargs="?", help="Mapping function (ave, median, max etc) \n(default: ave")
    parser.add_argument(
        "--wm_mod", metavar="float", type=float,default=0.0,nargs="?", help="Amount to modulate wm boundary, as a fraction \nof total wm-gm distance.\nNegative values imply moving the boundary \ntowards brain center, positive values towards skull \n(default: 0.0)")    
    parser.add_argument(
        "--gm_mod", metavar="float", type=float,default=0.0,nargs="?", help="Amount to modulate gm boundary, as a fraction \nof total wm-gm distance.\nNegative values imply moving the boundary \ntowards brain center, positive values towards skull \n(default: 0.0)")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],nargs="?", help="Full path to freesurfer directory  \n(default: as set in environment variable from bash)")
    parser.add_argument(
        "--surfvol", metavar="str", type=str,default="standard", nargs="?", help="Surface volume file to use \n(default: SurfVol+orig in SUMA folder)")
    parser.add_argument(
        "--prefix", metavar="str", type=str,default=".", nargs="?", help="Prefix to append to input file names \n(default: nothing)")
    parser.add_argument(
        "--index", metavar="str", type=str,default='voxels', nargs="?", help="-f_index value given to 3dVol2Surf \n(default: voxels)")
    parser.add_argument(
        "--steps", metavar="int", type=int,default=10, nargs="?", help="-f_steps value given to 3dVol2Surf \n(default: 10 + any wm or gm extension")
    parser.add_argument(
        "--mask", metavar="str", type=str,default='off', nargs="?", help="Can either be no masking ('off'), \nmask by nonzero values in each sub-brick of the data ('data')\nor in a different file (path to file) \n(default: off)")
    parser.add_argument(
        "--std141", help="Use std141 version of surface  \n(default: off)", action="store_true")
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
    