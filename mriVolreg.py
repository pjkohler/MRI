#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
    # assign remaining defaults
    if args.ref in "last":
        args.ref = args.infiles[-1] # use last as reference

    # make temporary, local folder
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    os.chdir(tmpdir)
    
    for curfile in args.infiles:
        if ".nii.gz" in curfile:
            filename = curfile[:-7]
            suffix = ".nii.gz"
        elif ".nii" in curfile:
            filename = curfile[:-4]
            suffix = ".nii"
        elif "+orig" in curfile:
            split = curfile.rpartition("+")            
            filename = split[0]
            s=""
            suffix = s.join(split[1:])
        
        # move files
        subprocess.call("3dcopy {1}/{0}{2} {0}+orig".format(filename,curdir,suffix), shell=True)
        
        # do volume registration
        if args.slow:
            subprocess.call("3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -heptic -twopass -maxite 50 {0}+orig"
                .format(filename,args.ref,curdir), shell=True)
        else:
            subprocess.call("3dvolreg -verbose -zpad 1 -base {2}/{1}''[0]'' -1Dfile {2}/motparam.{0}.vr.1D -prefix {2}/{0}.vr.nii.gz -Fourier {0}+orig"
                .format(filename,args.ref,curdir), shell=True)
    
    os.chdir(curdir)    
    if args.keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir) 
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "#############################################################\n"
        "Script for second stage of preprocessing: Volume registation.\n"
        "Typically run following mriPre.py\n"
        "See option --slow for difficult cases \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                  \n"
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument("infiles",
        type=str,
        nargs="+", help="List of EPI files") 
    parser.add_argument("-v", "--verbose", action="store_true",
        help="increase output verbosity")    
    parser.add_argument(
        "--ref", metavar="str", type=str,default="last",
         nargs="?", help="Name of reference EPI to align to \n(default: last given)")    
    parser.add_argument(
        "--slow", help="Do slow volume registration (difficult files)? \n(default: no)",
        action="store_true")
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder? \n(default: off)",
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