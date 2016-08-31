#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
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
        # compute mean
        subprocess.call("3dTstat -prefix mean_{0}+orig {0}+orig".format(filename), shell=True)
        # scale
        if args.no_dt:
            # save scaled data in data folder directly
            subprocess.call("3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {1}/{0}.sc.nii.gz"
                .format(filename,curdir), shell=True)
        else:
            # scale, then detrend and store in data folder
            subprocess.call("3dcalc -float -a {0}+orig -b mean_{0}+orig -expr 'min(200, a/b*100)*step(a)*step(b)' -prefix {0}.sc+orig"
                .format(filename), shell=True)
            subprocess.call("3dDetrend -prefix {1}/{0}.sc.dt.nii.gz -polort 2 -vector {1}/motparam.{0}.1D {0}.sc+orig"
                .format(filename,curdir), shell=True)
                
    os.chdir(curdir)    
    if args.keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir) 
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "############################################################\n"
        "Script for third stage of preprocessing: Scaling and Detrending.\n"
        "Typically run following mriPre.py and mriVolreg.py \n"
        "Detrending currently requires motion registration parameters\n"
        "as .1D files: motparam.xxx.1D \n"
        "############################################################"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "infiles",type=str,
        nargs="+", help="List of EPI files") 
    parser.add_argument("-v", "--verbose", action="store_true",
        help="increase output verbosity")    
    parser.add_argument(
        "--no_dt", help="Skip detrend \n(default: off)",
        action="store_true")    
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder \n(default: off)",
        action="store_true")

    args = parser.parse_args()
    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

main(args, loglevel)