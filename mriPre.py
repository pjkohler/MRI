#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)
    
    # assign remaining defaults
    if args.ref in "last":
        args.ref = args.infiles[-1] # use last as reference
    if args.trdur is 0:
        # TR not given, so compute
        args.trdur = subprocess.check_output("3dinfo -tr -short {0}".format(args.ref), shell=True)
        args.trdur = args.trdur.rstrip("\n")
    if args.total_tr is 0:
        # include all TRs, so get max subbrick value
        args.total_tr = subprocess.check_output("3dinfo -nvi -short {0}".format(args.ref), shell=True)
        args.total_tr = args.total_tr.rstrip("\n")
    else:
        # subject has given total number of TRs to include, add preTR to that
        args.total_tr = eval("args.pre_tr + args.total_tr")

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
        
        # crop and move files
        subprocess.call("3dTcat -prefix {0}+orig {1}/{0}{2}''[{3}..{4}]''"
            .format(filename,curdir,suffix,args.pre_tr,args.total_tr), shell=True)
        
        # slice timing correction
        if args.tfile is "none":
            subprocess.call("3dTshift -quintic -prefix {0}.ts+orig -TR {1}s -tzero 0 -tpattern alt+z {0}+orig".format(filename,args.trdur), shell=True)
        else:
            subprocess.call("3dTshift -quintic -prefix {0}/{1}.ts+orig -TR {2}s -tzero 0 -tpattern @{3} {0}/{1}+orig"
                .format(tmpdir,filename,args.trdur,args.tfile), shell=True)
            
        # deoblique
        subprocess.call("3dWarp -deoblique -prefix {0}.ts.do+orig {0}.ts+orig".format(filename), shell=True)
        
        # pad 
        if args.pad_ap is not 0 or args.pad_is is not 0:        
            subprocess.call("3dZeropad -A {1} -P {1} -I {2} -S {2} -prefix {0}.ts.do.pad+orig {0}.ts.do+orig".format(filename,args.pad_ap,args.pad_is), shell=True)
            subprocess.call("rm {0}.ts.do+orig*".format(filename), shell=True)
            subprocess.call("3dRename {0}.ts.do.pad+orig {0}.ts.do+orig".format(filename), shell=True)
                
        # take care of different matrices, and move files back        
        if args.diffmat:
            subprocess.call("@Align_Centers -base {1} {0}.ts.do+orig".format(filename,args.ref), shell=True)
            subprocess.call("3dresample -master {1} -prefix {0}.ts.do.rs+orig -inset {0}.ts.do_shft+orig".format(filename,args.ref), shell=True)
            subprocess.call("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.rs.nii.gz {0}.ts.do.rs+orig".format(filename,curdir), shell=True)    
        else:
            subprocess.call("3dAFNItoNIFTI -prefix {1}/{0}.ts.do.nii.gz {0}.ts.do+orig".format(filename,curdir), shell=True)
    
    os.chdir(curdir)    
    if args.keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir) 
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "############################################################\n"
        "Script for first stage of preprocessing\n"
        "Slice-time correction and deobliqueing, in that order.\n"
        "Also supports data with different number of slices,\n"
        "and padding of the matrices, via flags \n"
        "--diffmat, --pad_ap and --pad_is. \n"
        "############################################################"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "infiles",type=str,
        nargs="+", help="List of EPI files") 
    parser.add_argument("-v", "--verbose", action="store_true",
        help="increase output verbosity")    
    parser.add_argument(
        "--ref", metavar="str", type=str,default="last",
         nargs="?", help="Name of reference EPI to align to \n(default: last given)")    
    parser.add_argument(
        "--trdur", metavar="double", type=float, default=0,
         nargs="?", help="TR duration in seconds \n(default: compute automatically)")
    parser.add_argument(
        "--pre_tr", metavar="int", type=int, default=0,
         nargs="?", help="How many acquisitions to remove \n(default=0)")
    parser.add_argument(
        "--total_tr", metavar="double", type=int, default=0,
         nargs="?", help="total number of acquisitions to keep (default: all)")
    parser.add_argument(
        "--diffmat", help="EPIs with different matrices? \n(default: no)",
        action="store_true")
    parser.add_argument(
        "--tfile", metavar="str", type=str,default="none",
         nargs="?", help="Full path to time shift file \n(default: no file, odd-even interleaved") 
    parser.add_argument(
        "--pad_ap", metavar="int", type=int, default=0,
         nargs="?", help="Number of voxels to pad matrices in a/p direction \n(default: 0)")
    parser.add_argument(
        "--pad_is", metavar="int", type=int, default=0,
         nargs="?", help="Number of voxels to pad matrices in i/s direction \n(default: 0)")
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder? \n(default: off)",
        action="store_true")

    args = parser.parse_args()
    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

main(args, loglevel)