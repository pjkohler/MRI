#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil, glob,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)

    # get current directory    
    curdir = os.getcwd()    
    
    # check if subjects' SUMA directory exists
    if os.path.isdir("{0}/{1}/SUMA".format(args.fsdir,args.subject)):
        # no suffix needed
        suffix=""
    else:
        # suffix needed
        suffix="_fs4"
        if not os.path.isdir("{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix)):
            sys.exit("ERROR!\nSubject SUMA folder {0}/{1}/SUMA \ndoes not exist, without or with suffix '{2}'."
                .format(args.fsdir,args.subject,suffix))
    
    sumadir = "{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix)
    if args.std141:
        specfile="{0}/std.141.{1}{2}_{3}.spec".format(sumadir,args.subject,suffix,args.hemi)
    else:
        specfile="{0}/{1}{2}_{3}.spec".format(sumadir,args.subject,suffix,args.hemi)
    
    if args.volpath is "standard":
        args.volpath = "{0}/{1}{2}/SUMA/{1}{2}_SurfVol+orig".format(args.fsdir,args.subject,suffix)
    else:
        # if volpath was assigned, assume user wants to open volume
        args.openvol = True

    if args.openvol:        
        voldir = '/'.join(args.volpath.split('/')[0:-1])
        volfile = args.volpath.split('/')[-1]        
        if voldir: # if volume directory is not empty
             os.chdir(voldir)
        subprocess.call("afni -niml & SUMA -spec {0} -sv {1} &"
            .format(specfile,volfile), shell=True)
    else:
        subprocess.call("SUMA -spec {0} &"
            .format(specfile), shell=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "###############################################################\n"
        "Wrapper function for easy opening of SUMA viewer.\n"
        "Supports opening suma surfaces both in native and std141 space.\n"
        "Supports opening a volume file in afni, linked to the surfaces,\n"
        "via the --openvol and --volpath options. If volpath is given,  \n"
        "openvol will be assumed. Note that when a volume file is given,\n"
        "script will change to its directory.                           \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                    \n"
        "###############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')") 
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity")    
    parser.add_argument(
        "--hemi", metavar="both/lh/rh", type=str,default="both",nargs="?", help="Which hemisphere? default: both")
    parser.add_argument(
        "--openvol", help="Open volume in AFNI concurrently  \n(default: off)", action="store_true")  
    parser.add_argument(
        "--volpath", metavar="str", type=str,default="standard", nargs="?", help="Full path to specific volume file to open \n(default: SurfVol+orig in SUMA folder)")
    parser.add_argument(
        "--std141", help="Open std141 version of surface  \n(default: off)", action="store_true")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],
         nargs="?", help="Full path to output directory  \n(default: as set in environment variable from bash)")
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