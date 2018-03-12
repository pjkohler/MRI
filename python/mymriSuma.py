#!/usr/bin/env python

import argparse, sys, mymri    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "###############################################################\n"
        "Wrapper function for easy opening of SUMA viewer.\n"
        "Supports opening suma surfaces both in native and std141 space.\n"
        "Supports opening a volume file in afni, linked to the surfaces,\n"
        "via the --openvol and --surfvol options. If surfvol is given,  \n"
        "openvol will be assumed. Note that when a volume file is given,\n"
        "script will change to its directory.                           \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                    \n"
        "###############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')") 
    parser.add_argument(
        "--hemi", metavar="both/lh/rh", type=str,default="both",nargs="?", help="Which hemisphere? default: both")
    parser.add_argument(
        "--openvol", help="Open volume in AFNI concurrently  \n(default: off)", action="store_true")  
    parser.add_argument(
        "--surfvol", metavar="str", type=str,default="standard", nargs="?", 
        help="Full path to specific volume file to open \n(default: SurfVol+orig in SUMA folder)")
    parser.add_argument(
        "--std141", help="Open std141 version of surface  \n(default: off)", action="store_true")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=None,
         nargs="?", help="Full path to output directory  \n(default: as set in environment variable from bash)")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

mymri.Suma(args.subject, args.hemi, args.openvol, args.surfvol, args.std141, args.fsdir)