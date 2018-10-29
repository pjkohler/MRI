#!/usr/bin/env python

import argparse, sys
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "#############################################################\n"
        "Function for second stage of preprocessing: Volume registation.\n"
        "Typically run following mriPre.py\n"
        "Use option --slow for difficult cases \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                  \n"
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument("infiles",
        type=str,
        nargs="+", help="List of EPI files")  
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

from mymri import vol_reg

mymri.vol_reg(args.infiles, ref_file=args.ref, slow=args.slow, keep_temp=args.keeptemp)