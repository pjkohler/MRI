#!/usr/bin/env python

import argparse, sys
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "################################################################\n"
        "Function for third stage of preprocessing: Scaling and Detrending.\n"
        "Typically run following mriPre.py and mriVolreg.py \n"
        "Detrending currently requires motion registration parameters\n"
        "as .1D files: motparam.xxx.1D \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                 \n"
        "################################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "infiles",type=str,
        nargs="+", help="List of EPI files") 
    parser.add_argument(
        "--no_dt", help="Skip detrend \n(default: don't skip)",
        action="store_true")    
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder \n(default: off)",
        action="store_true")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

from mymri import scale

scale(in_files=args.infiles, no_dt=args.no_dt, keep_temp=args.keeptemp)