#!/usr/bin/env python

import argparse, sys, mymri
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "###########################################################\n"
        "Function for first stage of preprocessing\n"
        "Slice-time correction and deobliqueing, in that order.\n"
        "Also supports data with different number of slices,\n"
        "and padding of the matrices, via flags \n"
        "--diffmat, --pad_ap and --pad_is. \n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                \n"
        "###########################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "infiles",type=str,
        nargs="+", help="List of EPI files")
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
        "--tfile", metavar="str", type=str,default=None,
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
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

mymri.Pre(args.infiles, args.ref, args.trdur, args.pre_tr, args.total_tr, args.tfile, args.pad_ap, args.pad_is, args.diffmat, args.keeptemp)
