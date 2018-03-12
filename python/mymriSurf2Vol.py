#!/usr/bin/env python

import argparse, sys, mymri

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "#############################################################\n"
        "Function for converting from surface to volume space.  \n" 
        "Supports suma surfaces both in native and std141 space.\n"
        "Surfave volume can be given using the --surf_vol argument.\n"
        "Various other options from 3dSurf2Vol are implemented, \n"
        "sometimes with names that are more meaningful (to me).\n"
        "\n"
        "Author: pjkohler, Stanford University, 2016                  \n"
        "#############################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')")
    parser.add_argument(
        "infiles",type=str,nargs="+", help="List of EPI files ('.niml.dset' or 'niml.roi')")  
    parser.add_argument(
        "--mapfunc", metavar="str", type=str,default="ave",nargs="?", help="Mapping function (ave, median, max etc) \n(default: ave")
    parser.add_argument(
        "--wm_mod", metavar="float", type=float,default=0.0,nargs="?", help="Amount to modulate wm boundary, as a fraction \nof total wm-gm distance.\nNegative values imply moving the boundary \ntowards brain center, positive values towards skull \n(default: 0.0)")    
    parser.add_argument(
        "--gm_mod", metavar="float", type=float,default=0.0,nargs="?", help="Amount to modulate gm boundary, as a fraction \nof total wm-gm distance.\nNegative values imply moving the boundary \ntowards brain center, positive values towards skull \n(default: 0.0)")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=None,nargs="?", help="Full path to freesurfer directory  \n(default: as set in environment variable from bash)")
    parser.add_argument(
        "--outdir", metavar="str", type=str,default=None,nargs="?", help="Output directory  \n(default: current directory)")
    parser.add_argument(
        "--surfvol", metavar="str", type=str,default="standard", nargs="?", help="Surface volume file to use \n(default: SurfVol+orig in SUMA folder)")
    parser.add_argument(
        "--prefix", metavar="str", type=str,default=".", nargs="?", help="Prefix to append to input file names \n(default: nothing)")
    parser.add_argument(
        "--index", metavar="str", type=str,default='voxels', nargs="?", help="-f_index value given to 3dVol2Surf \n(default: voxels)")
    parser.add_argument(
        "--steps", metavar="int", type=int,default=10, nargs="?", help="-f_steps value given to 3dVol2Surf \n(default: 10 + any wm or gm extension)")
    parser.add_argument(
        "--std141", help="Use std141 version of surface  \n(default: off)", action="store_true")
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder? (default: off)",
        action="store_true")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

mymri.Surf2Vol(
    subject=args.subject, 
    in_files=args.infiles, 
    map_func=args.mapfunc, 
    wm_mod=args.wm_mod, 
    gm_mod=args.gm_mod, 
    prefix=args.prefix, 
    index=args.index, 
    steps=args.steps, 
    out_dir=args.outdir, 
    fs_dir=args.fsdir, 
    surf_vol=args.surf_vol, 
    std141=args.std141, 
    keep_temp=args.keeptemp)