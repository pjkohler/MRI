#!/usr/bin/env python

import argparse, logging, os, subprocess,tempfile, shutil, glob,sys
from os.path import expanduser

def main(args, loglevel):
    #logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    #logging.info("You passed an argument.")
    #logging.debug("Your Argument: %s" % args.file_names)

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
    
    
    if args.std141:
        specprefix = "std.141."
    else:
        specprefix = ""    
    
    # make temporary, local folder
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp("","tmp",expanduser("~/Desktop"))
    # and subfolders
    os.mkdir(tmpdir+"/output")
    
    # copy relevant SUMA files
    sumadir = "{0}/{1}{2}/SUMA".format(args.fsdir,args.subject,suffix)
    print(sumadir)
    for file in glob.glob(sumadir+"/*h.smoothwm.asc"):
        shutil.copy(file,tmpdir)
    for file in glob.glob("{0}/{1}{2}{3}*.spec".format(sumadir,specprefix,args.subject,suffix)):
        shutil.copy(file,tmpdir)

    os.chdir(tmpdir)
    
    print(args.infiles)

    for curfile in args.infiles:
        if ".niml.dset" in curfile:
            filename = curfile[:-10]
            splitString = filename.split(".",1)
            hemi = splitString[0]
            print hemi
            outname = "{0}_{1}fwhm".format(filename,args.blursize)
        else:
            sys.exit("ERROR!\n{0} is not .niml.dset format."
                .format(curfile))
        
        # move files
        subprocess.call("3dcopy {1}/{0}.niml.dset {0}.niml.dset".format(filename,curdir), shell=True)
        # compute mean
        subprocess.call("SurfSmooth -spec {0}{1}{2}_{3}.spec \
                    -surf_A smoothwm -met HEAT_07 -target_fwhm {4} -input {5}.niml.dset \
                    -cmask '-a {5}.niml.dset[0] -expr bool(a)' -output  {6}/{7}.niml.dset"
                    .format(specprefix,args.subject,suffix,hemi,args.blursize,filename,curdir,outname), shell=True)
     
    os.chdir(curdir)    
    if args.keeptemp is not True:
        # remove temporary directory
        shutil.rmtree(tmpdir) 
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        " \n"
        "################################################################\n"
        "Function implementing AFNIs surface-based smoothing.\n"
        "First argument is subject ID, second is data.\n"
        "Input data format should be .niml.dset.\n"
        "Author: pjkohler, Stanford University, 2016                 \n"
        "################################################################\n"
        ,formatter_class=argparse.RawTextHelpFormatter,usage=argparse.SUPPRESS)
    parser.add_argument(
        "subject",type=str,nargs="?", help="Subject ID (without '_fs4')")    
    parser.add_argument(
        "infiles",type=str,nargs="+", help="List of files, in niml.dset format") 
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument(
        "--std141", help="Use std141 version of surface  \n(default: off)", action="store_true")
    parser.add_argument(
        "--blursize", metavar="double", type=int, default=3,nargs="?", help="Blur size (fwhm) default: 3")
    parser.add_argument(
        "--fsdir", metavar="str", type=str,default=os.environ["SUBJECTS_DIR"],
         nargs="?", help="Full path to freesurfer directory  \n(default: as set in environment variable from bash)")
    parser.add_argument(
        "--keeptemp", help="Keep temporary folder \n(default: off)",action="store_true")
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