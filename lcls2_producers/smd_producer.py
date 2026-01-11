#!/usr/bin/env python

import numpy as np
import json
import _xtcpp
import time
from datetime import datetime

begin_job_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
start_job = time.time()
import argparse
import socket
import os
import logging
import requests
import sys
from glob import glob
from requests.auth import HTTPBasicAuth
from pathlib import Path
from importlib import import_module
from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
log_level = "INFO"
# log_level = DEBUG
log_format = "[ %(asctime)s | %(levelname)-3s | %(filename)s] %(message)s"
logging.basicConfig(format=log_format)
logger.setLevel(
    logging.INFO
)  # Set level here instead of in the basic config so other loggers are  not affected

if rank == 0:
    logger.info(f"MPI size: {size}")
    if os.getenv("CONDA_DEFAULT_ENV") is not None:
        logger.info(
            "psana conda environment is {0}".format(os.environ["CONDA_DEFAULT_ENV"])
        )
    else:
        spack_env = os.getenv("SPACK_ENV")
        if spack_env is not None:
            logger.info(f"psana spack environment is {spack_env}")
        else:
            logger.info("Could not determine what psana environment is in use.")


# -----------------------------
# xtcpp_pack helpers
# -----------------------------
def _is_scalar_number(x):
    return isinstance(x, (int, float, np.integer, np.floating, bool))


def _to_numpy_1d(x):
    """Convert x into a 1D numpy array (numeric) suitable for the pybind writer."""
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            # object arrays are dangerous for pybind/std::any; try best-effort cast
            try:
                x = x.astype(np.float32)
            except Exception:
                # fall back to bytes of repr
                x = np.array([repr(v) for v in x.ravel()], dtype="S")
        return np.asarray(x).ravel()
    if isinstance(x, np.ma.MaskedArray):
        return np.asarray(x.filled()).ravel()
    if _is_scalar_number(x):
        return np.asarray([x], dtype=np.float32)
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(bytes(x), dtype=np.uint8)
    if isinstance(x, str):
        # store as fixed-width bytes (1D array)
        return np.asarray([x.encode("utf-8")], dtype="S")
    if isinstance(x, (list, tuple)):
        # Try numeric first
        try:
            arr = np.asarray(x)
            if arr.dtype.kind in ("i", "u", "f", "b"):
                return arr.ravel()
        except Exception:
            pass
        # Fall back to string bytes
        return np.asarray([repr(v).encode("utf-8") for v in x], dtype="S")
    # Unknown type: store repr as bytes
    return np.asarray([repr(x).encode("utf-8")], dtype="S")


def xtcpp_pack(data_dict, prefix=""):
    """
    Pack an arbitrarily nested dict into (flat_data, shape_dict) expected by xtcpp.
    - flat_data: dict[str, np.ndarray] (1D arrays)
    - shape_dict: dict[str, list[int]] original shapes (empty list for scalars)
    Keys are joined with '/'.
    """
    flat = {}
    shapes = {}

    def _walk(obj, pfx):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{pfx}/{k}" if pfx else str(k)
                _walk(v, key)
            return

        arr = _to_numpy_1d(obj)
        flat[pfx] = arr

        # For shape: preserve original ndarray shape; scalars -> []
        if isinstance(obj, np.ndarray):
            shapes[pfx] = list(obj.shape)
        elif isinstance(obj, np.ma.MaskedArray):
            shapes[pfx] = list(np.asarray(obj).shape)
        elif _is_scalar_number(obj) or isinstance(obj, (str, bytes, bytearray)):
            shapes[pfx] = []
        elif isinstance(obj, (list, tuple)):
            # If list-of-numbers treat as 1D length
            try:
                arr2 = np.asarray(obj)
                if arr2.dtype.kind in ("i", "u", "f", "b"):
                    shapes[pfx] = [len(arr2)]
                else:
                    shapes[pfx] = [len(obj)]
            except Exception:
                shapes[pfx] = [len(obj)]
        else:
            shapes[pfx] = []

    _walk(data_dict, prefix)
    return flat, shapes


# DEFINE DETECTOR AND ADD ANALYSIS FUNCTIONS
def define_dets(run, det_list):
    # Load DetObjectFunc parameters (if defined)
    # Assumes that the config file with function parameters definition
    # has been imported under "config"

    rois_args = []  # special implementation as a list to support multiple ROIs.
    dimgs_args = {}
    wfs_int_args = {}
    wfs_hitfinder_args = {}
    d2p_args = {}
    wfs_svd_args = {}
    droplet_args = {}
    azav_args = {}
    azav_pyfai_args = {}
    polynomial_args = {}
    sum_algo_args = {}
    pressio_compression_args = {}

    # Get the functions arguments from the production config
    if "getROIs" in dir(config):
        rois_args = config.getROIs(run)
    if "getDetImages" in dir(config):
        dimgs_args = config.getDetImages(run)
    if "get_wf_integrate" in dir(config):
        wfs_int_args = config.get_wf_integrate(run)
    if "get_wf_hitfinder" in dir(config):
        wfs_hitfinder_args = config.get_wf_hitfinder(run)
    if "get_droplet2photon" in dir(config):
        d2p_args = config.get_droplet2photon(run)
    if "get_wf_svd" in dir(config):
        wfs_svd_args = config.get_wf_svd(run)
    if "get_droplet" in dir(config):
        droplet_args = config.get_droplet(run)
    if "get_azav" in dir(config):
        azav_args = config.get_azav(run)
    if "get_azav_pyfai" in dir(config):
        azav_pyfai_args = config.get_azav_pyfai(run)
    if "get_polynomial_correction" in dir(config):
        polynomial_args = config.get_polynomial_correction(run)
    if "get_sum_algos" in dir(config):
        sum_algo_args = config.get_sum_algos(run)
    if "get_pressio_compression" in dir(config):
        pressio_compression_args = config.get_pressio_compression(run)

    dets = []

    for detname in det_list:
        # For xtcpp, we can't check detnames the same way
        # We'll try to create the detector and let DetObject handle errors
        # Common mode (default: None)
        common_mode = None

        if detname.find("fim") >= 0 or detname.find("w8") >= 0:
            # why different name for w8? To differentiate full det from BLD
            det = DetObject(
                detname, ds, common_mode=common_mode, name=f"det_{detname}"
            )
        else:
            det = DetObject(detname, ds, common_mode=common_mode)
        
        # Skip if detector creation failed (returns NullDetObject)
        if isinstance(det, NullDetObject):
            continue
        logger.debug(f"Instantiated det {detname}: {det}")

        #             **** Compression MUST be the first operation ****             #
        # It is "transparent" in that it just compresses then decompresses the data #
        if detname in pressio_compression_args:
            det.addFunc(pressioCompressDecompress(**pressio_compression_args[detname]))

        # HSD need special treatment due to their data structure.
        # TO REVIEW / REVISE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if detname.find("hsd") >= 0:  # and not args.nohsd:
            hsdsplit = hsdsplitFunc(writeHsd=False)
            if detname in rois_args:
                for sdetname in rois_args[detname]:
                    funcname = "%s__%s" % (sdetname, "ROI")
                    if rank == 0:
                        print(
                            f"sdetname {sdetname} args: {rois_args[detname][sdetname]}, func name: {funcname}"
                        )
                    RF = hsdROIFunc(
                        name=funcname,
                        writeArea=True,
                        ROI=rois_args[detname][sdetname],
                    )
                    hsdsplit.addFunc(RF)
            det.addFunc(hsdsplit)
            dets.append(det)
            continue

        ####################################
        ######## Standard detectors ########
        ####################################
        if detname in azav_args:
            det.addFunc(azimuthalBinning(**azav_args[detname]))

        if detname in azav_pyfai_args:
            det.addFunc(azav_pyfai(**azav_pyfai_args[detname]))

        if detname in rois_args:
            # ROI extraction
            for iROI, ROI in enumerate(rois_args[detname]):
                proj_ax = ROI.pop("proj_ax", None)

                thisROIFunc = ROIFunc(**ROI)
                if proj_ax is not None:
                    thisROIFunc.addFunc(projectionFunc(axis=proj_ax))
                det.addFunc(thisROIFunc)

        if detname in d2p_args:
            if rank == 0:
                print(f"\n\nSETTING UP DROPLET TO PHOTON FOR {det._name}")
                print(json.dumps(d2p_args[detname], indent=4))
                print("\n")
            if "nData" in d2p_args[detname]:  # defines how sparsifcation is saved
                nData = d2p_args[detname].pop("nData")
            else:
                nData = None
            if "get_photon_img" in d2p_args[detname]:  # save unsparsified image or not
                unsparsify = d2p_args[detname].pop("get_photon_img")
            else:
                unsparsify = False

            # Make individual analysis functions
            # (i) droplet
            droplet_dict = d2p_args[detname]["droplet"]
            dropfunc = dropletFunc(**droplet_dict)
            # (ii) droplet2Photon
            d2p_dict = d2p_args[detname]["d2p"]
            drop2phot_func = droplet2Photons(**d2p_dict)
            # (iii) sparsify
            sparsify = sparsifyFunc(nData=nData)

            # Assemble pipeline last to first
            if unsparsify:
                unsparsify_func = unsparsifyFunc()
                drop2phot_func.addFunc(unsparsify_func)
            drop2phot_func.addFunc(sparsify)
            dropfunc.addFunc(drop2phot_func)
            det.addFunc(dropfunc)

        if detname in wfs_int_args:
            # Waveform integration
            wfs_int_func = WfIntegration(**wfs_int_args[detname])
            det.addFunc(wfs_int_func)

        if detname in wfs_hitfinder_args:
            # Simple hit finder on waveform
            det.addFunc(SimpleHitFinder(**wfs_hitfinder_args[detname]))

        if detname in wfs_svd_args:
            if isinstance(
                wfs_svd_args[detname], list
            ):  # handle mutliple channel on the same det
                for i_ch, ch in enumerate(wfs_svd_args[detname]):
                    # Check that the basis file exists, skip instantiation if not
                    if not os.path.isfile(ch["basis_file"]):
                        logger.info(
                            f"SVD basis file for {detname} ch{i_ch} cannot be found. Will not instantiate SvdFit for it."
                        )
                        continue
                    this_func = SvdFit(**ch)
                    det.addFunc(this_func)
            else:
                det.addFunc(SvdFit(**wfs_svd_args[detname]))

        if detname in droplet_args:
            if "nData" in droplet_args[detname].keys():
                nData = droplet_args[detname].pop("nData")
            else:
                nData = None
            dFunc = dropletFunc(**droplet_args[detname])
            dFunc.addFunc(sparsifyFunc(nData=nData))
            det.addFunc(dFunc)

        if detname in polynomial_args:
            obj = det
            if "droplet" in polynomial_args[detname]:
                # If polynomial correction is applied to droplet data, we need to
                # add the droplet function first.
                droplet_args = polynomial_args[detname].pop("droplet")
                droplet_func = dropletFunc(**droplet_args)
                unsparsify = unsparsifyFunc()
                droplet_func.addFunc(unsparsify)
                det.addFunc(droplet_func)
                obj = unsparsify

            if "projection" in polynomial_args[detname]:
                # If projection is requested, add it to the polynomial correction.
                proj_args = polynomial_args[detname].pop("projection")
                proj_func = projectionFunc(**proj_args)

            poly_corr_func = PolynomialCurveCorrection(**polynomial_args[detname])
            if "proj_func" in locals():
                print("Add projection")
                poly_corr_func.addFunc(proj_func)
            obj.addFunc(poly_corr_func)

        if sum_algo_args == {}:
            # Store calib by default even if algos. dictionary not defined
            det.storeSum(sumAlgo="calib")
        else:
            # Add `all` algorithms
            if "all" in sum_algo_args:
                for algo in sum_algo_args["all"]:
                    det.storeSum(sumAlgo=algo)
            if detname in sum_algo_args:
                for algo in sum_algo_args[detname]:
                    det.storeSum(sumAlgo=algo)

        logger.debug(f"Rank {rank} Add det {detname}: {det}")
        dets.append(det)

    return dets


# General Workflow
# This is meant for arp which means we will always have an exp and run

fpath = os.path.dirname(os.path.abspath(__file__))
fpathup = "/".join(fpath.split("/")[:-1])
sys.path.append(fpathup)
if rank == 0:
    logger.info(fpathup)

from smalldata_tools.utilities import printMsg, checkDet
from smalldata_tools.common.detector_base import detData, getUserData, getUserEnvData
from smalldata_tools.lcls2.default_detectors import (
    detOnceData,
    epicsDetector,
    genericDetector,
)
from smalldata_tools.lcls2.hutch_default import defaultDetectors
from smalldata_tools.lcls2.DetObject_xtcpp import DetObject, NullDetObject

from smalldata_tools.ana_funcs.roi_rebin import (
    ROIFunc,
    spectrumFunc,
    projectionFunc,
    imageFunc,
)
from smalldata_tools.ana_funcs.sparsifyFunc import sparsifyFunc, unsparsifyFunc
from smalldata_tools.ana_funcs.waveformFunc import WfIntegration, SimpleHitFinder
from smalldata_tools.ana_funcs.waveformFunc import getCMPeakFunc, templateFitFunc
from smalldata_tools.ana_funcs.waveformFunc import (
    hsdsplitFunc,
    hsdBaselineCorrectFunc,
    hitFinderCFDFunc,
    hsdROIFunc,
)
from smalldata_tools.ana_funcs.droplet import dropletFunc
from smalldata_tools.ana_funcs.photons import photonFunc
from smalldata_tools.ana_funcs.droplet2Photons import droplet2Photons
from smalldata_tools.ana_funcs.azimuthalBinning import azimuthalBinning
from smalldata_tools.ana_funcs.azav_pyfai import azav_pyfai
from smalldata_tools.ana_funcs.smd_svd import SvdFit
from smalldata_tools.ana_funcs.detector_corrections import PolynomialCurveCorrection
from smalldata_tools.ana_funcs.compression import pressioCompressDecompress

import psplot
from psmon import publish


# Constants
HUTCHES = ["TMO", "RIX", "UED", "MFX", "XPP"]

S3DF_BASE = Path("/sdf/data/lcls/ds/")
FFB_BASE = Path("/cds/data/drpsrcf/")
PSDM_BASE = Path(os.environ.get("SIT_PSDM_DATA", S3DF_BASE))
SD_EXT = Path("./hdf5/smalldata/")
logger.debug(f"PSDM_BASE={PSDM_BASE}")

# Define Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", help="run", type=str, default=os.environ.get("RUN_NUM", "")
)
parser.add_argument(
    "--experiment",
    help="experiment name",
    type=str,
    default=os.environ.get("EXPERIMENT", ""),
)
parser.add_argument("--stn", help="hutch station", type=int, default=0)
parser.add_argument("--nevents", help="number of events", type=int, default=0)
parser.add_argument(
    "--directory", help="directory for output files (def <exp>/hdf5/smalldata)"
)
parser.add_argument("--gather_interval", help="gather interval", type=int, default=100)
parser.add_argument(
    "--norecorder", help="ignore recorder streams", action="store_true", default=False
)
parser.add_argument("--url", default="https://pswww.slac.stanford.edu/ws-auth/lgbk/")
parser.add_argument(
    "--epicsAll", help="store all epics PVs", action="store_true", default=False
)
parser.add_argument(
    "--full",
    help="store all data (please think before usig this)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--default", help="store only minimal data", action="store_true", default=False
)
parser.add_argument(
    "--image",
    help="save everything as image (use with care)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--tiff",
    help="save all images also as single tiff (use with even more care)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--postRuntable",
    help="postTrigger for seconday jobs",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--wait", help="wait for a file to appear", action="store_true", default=False
)
parser.add_argument(
    "--rawFim", help="save raw Fim data", action="store_true", default=False
)
parser.add_argument(
    "--nohsd", help="dont save HSD data", action="store_true", default=False
)
parser.add_argument(
    "--nosum", help="dont save sums", action="store_true", default=False
)
parser.add_argument(
    "--noarch", help="dont use archiver data", action="store_true", default=False
)
parser.add_argument(
    "--psplot_live_mode",
    help="Run as a server for psplot live mode, i.e. no h5 file being written.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--intg_delta_t",
    help="Offset for the integrating detector batch.",
    type=int,
    default=0,
)
parser.add_argument(
    "--config", help="Producer config file to use", default=None, type=str
)
parser.add_argument(
    "--all_events",
    help="Will write all events. If False, only writes h5 at slow detectors rate.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--psdm_dir",
    help="Override SIT_PSDM_DATA regardless of environment variables set.",
    type=str,
    default="",
)

args = parser.parse_args()

logger.debug("Args to be used for small data run: {0}".format(args))


###### Helper Functions ##########
def get_xtc_files(base, exp, run):
    """File all xtc files for given experiment and run"""
    run_format = "".join(["r", run.zfill(4)])
    data_dir = Path(base) / exp[:3] / exp / "xtc"
    xtc_files = list(data_dir.glob(f"*{run_format}*"))
    if rank == 0:
        logger.debug(f"xtc file list: {xtc_files}")
    return xtc_files


def get_sd_file(write_dir, exp, hutch):
    """Generate directory to write to, create file name"""
    if write_dir is None:
        if onS3DF:  # S3DF should now be the default
            write_dir = S3DF_BASE / hutch.lower() / exp / SD_EXT
        else:
            logger.error("On an unknown system, cannot figure where to save data.")
            logger.error("Please fix or pass a write_dir argument.")
            sys.exit()
    logger.debug(f"hdf5 directory: {write_dir}")

    write_dir = Path(write_dir)
    h5_f_name = write_dir / f"{exp}_Run{run.zfill(4)}.h5"
    if not write_dir.exists():
        if rank == 0:
            logger.info(f"{write_dir} does not exist, creating directory now.")
            try:
                write_dir.mkdir(parents=True)
            except (PermissionError, FileNotFoundError) as e:
                logger.error(
                    f"Unable to make directory {write_dir} for output"
                    f"exiting on error: {e}"
                )
                sys.exit()
    if rank == 0 and not args.psplot_live_mode:
        logger.info("Will write small data file to {0}".format(h5_f_name))
    elif rank == 0 and args.psplot_live_mode:
        logger.warning("Running in psplot_live mode, will not write any h5 file.")
    return h5_f_name


##### START SCRIPT ########
hostname = socket.gethostname()

# Parse hutch name from experiment and check it's a valid hutch
exp = args.experiment
run = args.run
station = args.stn
logger.debug("Analyzing data for EXP:{0} - RUN:{1}".format(args.experiment, args.run))

begin_prod_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

hutch = exp[:3].upper()
if hutch not in HUTCHES:
    logger.error("Could not find {0} in list of available hutches".format(hutch))
    sys.exit()


# Get config file
def get_config_file(name, folder_path=Path(fpathup) / "lcls2_producers"):
    """Find config file using pathlib"""
    folder = Path(folder_path)
    target_file = folder / f"prod_config_{name}.py"

    if target_file.exists():
        return target_file.stem  # return the file name without extension
    else:
        if rank == 0:
            logger.error(f"Config file {target_file} not found.")
        sys.exit(1)


if args.config is None:
    prod_cfg = f"prod_config_{hutch.lower()}"
else:
    prod_cfg = get_config_file(args.config)
if rank == 0:
    logger.info(f"Producer cfg file: <{prod_cfg}>.")
config = import_module(prod_cfg)


# Figure out where we are running from and check that the
# xtc files are where we expect them.
onS3DF = False
useFFB = False
xtc_files = []

if not args.psdm_dir:
    if hostname.find("sdf") >= 0:
        logger.debug("On S3DF")
        onS3DF = True
        if "ffb" in PSDM_BASE.as_posix():
            useFFB = True
            # wait for files to appear
            nFiles = 0
            n_wait = 0
            max_wait = 20  # 10s wait per cycle.
            waitFilesStart = datetime.now()
            while nFiles == 0:
                if n_wait > max_wait:
                    raise RuntimeError(
                        "Waited {str(n_wait*10)}s, still no files available. Giving up."
                    )
                xtc_files = get_xtc_files(PSDM_BASE, exp, run)
                nFiles = len(xtc_files)
                if nFiles == 0:
                    if rank == 0:
                        print(
                            f"We have no xtc files for run {run} in {exp} in the FFB system, "
                            "we will wait for 10 second and check again."
                        )
                    n_wait += 1
                    time.sleep(10)
            waitFilesEnd = datetime.now()
            if rank == 0:
                print(
                    f"Files appeared after {str(waitFilesEnd-waitFilesStart)} seconds"
                )

        xtc_files = get_xtc_files(PSDM_BASE, exp, run)
        if len(xtc_files) == 0:
            raise RuntimeError(
                f"We have no xtc files for run {run} in {exp} in the offline system."
            )
    else:
        logger.warning("On an unknow system, things may get weird.")
else:
    logger.info(f"Requested data be found at: {args.psdm_dir}")

# Get output file, check if we can write to it
h5_f_name = get_sd_file(args.directory, exp, hutch)

# Setup if integrating detectors are requested.
if hasattr(config, "get_intg"):
    intg_main, intg_addl = config.get_intg(run)
    integrating_detectors = []
    skip_intg = False
else:
    intg_main, intg_addl = (None, None)
    skip_intg = True

# Create data source using xtcpp
events_per_read = 4000  # Default value, can be made configurable
if args.nevents != 0:
    # Note: xtcpp doesn't have max_events parameter in the same way
    # This would need to be handled in the event loop
    pass

ds = _xtcpp.MPIDataSource(exp, int(run), events_per_read)

if rank == 0:
    print("#### DATASOURCE INFO ####")
    print(f"Instantiated data source with experiment: {exp}, run: {run}")
    print(f"MPI size: {size}")
    print("#### END DATASOURCE INFO ####\n")
    logger.info(f"Rank: {rank}")

# Generate smalldata object
# CRITICAL: xtcpp writes files to the current working directory as test_<rank>.h5
# We need to change to the output directory before opening the file
output_dir = h5_f_name.parent
original_cwd = os.getcwd()
try:
    os.chdir(str(output_dir))
    if rank == 0:
        logger.info(f"Changed to output directory: {output_dir}")
        logger.info(
            "Opening the h5file %s, gathering at %d" % (h5_f_name, args.gather_interval)
        )
except Exception as e:
    if rank == 0:
        logger.warning(f"Could not change to output directory {output_dir}: {e}")
        logger.warning(
            "Files will be written to current directory: {0}".format(original_cwd)
        )

if args.psplot_live_mode:
    if rank == 0:
        logger.info("Setting up psplot_live plots.")
        logger.warning("psplot_live_mode not yet implemented for xtcpp")
    small_data = _xtcpp.SmallData(args.gather_interval)
else:
    small_data = _xtcpp.SmallData(args.gather_interval)
    # CRITICAL: Must call open_file() before using small_data, otherwise destructor will hang
    small_data.open_file()
if rank == 0:
    logger.info(
        "smalldata file has been opened (per-rank files: test_<rank>.h5 in {0})".format(
            os.getcwd()
        )
    )


##########################################################
##
## Setting up the default detectors
##
##########################################################
# For xtcpp, all ranks can access detectors (no srv nodes concept)
default_dets = []
if rank == 0:
    logger.info("Default detectors setup skipped for xtcpp (needs adaptation)")

EODet = None
EODetData = {"epicsOnce": {}}
EODetTS = None

default_det_aliases = []

dets = []
int_dets = []
if not args.default:
    # Try to discover available detectors by attempting to access common detector names
    # This helps diagnose when config doesn't match actual detector names in data
    if rank == 0:
        logger.info("Attempting to discover available detectors in data...")
        common_detector_names = ["jungfrau", "epix100", "epix100_0", "epix100_1", "alvium", "opal", "rayonix"]
        available_detectors = []
        for detname in common_detector_names:
            try:
                test_det = ds.detector(detname)
                if test_det is not None:
                    available_detectors.append(detname)
                    logger.info(f"  Found detector: {detname}")
            except Exception:
                pass  # Detector not available, skip
        if available_detectors:
            logger.info(f"Available detectors found: {available_detectors}")
            if config.detectors and not any(d in available_detectors for d in config.detectors):
                logger.warning(f"WARNING: Config detectors {config.detectors} don't match available detectors {available_detectors}")
                logger.warning("Consider updating prod_config_mfx.py to include available detectors")
        else:
            logger.warning("Could not auto-detect any common detectors. Detector names may be different.")
    
    dets = define_dets(int(args.run), config.detectors)
    if not skip_intg:
        int_dets = define_dets(int(args.run), integrating_detectors)
if rank == 0:
    logger.info(f"Detectors configured: {[det._name for det in dets]}")
    logger.info(f"Integrating detectors: {[det._name for det in int_dets]}")
    if not dets:
        logger.warning("No detectors were successfully configured! Check detector names in config file.")
logger.debug(f"Rank {rank} detectors: {[det._name for det in dets]}")
logger.debug(
    f"Rank {rank} integrating detectors: {[det._name for det in int_dets]}"
)

det_presence = {}
if args.full:
    if rank == 0:
        logger.warning("--full option not fully supported with xtcpp yet")

evt_num = -1
if rank == 0:
    logger.info("And now the event loop user....")

normdict = {}
for det in int_dets:
    normdict[det._name] = {"count": 0, "timestamp_min": 0, "timestamp_max": 0}

# For xtcpp, iterate directly over the datasource
event_iter = ds

for evt_num, evt in enumerate(event_iter):
    # Respect --nevents parameter if specified
    if args.nevents > 0 and evt_num >= args.nevents:
        if rank == 0:
            logger.info(f"Reached event limit ({args.nevents}), stopping event loop")
        break

    det_data = detData(default_dets, evt)

    # detector data using DetObject
    userDict = {}
    for det in dets:
        try:
            det.getData(evt_num)
            det.processFuncs()
            userDict[det._name] = getUserData(det)
            try:
                envData = getUserEnvData(det)
                if len(envData.keys()) > 0:
                    userDict[det._name + "_env"] = envData
            except Exception:
                pass
            det.processSums()
        except Exception as e:
            logger.warning(f"Failed analyzing det {det} on evt {evt_num}: {e}")
            pass

    # Combine default data & user data into single dict.
    det_data.update(userDict)

    # Integrating detectors
    if len(int_dets) > 0:
        userDictInt = {}
        # Get summed fast detectors' data for integrating detector event
        for det in int_dets:
            normdict[det._name]["count"] += 1

            # for now, sum up all default data....
            for k, v in det_data.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        sumkey = k + "_sum_" + kk
                        if k not in normdict[det._name]:
                            normdict[det._name][k] = {}
                            normdict[det._name][k][sumkey] = np.array(vv)
                        if sumkey in normdict[det._name][k].keys():
                            normdict[det._name][k][sumkey] += np.array(vv)
                        else:
                            normdict[det._name][k][sumkey] = np.array(vv)
                else:
                    sumkey = k + "_sum"
                    if sumkey in normdict[det._name]:
                        normdict[det._name][sumkey] += v
                    else:
                        normdict[det._name][sumkey] = v

            normdict[det._name]["timestamp_max"] = max(
                normdict[det._name]["timestamp_max"], evt.timestamp
            )
            if normdict[det._name]["timestamp_min"] == 0:
                normdict[det._name]["timestamp_min"] = evt.timestamp
            else:
                normdict[det._name]["timestamp_min"] = min(
                    normdict[det._name]["timestamp_min"], evt.timestamp
                )

            if evt.EndOfBatch():
                det.getData(evt_num)

                if det.evt.dat is None:
                    logger.info(
                        f"Rank {rank}: Integrating detector {det._name} has no data on evt {evt_num}"
                    )
                    userDictInt[det._name] = {}  # still get summed fast data

                else:
                    det.processFuncs()
                    userDictInt[det._name] = {}
                    tmpdict = getUserData(det)
                    for k, v in tmpdict.items():
                        userDictInt[det._name][k] = v

                    try:
                        envData = getUserEnvData(det)
                        if len(envData.keys()) > 0:
                            userDictInt[det._name + "_env"] = envData
                    except Exception:
                        pass

                # save data in integrating det dictionary & reset norm dictionary
                for k, v in normdict[det._name].items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            userDictInt[det._name][kk] = vv
                            normdict[det._name][k][kk] = vv * 0
                    else:
                        userDictInt[det._name][k] = v
                        normdict[det._name][k] = v * 0

                packed_data, packed_shape = xtcpp_pack(userDictInt)
                small_data.event(packed_data, packed_shape)

    # store event-based data
    if len(int_dets) == 0 or args.all_events:
        # CRITICAL: HDF5Writer::event() returns early if event_data is empty
        # This causes completely empty HDF5 files when no detectors are found
        # Ensure we always have at least minimal data to write
        if not det_data:
            try:
                timestamp = evt.timestamp if hasattr(evt, 'timestamp') else 0
                det_data = {
                    "event_num": evt_num,
                    "timestamp": timestamp
                }
            except Exception:
                det_data = {"event_num": evt_num}
        
        packed_data, packed_shape = xtcpp_pack(det_data)
        # Double-check: HDF5Writer skips empty data, so ensure we have something
        if not packed_data:
            logger.warning(f"Rank {rank}: No data after packing for event {evt_num}, writing minimal metadata")
            packed_data = {"event_num": np.array([evt_num], dtype=np.int32)}
            packed_shape = {"event_num": []}
        
        small_data.event(packed_data, packed_shape)
    else:
        scan_data = det_data.get("scan", {})
        timing_data = det_data.get("timing", {})
        data_for_smd = {"scan": scan_data, "timing": timing_data}
        # Ensure at least some minimal data is written
        if not data_for_smd or (not scan_data and not timing_data):
            try:
                timestamp = evt.timestamp if hasattr(evt, 'timestamp') else 0
                data_for_smd = {
                    "event_num": evt_num,
                    "timestamp": timestamp
                }
            except Exception:
                data_for_smd = {"event_num": evt_num}
        
        packed_data, packed_shape = xtcpp_pack(data_for_smd)
        # Double-check: ensure we have data to write
        if not packed_data:
            logger.warning(f"Rank {rank}: No data after packing for event {evt_num}, writing minimal metadata")
            packed_data = {"event_num": np.array([evt_num], dtype=np.int32)}
            packed_shape = {"event_num": []}
        
        small_data.event(packed_data, packed_shape)

    # the ARP will pass run & exp via the environment, if I see that info, the post updates
    if (
        (evt_num < 10)
        or (evt_num < 100 and (evt_num % 10) == 0)
        or (evt_num < 1000 and evt_num % 100 == 0)
        or (evt_num % 1000 == 0)
    ):
        if (os.environ.get("ARP_JOB_ID", None)) is not None and rank == 0:
            try:
                requests.post(
                    os.environ["JID_UPDATE_COUNTERS"],
                    json=[
                        {
                            "key": "<b>Current Event / rank </b>",
                            "value": evt_num + 1,
                        }
                    ],
                )
            except Exception:
                print("ARP update post failed")
                pass
        elif rank == 0:
            print("Processed evt %d" % evt_num)

# For xtcpp, all ranks can process sums
sumDict = {"Sums": {}}
for det in dets:
    for key in det.storeSum().keys():
        try:
            sumData = det.storeSum()[key]
            sumDict["Sums"]["%s_%s" % (det._name, key)] = sumData
        except Exception as e:
            print(
                "Problem with data sum for %s and key %s: %s"
                % (det._name, key, str(e))
            )

# Save summary data, if any
if len(sumDict["Sums"].keys()) > 0:
    try:
        packed_data, packed_shape = xtcpp_pack(sumDict)
        small_data.save_summary(packed_data, packed_shape)
    except Exception as e:
        logger.warning(f"Failed to save summary data with save_summary: {e}")
        logger.warning(
            "Summary data (Sums) will not be saved. This may be expected if save_summary is not fully supported."
        )

# Save detector configuration (numeric only)
if rank == 0:
    logger.info("Saving detector configuration to UserDataCfg")
    userDataCfg = {}
    for det in default_dets:
        noConfigSave = ["scan", "damage"]
        if det.name not in noConfigSave:
            userDataCfg[det.name] = det.params_as_dict()
    for det in dets:
        try:
            userDataCfg[det._name] = det.params_as_dict()
        except Exception:
            userDataCfg[det.name] = det.params_as_dict()
    for det in int_dets:
        try:
            userDataCfg[det._name] = det.params_as_dict()
        except Exception:
            userDataCfg[det.name] = det.params_as_dict()

    Config = {"UserDataCfg": userDataCfg}

    # Filter out non-numeric values (xtcpp writer is numeric-friendly; strings via bytes also OK,
    # but keep it simple and avoid huge dict blobs).
    def _filter_numeric_tree(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                vv = _filter_numeric_tree(v)
                if vv is not None:
                    out[k] = vv
            return out if out else None
        if isinstance(obj, (np.ndarray, np.ma.MaskedArray)):
            if np.asarray(obj).dtype.kind in ("i", "u", "f", "b"):
                return obj
            return None
        if _is_scalar_number(obj):
            return float(obj)
        # allow short strings as bytes; skip long/complex non-numeric by default
        return None

    Config_filtered = _filter_numeric_tree(Config)
    if Config_filtered is not None:
        try:
            packed_data, packed_shape = xtcpp_pack(Config_filtered)
            small_data.save_summary(packed_data, packed_shape)  # this only works w/ 1 rank!
        except Exception as e:
            logger.warning(f"Failed to save config data with save_summary: {e}")
            logger.warning(
                "Config data (UserDataCfg) will not be saved. This may be expected if save_summary is not fully supported."
            )

# Finishing up:
logger.info(f"Rank {rank}: Finishing up, about to delete small_data object")

# Explicitly delete small_data to trigger destructor and ensure cleanup happens
if rank == 0:
    logger.info("Rank 0: Deleting small_data object (will trigger destructor)")
try:
    del small_data
except Exception:
    pass
if rank == 0:
    logger.info("Rank 0: small_data object deleted, destructor should have completed")

# CRITICAL: The segfault occurs when C++ Detector destructors call MPI_Win_free.
# We'll do minimal cleanup here and defer the rest until just before os._exit()
if rank == 0:
    logger.info("Rank 0: Event processing complete, proceeding to final steps")

# Epics data from the archiver (skipped for xtcpp)
h5_rank = 0
if rank == h5_rank:
    logger.info(f"Getting epics data from Archiver (rank: {rank})")
    logger.warning(
        f"Rank {rank}: Skipping epics archiver for xtcpp (per-rank files not compatible)"
    )

if rank == 0:
    logger.info("Rank 0: About to call MPI barrier")
MPI.COMM_WORLD.Barrier()
if rank == 0:
    logger.info("Rank 0: Passed MPI barrier")
if rank == 0:
    logger.warning(
        "Rank 0: Skipping config writing to main file for xtcpp (per-rank files not compatible)"
    )

end_prod_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
end_job = time.time()
prod_time = (end_job - start_job) / 60
if rank == 0:
    print("########## JOB TIME: {:03f} minutes ###########".format(prod_time))
logger.debug("rank {0} on {1} is finished".format(rank, hostname))

if rank == 0:
    if os.environ.get("ARP_JOB_ID", None) is not None:
        requests.post(
            os.environ["JID_UPDATE_COUNTERS"],
            json=[
                {
                    "key": "<b>Last Event</b>",
                    "value": "~ %d cores * %d evts" % (size, evt_num),
                }
            ],
        )
    else:
        print(f"Last Event: {evt_num}")

if args.postRuntable and rank == 0:
    print("Posting to the run tables.")
    locStr = ""
    try:
        runtable_data = {
            "Prod%s_end" % locStr: end_prod_time,
            "Prod%s_start" % locStr: begin_prod_time,
            "Prod%s_jobstart" % locStr: begin_job_time,
            "Prod%s_ncores" % locStr: size,
        }
    except Exception:
        runtable_data = {
            "Prod%s_end" % locStr: end_prod_time,
            "Prod%s_start" % locStr: begin_prod_time,
            "Prod%s_jobstart" % locStr: begin_job_time,
        }
    if args.default:
        runtable_data["SmallData%s" % locStr] = "default"
    else:
        runtable_data["SmallData%s" % locStr] = "done"

    time.sleep(5)

    ws_url = args.url + "/run_control/{0}/ws/add_run_params".format(args.experiment)
    logger.debug("URL: ", ws_url)
    user = (args.experiment[:3] + "opr").replace("dia", "mcc")
    if os.environ.get("ARP_LOCATION", None) == "S3DF":
        with open("/sdf/group/lcls/ds/tools/forElogPost.txt") as reader:
            answer = reader.readline()

        r = requests.post(
            ws_url,
            params={"run_num": args.run},
            json=runtable_data,
            auth=HTTPBasicAuth(args.experiment[:3] + "opr", answer[:-1]),
        )
        logger.debug(r)
    if det_presence != {}:
        rp = requests.post(
            ws_url,
            params={"run_num": args.run},
            json=det_presence,
            auth=HTTPBasicAuth(args.experiment[:3] + "opr", answer[:-1]),
        )
        logger.debug(rp)

# Final cleanup before exit
if rank == 0:
    logger.info("Rank 0: Performing final cleanup before exit")

dets.clear()
int_dets.clear()

# Synchronize all ranks before exit to ensure consistent state
MPI.COMM_WORLD.Barrier()

MPI.Finalize()
os._exit(0)
