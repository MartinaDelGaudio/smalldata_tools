import os
import copy
import numpy as np
import logging
from smalldata_tools.common.detector_base import DetObjectFunc, Event
from smalldata_tools.utilities import cm_epix
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

import _xtcpp  # noqa: F401

logger = logging.getLogger(__name__)


def _asarray(x):
    """Convert to numpy array if needed (leave None as None)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def DetObject(srcName, ds, **kwargs):
    """
    Factory function to create detector objects using xtcpp.
    Changed from psana version: takes ds (datasource) instead of run.
    """
    if rank == 0:
        logger.info(f"Getting the detector for: {srcName}")
    det = None
    try:
        det = ds.detector(srcName)
    except Exception as e:
        if rank == 0:
            logger.warning(f"failed to make detector for {srcName}: {e}")
        return NullDetObject(name=srcName)

    # For xtcpp, infer type from name / capabilities
    detector_type = None
    if "epix100" in srcName.lower():
        detector_type = "epix100"
    elif "jungfrau" in srcName.lower():
        detector_type = "jungfrau"
    elif "pv" in srcName.lower() or hasattr(det, "get"):
        detector_type = "pv"
    else:
        if rank == 0:
            logger.warning(
                f"Unknown detector type for {srcName}, defaulting to NullDetObject"
            )
        return NullDetObject(name=srcName)

    detector_classes = {
        "epix100": Epix100Object,
        "jungfrau": JungfrauObject,
        "pv": PVObject,
    }

    cls = detector_classes.get(detector_type)
    if cls is None:
        return NullDetObject(name=srcName)

    return cls(det, ds, **kwargs)


class DetObjectClass(object):
    def __init__(self, det, ds, **kwargs):
        self.det = det
        try:
            self._detid = getattr(det, "_detid", None)
        except Exception:
            self._detid = None
        self._name = kwargs.get("name", getattr(det, "_det_name", "unknown"))

        self.ds = ds
        self._storeSum = {}
        self.applyMask = kwargs.get("applyMask", 0)

        self.dataAccessTime = 0.0

    def params_as_dict(self):
        """Returns parameters as dictionary to be stored in the hdf5 file (once/file)"""
        parList = {
            key: self.__dict__[key]
            for key in self.__dict__
            if (
                key[0] != "_"
                and isinstance(getattr(self, key), (str, int, float, np.ndarray))
            )
        }
        parList.update(
            {
                key: np.array(self.__dict__[key])
                for key in self.__dict__
                if (
                    key[0] != "_"
                    and isinstance(getattr(self, key), tuple)
                    and isinstance(getattr(self, key)[0], (str, int, float, np.ndarray))
                )
            }
        )
        parList.update(
            {
                key: np.array(self.__dict__[key])
                for key in self.__dict__
                if (
                    key[0] != "_"
                    and isinstance(getattr(self, key), list)
                    and getattr(self, key)
                    and isinstance(getattr(self, key)[0], (str, int, float, np.ndarray))
                )
            }
        )

        subFuncs = [
            self.__dict__[key]
            for key in self.__dict__
            if isinstance(self.__dict__[key], DetObjectFunc)
        ]
        for sf in subFuncs:
            sfPars = sf.params_as_dict()
            parList.update(
                {
                    "%s__%s" % (sf._name, key): value
                    for key, value in sfPars.items()
                    if (
                        key[0] != "_"
                        and isinstance(value, (str, int, float, np.ndarray, tuple))
                        and key.find("coords") < 0
                    )
                }
            )
        return parList

    def _getMasks(self):
        self.mask = None
        self.cmask = None

    def _applyMask(self):
        try:
            if self.applyMask == 1 and self.mask is not None:
                self.evt.dat[self.mask == 0] = 0
            if self.applyMask == 2 and self.cmask is not None:
                self.evt.dat[self.cmask == 0] = 0
        except Exception:
            print("Could not apply mask to data for detector ", self._name)

    def storeSum(self, sumAlgo=None):
        if sumAlgo is not None:
            self._storeSum[sumAlgo] = None
        else:
            return self._storeSum

    def setPed(self, ped):
        self.ped = ped

    def setMask(self, mask):
        self.mask = mask

    def setcMask(self, mask):
        self.cmask = np.amin(np.array([self.mask, mask]), axis=0)

    def setGain(self, gain):
        """
        Set a local gain.
        This file is supposed to be applied on top of whatever corrections DetObject will apply.
        """
        self.local_gain = gain

    def getData(self, evt):
        try:
            getattr(self, "evt")
        except Exception:
            self.evt = Event()
        self.evt.dat = None

    def addFunc(self, func):
        func.setFromDet(self)
        try:
            func.setFromFunc()
        except Exception:
            print("Failed to pass parameters to children of ", func._name)
        self.__dict__[func._name] = func

    def processFuncs(self):
        if self.evt.dat is None:
            logger.debug("This event has no data to be processed for %s" % self._name)
            return
        for func in [
            self.__dict__[k]
            for k in self.__dict__
            if isinstance(self.__dict__[k], DetObjectFunc)
        ]:
            retData = func.process(self.evt.dat)
            self.evt.__dict__["_write_%s" % func._name] = retData

    def processSums(self):
        for key in self._storeSum.keys():
            thres = -1.0e9
            for skey in key.split("_"):
                if skey.find("thresADU") >= 0:
                    thres = float(skey.replace("thresADU", ""))

            if self.evt.dat is None:
                return

            dat_to_be_summed = self.evt.dat
            if thres > 1e-9:
                dat_to_be_summed = dat_to_be_summed.copy()
                dat_to_be_summed[self.evt.dat < thres] = 0.0

            if key.find("nhits") >= 0:
                dat_to_be_summed = dat_to_be_summed.copy()
                dat_to_be_summed[dat_to_be_summed > 0] = 1

            if key.find("square") >= 0:
                dat_to_be_summed = np.square(dat_to_be_summed)

            if self._storeSum[key] is None:
                if dat_to_be_summed is not None:
                    self._storeSum[key] = dat_to_be_summed.copy()
            else:
                try:
                    if key.find("max") < 0:
                        self._storeSum[key] += dat_to_be_summed
                    else:
                        self._storeSum[key] = np.maximum(
                            self._storeSum[key], dat_to_be_summed
                        )
                except Exception:
                    print("could not add ", dat_to_be_summed)
                    print("could not to ", self._storeSum[key])


class NullDetObject:
    """
    A dummy detector object that does nothing.
    Useful to handle instantiation in case the detector is not present in the data.
    """

    def __init__(self, *args, **kwargs):
        self._name = kwargs.get("name", "NullDetObject")
        self.det = None
        self.ds = None
        self.evt = Event()
        self._storeSum = {}
        self.applyMask = 0
        self.dataAccessTime = 0.0

    def addFunc(self, func):
        pass


class CameraObject(DetObjectClass):
    def __init__(self, det, ds, **kwargs):
        super(CameraObject, self).__init__(det, ds, **kwargs)
        self._common_mode_list = [0, -1, 30]  # none, raw, calib
        self.common_mode = kwargs.get("common_mode", self._common_mode_list[0])
        if self.common_mode is None:
            self.common_mode = self._common_mode_list[0]
        if self.common_mode not in self._common_mode_list and type(self) is CameraObject:
            print(
                "Common mode %d is not an option for a CameraObject, please choose from: "
                % self.common_mode,
                self._common_mode_list,
            )
        self.pixelsize = None
        self.isGainswitching = False

        # xtcpp: likely None unless user sets via setPed/setMask/etc.
        self.ped = None
        self.rms = None
        self.gain = None
        self.mask = None
        self.cmask = None

        self.local_gain = None
        self._getImgShape()
        self._gainSwitching = False
        self.x, self.y, self.z = None, None, None

    def getData(self, evt):
        super(CameraObject, self).getData(evt)

    def _getImgShape(self):
        self.imgShape = None


class TiledCameraObject(CameraObject):
    def __init__(self, det, ds, **kwargs):
        super(TiledCameraObject, self).__init__(det, ds, **kwargs)
        self.ix = None
        self.iy = None
        self._needsGeo = True

    def getData(self, evt):
        super(TiledCameraObject, self).getData(evt)


class Epix100Object(TiledCameraObject):
    def __init__(self, det, ds, **kwargs):
        super().__init__(det, ds, **kwargs)
        self._common_mode_list = [6, 36, 4, 34, 45, 46, 47, 0, -1, 30]
        self.common_mode = kwargs.get("common_mode", self._common_mode_list[0])
        if self.common_mode is None:
            self.common_mode = self._common_mode_list[0]
        if self.common_mode not in self._common_mode_list:
            print(
                "Common mode %d is not an option for as Epix detector, please choose from: "
                % self.common_mode,
                self._common_mode_list,
            )

        self.pixelsize = [50e-6]
        self.areas = None
        self.imgShape = None

        # bank masks will be built lazily once we know the frame shape
        self.bankMasks = []

    def _ensure_ped_rms(self, raw_data):
        """
        xtcpp: if ped/rms are not available, create safe defaults from first frame
        so cm_epix doesn't crash.
        """
        if raw_data is None:
            return

        # use float32 for downstream math / sums
        if self.ped is None:
            self.ped = np.zeros_like(raw_data, dtype=np.float32)
        else:
            self.ped = _asarray(self.ped)
            if self.ped is None or self.ped.shape != raw_data.shape:
                self.ped = np.zeros_like(raw_data, dtype=np.float32)

        if self.rms is None:
            self.rms = np.ones_like(raw_data, dtype=np.float32)
        else:
            self.rms = _asarray(self.rms)
            if self.rms is None or self.rms.shape != raw_data.shape:
                self.rms = np.ones_like(raw_data, dtype=np.float32)

        # ensure mask is array if present
        if self.mask is not None:
            self.mask = _asarray(self.mask)

    def _ensure_bank_masks(self, shape):
        if self.common_mode != 47:
            return
        if self.bankMasks:
            return
        # epix100 assumed 704x768 (your sums show 704x768). Still build generically.
        # Keep original layout logic but use integer division.
        try:
            h, w = shape
        except Exception:
            return
        # only build if shape is at least as big as expected
        if h < 704 or w < 768:
            return

        # build masks on a dummy array
        dummy = np.zeros(shape, dtype=np.uint8)
        for i in range(16):
            bmask = dummy.copy()
            row0 = (i % 2) * 352
            row1 = (i % 2 + 1) * 352
            col0 = (768 // 8) * (i // 2)
            col1 = (768 // 8) * (i // 2 + 1)
            bmask[row0:row1, col0:col1] = 1
            self.bankMasks.append(bmask.astype(bool))

    def getData(self, evt):
        super().getData(evt)

        # epix100 xtcpp: raw only
        try:
            raw_data = self.det.raw.raw(evt)
        except Exception:
            raw_data = None

        raw_data = _asarray(raw_data)
        if raw_data is None:
            self.evt.dat = None
            return

        # make sure ped/rms exist if we plan to do cm_epix modes
        if self.common_mode % 100 in [6, 36, 45, 46, 47]:
            self._ensure_ped_rms(raw_data)

        # do math in float32 (your sums are float32 anyway)
        raw_f = raw_data.astype(np.float32, copy=False)

        cm = self.common_mode

        # default
        self.evt.dat = raw_f

        if cm % 100 in [6, 36, 34, 4, 45, 46, 47]:
            # pedestal subtract if ped exists (it will after _ensure_ped_rms for cm modes above)
            if self.ped is not None:
                try:
                    self.evt.dat = raw_f - self.ped
                except Exception as e:
                    logger.warning(f"Error ped-sub epix100 ({self._name}) cm={cm}: {e}")
                    self.evt.dat = raw_f

            # IMPORTANT FIX: only call cm_epix if rms is a real array
            if cm % 100 in [6, 36, 45, 46, 47]:
                if self.rms is None or not isinstance(self.rms, np.ndarray):
                    # no rms -> skip cm_epix (avoid your NoneType crash)
                    if rank == 0:
                        logger.warning(
                            f"epix100 ({self._name}) cm={cm}: rms is None, skipping cm_epix"
                        )
                else:
                    try:
                        if cm % 100 == 6:
                            self.evt.dat = cm_epix(
                                self.evt.dat, self.rms, normAll=True, mask=self.mask
                            )
                        elif cm % 100 == 36:
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
                        elif cm % 100 == 45:
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
                        elif cm % 100 == 46:
                            self.evt.dat = cm_epix(
                                self.evt.dat, self.rms, normAll=True, mask=self.mask
                            )
                        elif cm % 100 == 47:
                            self._ensure_bank_masks(self.evt.dat.shape)
                            for bMask in self.bankMasks:
                                self.evt.dat[bMask] -= np.median(self.evt.dat[bMask])
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
                    except Exception as e:
                        logger.warning(
                            f"Error common-mode epix100 ({self._name}) cm={cm}: {e}"
                        )
                        # fall back to ped-sub (or raw)
                        # (don't revert to raw_data int dtype)
                        self.evt.dat = raw_f - self.ped if self.ped is not None else raw_f

        elif cm in [0, -1, 30]:
            # raw passthrough
            self.evt.dat = raw_f

        # override gain if desired
        if (
            self.local_gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, "shape")
            and hasattr(self.local_gain, "shape")
            and self.local_gain.shape == self.evt.dat.shape
            and self.common_mode in [6, 36, 34, 3, 4, 45, 46, 47]
        ):
            self.evt.dat *= self.local_gain
        elif (
            self.local_gain is None
            and self.gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, "shape")
            and hasattr(self.gain, "shape")
            and self.gain.shape == self.evt.dat.shape
            and self.common_mode in [45, 46, 47]
        ):
            self.evt.dat *= self.gain

        if self.areas is not None and self.evt.dat is not None:
            self.evt.dat /= self.areas


class JungfrauObject(TiledCameraObject):
    def __init__(self, det, ds, **kwargs):
        super().__init__(det, ds, **kwargs)
        self._common_mode_list = [0, 7, 71, 72, -1, 30]
        self.common_mode = kwargs.get("common_mode", self._common_mode_list[0])
        if self.common_mode is None:
            self.common_mode = self._common_mode_list[0]
        if self.common_mode not in self._common_mode_list:
            print(
                "Common mode %d is not an option for Jungfrau, please choose from: "
                % self.common_mode,
                self._common_mode_list,
            )
        self.pixelsize = [75e-6]
        self.isGainswitching = True
        self.imgShape = None
        self._gainSwitching = True

    def getData(self, evt):
        super(JungfrauObject, self).getData(evt)

        # xtcpp jungfrau: calib exists
        try:
            if self.common_mode in [0, 7, 71, 72, 30]:
                self.evt.dat = _asarray(self.det.raw.calib(evt))
            elif self.common_mode == -1:
                self.evt.dat = _asarray(self.det.raw.raw(evt))
            else:
                self.evt.dat = _asarray(self.det.raw.calib(evt))
        except Exception:
            self.evt.dat = None

        if self.evt.dat is not None:
            self.evt.dat = self.evt.dat.astype(np.float32, copy=False)

        if (
            self.local_gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, "shape")
            and hasattr(self.local_gain, "shape")
            and self.local_gain.shape == self.evt.dat.shape
            and self.common_mode in [7, 71, 72, 0]
        ):
            self.evt.dat *= self.local_gain


class PVObject(CameraObject):
    def __init__(self, det, ds, **kwargs):
        super(PVObject, self).__init__(det, ds, **kwargs)

    def getData(self, evt):
        super(PVObject, self).getData(evt)
        try:
            self.evt.dat = self.det.get(evt)
        except Exception:
            try:
                self.evt.dat = self.det.raw.value(evt)
            except Exception:
                self.evt.dat = None
