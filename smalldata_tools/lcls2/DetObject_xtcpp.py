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


# ----------------------------
# Helper: robust detname resolve
# ----------------------------
def _try_det(ds, name):
    try:
        return ds.detector(name)
    except Exception:
        return None


def _candidate_names(srcName):
    """
    Generate likely alternative names when config name doesn't match xtcpp name.
    Typical case: epix100_1 (missing) -> epix100_0 (exists)
    """
    cands = [srcName]

    # If name ends with _<int>, try _0 and also decrement
    parts = srcName.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base, idx = parts[0], int(parts[1])
        cands.append(f"{base}_0")
        for j in range(idx - 1, -1, -1):
            cands.append(f"{base}_{j}")
        cands.append(base)  # also try stripping numeric suffix

    # If name contains ':' or '/', try last component
    if ":" in srcName:
        cands.append(srcName.split(":")[-1])
    if "/" in srcName:
        cands.append(srcName.split("/")[-1])

    # De-dup while preserving order
    seen = set()
    out = []
    for n in cands:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _resolve_detname_and_det(ds, srcName):
    for name in _candidate_names(srcName):
        det = _try_det(ds, name)
        if det is not None:
            return name, det
    return srcName, None


def DetObject(srcName, ds, **kwargs):
    """
    Factory function to create detector objects using xtcpp.
    Robust python-only version:
      - tries alternate names when ds.detector(srcName) fails
      - chooses class using name hints + available methods
    """
    if rank == 0:
        logger.info(f"Getting the detector for: {srcName}")

    resolved_name, det = _resolve_detname_and_det(ds, srcName)
    if det is None:
        if rank == 0:
            logger.warning(
                f"failed to make detector for {srcName} (tried: {_candidate_names(srcName)})"
            )
        return NullDetObject(name=srcName)

    # Keep logical name for output keys, but store resolved name for debugging
    kwargs.setdefault("name", srcName)
    kwargs.setdefault("resolved_name", resolved_name)

    lname = srcName.lower()

    # PV / epics: in your C++ wrapper, epics dets usually have .get or are callable
    is_pv = ("pv" in lname) or hasattr(det, "get") or callable(det)

    # Camera-ish detectors: xtcpp typically adds .raw with methods on it
    has_raw = hasattr(det, "raw")
    has_calib = has_raw and hasattr(det.raw, "calib")
    has_rawraw = has_raw and hasattr(det.raw, "raw")

    # Prefer explicit name hints
    if "jungfrau" in lname or "jungfr" in lname:
        return JungfrauObject(det, ds, **kwargs)
    if "epix100" in lname or "epix" in lname:
        return Epix100Object(det, ds, **kwargs)

    # If not hinted, decide by capabilities
    if is_pv and not has_raw:
        return PVObject(det, ds, **kwargs)

    # Fallback: if it looks like a camera, default to Epix100Object (2D raw path exists there)
    if has_calib or has_rawraw:
        return Epix100Object(det, ds, **kwargs)

    if rank == 0:
        logger.warning(
            f"Unknown detector type for {srcName} (resolved: {resolved_name}), defaulting to NullDetObject"
        )
    return NullDetObject(name=srcName)


class DetObjectClass(object):
    def __init__(self, det, ds, **kwargs):
        self.det = det

        try:
            self._detid = getattr(det, "_detid", None)
        except Exception:
            self._detid = None

        # IMPORTANT: don't default to 'unknown' based on det internals (xtcpp wrapper won't have _det_name)
        self._name = kwargs.get("name", "unknown")
        self._resolved_name = kwargs.get("resolved_name", self._name)

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
                    and getattr(self, key)  # Check for empty list
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
            if self.applyMask == 1:
                self.evt.dat[self.mask == 0] = 0
            if self.applyMask == 2:
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
                dat_to_be_summed[self.evt.dat < thres] = 0.0

            if key.find("nhits") >= 0:
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
                        self._storeSum[key] = np.maximum(self._storeSum[key], dat_to_be_summed)
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
        self._resolved_name = kwargs.get("resolved_name", self._name)
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

        # xtcpp: calib constants/geometry not available (yet)
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
        # no geometry yet
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

        if self.rms is None or (
            self.ped is not None
            and hasattr(self.rms, "shape")
            and hasattr(self.ped, "shape")
            and self.rms.shape != self.ped.shape
        ):
            self.rms = np.ones_like(self.ped) if self.ped is not None else None
        elif self.rms is not None and not hasattr(self.rms, "shape"):
            self.rms = None

        self.imgShape = None

        # FIX: integer slicing (Python3 division was producing floats)
        self.bankMasks = []
        if self.common_mode == 47 and self.rms is not None:
            for i in range(0, 16):
                bmask = np.zeros_like(self.rms)
                col0 = (768 // 8) * (i // 2)
                col1 = (768 // 8) * ((i // 2) + 1)
                bmask[(i % 2) * 352 : (i % 2 + 1) * 352, col0:col1] = 1
                self.bankMasks.append(bmask.astype(bool))

    def getData(self, evt):
        super().getData(evt)

        # xtcpp epix100: use raw.raw(evt)
        def _raw():
            try:
                rd = self.det.raw.raw(evt)
                return np.asarray(rd) if rd is not None and not isinstance(rd, np.ndarray) else rd
            except Exception:
                return None

        raw_data = _raw()

        if raw_data is None:
            self.evt.dat = None
            return

        cm = self.common_mode

        if cm in [0, -1, 30]:
            self.evt.dat = raw_data
        elif cm % 100 in [6, 36, 34, 4, 45, 46, 47]:
            # pedestal subtract if available
            if self.ped is not None and hasattr(self.ped, "shape") and hasattr(raw_data, "shape"):
                try:
                    self.evt.dat = raw_data - self.ped
                except Exception as e:
                    logger.warning(f"Error pedestal-sub epix100 ({self._name}) cm={cm}: {e}")
                    self.evt.dat = raw_data
            else:
                self.evt.dat = raw_data

            # apply common-mode where your original code did
            try:
                if cm % 100 == 6:
                    self.evt.dat = cm_epix(self.evt.dat, self.rms, normAll=True, mask=self.mask)
                elif cm % 100 == 36:
                    self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
                elif cm % 100 == 45:
                    self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
                elif cm % 100 == 46:
                    self.evt.dat = cm_epix(self.evt.dat, self.rms, normAll=True, mask=self.mask)
                elif cm % 100 == 47:
                    for bMask in self.bankMasks:
                        self.evt.dat[bMask] -= np.median(self.evt.dat[bMask])
                    self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=self.mask)
            except Exception as e:
                logger.warning(f"Error common-mode epix100 ({self._name}) cm={cm}: {e}")
        else:
            # unknown mode: just raw
            self.evt.dat = raw_data

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

        # xtcpp jungfrau: calib exists (C++ binding adds calib for jungfrau)
        try:
            if self.common_mode in [0, 7, 71, 72, 30]:
                self.evt.dat = self.det.raw.calib(evt)
            elif self.common_mode == -1:
                self.evt.dat = self.det.raw.raw(evt)
            else:
                self.evt.dat = self.det.raw.calib(evt)
        except Exception as e:
            logger.warning(f"Jungfrau getData failed for {self._name}: {e}")
            self.evt.dat = None

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
