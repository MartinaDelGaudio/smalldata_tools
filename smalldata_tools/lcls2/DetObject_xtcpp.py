import os
import copy
import numpy as np
import logging
from smalldata_tools.common.detector_base import DetObjectFunc, Event
from smalldata_tools.utilities import cm_epix
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

import _xtcpp

logger = logging.getLogger(__name__)


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
    
    # For xtcpp, we need to determine detector type differently
    # For now, we'll use the detector name to infer type
    detector_type = None
    if "epix100" in srcName.lower():
        detector_type = "epix100"
    elif "jungfrau" in srcName.lower():
        detector_type = "jungfrau"
    elif "pv" in srcName.lower() or hasattr(det, 'get'):
        detector_type = "pv"
    else:
        if rank == 0:
            logger.warning(f"Unknown detector type for {srcName}, defaulting to NullDetObject")
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
    def __init__(
        self, det, ds, **kwargs
    ):  # Changed: ds instead of run
        self.det = det
        # For xtcpp, we may not have _detid or _det_name
        try:
            self._detid = getattr(det, '_detid', None)
        except:
            self._detid = None
        self._name = kwargs.get("name", getattr(det, '_det_name', 'unknown'))
        
        self.ds = ds  # Changed: store ds instead of run
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

        # Add parameters of function to dict with composite keyt(sf._name, key)
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
        except:
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
        This file is supposed to be applied on top of whatever corrections DetObject will apply, given the common mode
        """
        self.local_gain = gain

    def getData(self, evt):
        try:
            getattr(self, "evt")
        except:
            self.evt = Event()
        self.evt.dat = None

    def addFunc(self, func):
        func.setFromDet(self)  #  Pass parameters from det (rms, geometry, .....)
        try:
            func.setFromFunc()  # Pass parameters from itself to children (rms, bounds, .....)
        except:
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
            if 1:
                retData = func.process(self.evt.dat)
                self.evt.__dict__["_write_%s" % func._name] = retData

    def processSums(self):
        for key in self._storeSum.keys():
            asImg = False
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
                        self._storeSum[key] = np.maximum(
                            self._storeSum[key], dat_to_be_summed
                        )
                except:
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
        """
        Do nothing, as this is a null object.
        """
        pass


class CameraObject(DetObjectClass):
    def __init__(self, det, ds, **kwargs):  # Changed: ds instead of run
        super(CameraObject, self).__init__(det, ds, **kwargs)
        self._common_mode_list = [0, -1, 30]  # none, raw, calib
        self.common_mode = kwargs.get("common_mode", self._common_mode_list[0])
        if self.common_mode is None:
            self.common_mode = self._common_mode_list[0]
        if (
            self.common_mode not in self._common_mode_list
            and type(self) is CameraObject
        ):
            print(
                "Common mode %d is not an option for a CameraObject, please choose from: "
                % self.common_mode,
                self._common_mode_list,
            )
        self.pixelsize = None
        self.isGainswitching = False

        # For xtcpp, calibration constants may not be available the same way
        # These will need to be set manually or retrieved differently
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
    def __init__(self, det, ds, **kwargs):  # Changed: ds instead of run
        super(TiledCameraObject, self).__init__(det, ds, **kwargs)
        # For xtcpp, geometry may not be available yet
        # See user's note: "I didn't add the geometry yet. So anything like det.raw.image will not work"
        self.ix = None
        self.iy = None
        self._needsGeo = True

    def getData(self, evt):
        super(TiledCameraObject, self).getData(evt)


class Epix100Object(TiledCameraObject):
    def __init__(self, det, ds, **kwargs):  # Changed: ds instead of run
        super().__init__(det, ds, **kwargs)
        self._common_mode_list = [
            6,
            36,
            4,
            34,
            45,
            46,
            47,
            0,
            -1,
            30,
        ]
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

        if self.rms is None or (self.ped is not None and hasattr(self.rms, 'shape') and hasattr(self.ped, 'shape') and self.rms.shape != self.ped.shape):
            if self.ped is not None:
                self.rms = np.ones_like(self.ped)
            else:
                self.rms = None
        elif self.rms is not None and not hasattr(self.rms, 'shape'):
            # If rms is not an array, convert it
            self.rms = None
        
        # For xtcpp, imgShape may need to be determined from data
        self.imgShape = None

        self.bankMasks = []
        if self.common_mode == 47 and self.rms is not None:
            for i in range(0, 16):
                bmask = np.zeros_like(self.rms)
                bmask[
                    (i % 2) * 352 : (i % 2 + 1) * 352,
                    768 / 8 * (i / 2) : 768 / 8 * (i / 2 + 1),
                ] = 1
                self.bankMasks.append(bmask.astype(bool))

    def getData(self, evt):
        super().getData(evt)
        mbits = 0

        # For epix100 with xtcpp: use det.raw.raw(evt) instead of det.raw.calib(evt)
        # as per user's note: "det.raw.calib(evt) doesn't exist for epix100"
        # Note: evt is an integer (event index) in xtcpp, not an event object
        if self.common_mode % 100 == 6:
            # Use raw instead of calib for xtcpp
            raw_data = self.det.raw.raw(evt)
            # Ensure raw_data is a numpy array
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape') and hasattr(raw_data, 'shape'):
                    try:
                        self.evt.dat = raw_data - self.ped
                        if self.rms is not None and hasattr(self.rms, 'shape') and hasattr(self.evt.dat, 'shape'):
                            # Ensure mask is also an array if provided
                            mask_to_use = self.mask
                            if mask_to_use is not None and not isinstance(mask_to_use, np.ndarray):
                                mask_to_use = np.asarray(mask_to_use)
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, normAll=True, mask=mask_to_use)
                    except Exception as e:
                        logger.warning(f"Error processing epix100 data with common_mode 6: {e}")
                        self.evt.dat = raw_data
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 36:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape') and hasattr(raw_data, 'shape'):
                    try:
                        self.evt.dat = raw_data - self.ped
                        if self.rms is not None and hasattr(self.rms, 'shape') and hasattr(self.evt.dat, 'shape'):
                            mask_to_use = self.mask
                            if mask_to_use is not None and not isinstance(mask_to_use, np.ndarray):
                                mask_to_use = np.asarray(mask_to_use)
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=mask_to_use)
                    except Exception as e:
                        logger.warning(f"Error processing epix100 data with common_mode 36: {e}")
                        self.evt.dat = raw_data
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 34:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape'):
                    self.evt.dat = raw_data - self.ped
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 4:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape'):
                    self.evt.dat = raw_data - self.ped
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 45:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape') and hasattr(raw_data, 'shape'):
                    try:
                        self.evt.dat = raw_data - self.ped
                        if self.rms is not None and hasattr(self.rms, 'shape') and hasattr(self.evt.dat, 'shape'):
                            mask_to_use = self.mask
                            if mask_to_use is not None and not isinstance(mask_to_use, np.ndarray):
                                mask_to_use = np.asarray(mask_to_use)
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=mask_to_use)
                    except Exception as e:
                        logger.warning(f"Error processing epix100 data with common_mode 45: {e}")
                        self.evt.dat = raw_data
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 46:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape') and hasattr(raw_data, 'shape'):
                    try:
                        self.evt.dat = raw_data - self.ped
                        if self.rms is not None and hasattr(self.rms, 'shape') and hasattr(self.evt.dat, 'shape'):
                            mask_to_use = self.mask
                            if mask_to_use is not None and not isinstance(mask_to_use, np.ndarray):
                                mask_to_use = np.asarray(mask_to_use)
                            self.evt.dat = cm_epix(
                                self.evt.dat, self.rms, normAll=True, mask=mask_to_use
                            )
                    except Exception as e:
                        logger.warning(f"Error processing epix100 data with common_mode 46: {e}")
                        self.evt.dat = raw_data
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode % 100 == 47:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
                if self.ped is not None and hasattr(self.ped, 'shape') and hasattr(raw_data, 'shape'):
                    try:
                        self.evt.dat = raw_data - self.ped
                        for _, bMask in enumerate(self.bankMasks):
                            if self.evt.dat is not None and hasattr(self.evt.dat, '__getitem__'):
                                self.evt.dat[bMask] -= np.median(self.evt.dat[bMask])
                        if self.rms is not None and hasattr(self.rms, 'shape') and hasattr(self.evt.dat, 'shape'):
                            mask_to_use = self.mask
                            if mask_to_use is not None and not isinstance(mask_to_use, np.ndarray):
                                mask_to_use = np.asarray(mask_to_use)
                            self.evt.dat = cm_epix(self.evt.dat, self.rms, mask=mask_to_use)
                    except Exception as e:
                        logger.warning(f"Error processing epix100 data with common_mode 47: {e}")
                        self.evt.dat = raw_data
                else:
                    self.evt.dat = raw_data
            else:
                self.evt.dat = None
        elif self.common_mode == 0:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
            self.evt.dat = raw_data
        elif self.common_mode == -1:
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
            self.evt.dat = raw_data
        elif self.common_mode == 30:
            # For xtcpp, calib doesn't exist for epix100, so use raw
            raw_data = self.det.raw.raw(evt)
            if raw_data is not None:
                if not isinstance(raw_data, np.ndarray):
                    raw_data = np.asarray(raw_data)
            self.evt.dat = raw_data

        # override gain if desired
        if (
            self.local_gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, 'shape')
            and hasattr(self.local_gain, 'shape')
            and self.local_gain.shape == self.evt.dat.shape
            and self.common_mode in [6, 36, 34, 3, 4, 45, 46, 47]
        ):
            self.evt.dat *= self.local_gain
        elif (
            self.local_gain is None
            and self.gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, 'shape')
            and hasattr(self.gain, 'shape')
            and self.gain.shape == self.evt.dat.shape
            and self.common_mode in [45, 46, 47]
        ):
            self.evt.dat *= self.gain

        # correct for area of pixels.
        if self.areas is not None and self.evt.dat is not None:
            self.evt.dat /= self.areas


class JungfrauObject(TiledCameraObject):
    def __init__(self, det, ds, **kwargs):  # Changed: ds instead of run, removed run usage
        super().__init__(det, ds, **kwargs)
        self._common_mode_list = [
            0,
            7,
            71,
            72,
            -1,
            30,
        ]
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

        # Removed: self.imgShape = self.det.raw.image(run, self.ped[0]).shape
        # Geometry not available yet in xtcpp
        self.imgShape = None
        self._gainSwitching = True

    def getData(self, evt):
        super(JungfrauObject, self).getData(evt)
        mbits = 0
        
        # For jungfrau with xtcpp: use det.raw.calib(evt) as per user's note
        if self.common_mode == 0:
            self.evt.dat = self.det.raw.calib(evt)
        elif self.common_mode % 100 == 71:
            self.evt.dat = self.det.raw.calib(evt)
        elif self.common_mode % 100 == 72:
            self.evt.dat = self.det.raw.calib(evt)
        elif self.common_mode % 100 == 7:
            self.evt.dat = self.det.raw.calib(evt)
        elif self.common_mode == -1:
            self.evt.dat = self.det.raw.raw(evt)
        elif self.common_mode == 30:
            self.evt.dat = self.det.raw.calib(evt)

        # override gain if desired
        if (
            self.local_gain is not None
            and self.evt.dat is not None
            and hasattr(self.evt.dat, 'shape')
            and hasattr(self.local_gain, 'shape')
            and self.local_gain.shape == self.evt.dat.shape
            and self.common_mode in [7, 71, 72, 0]
        ):
            self.evt.dat *= self.local_gain


class PVObject(CameraObject):
    def __init__(self, det, ds, **kwargs):  # Changed: ds instead of run
        super(PVObject, self).__init__(det, ds, **kwargs)

    def getData(self, evt):
        super(PVObject, self).getData(evt)
        # For PV with xtcpp: use det.get(evt) as per user's note
        try:
            self.evt.dat = self.det.get(evt)
        except:
            # Fallback to raw.value if get doesn't work
            try:
                self.evt.dat = self.det.raw.value(evt)
            except:
                self.evt.dat = None

