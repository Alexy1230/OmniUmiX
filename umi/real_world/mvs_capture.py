"""Direct MVS USB capture, adapted from OmniUmiX's MVSCamController.

The SDK is loaded only in open(), inside the camera worker. Frames stay BGR
so UMI's existing recording/visualization and BGR-to-RGB transforms work.
No ROS or OmniUmiX installation is needed.
"""
import ctypes as ct
import importlib
import math
import os
import sys
import time
import warnings

import cv2
import numpy as np


def crop_with_black_padding(image, config):
    """Crop in rotated-frame coordinates; missing pixels become zero BGR.

    offset_x/offset_y specify the CENTER in the rotated capture frame.
    Centers are rounded with int(round(value)), matching convert_dataset.py.
    Top-left = (center_x - width // 2, center_y - height // 2).
    For odd sizes, the center pixel is at output index (height//2, width//2).
    This does not detect or automatically center the lens image circle.
    """
    if not config or not config.get('enabled', False):
        return image
    values = [config.get('offset_x', image.shape[1] // 2), config.get('offset_y', image.shape[0] // 2),
              config.get('width'), config.get('height')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values[:2]):
        raise ValueError('sensor_roi centers must be finite numbers')
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values[2:]):
        raise ValueError('sensor_roi width and height must be integers')
    x, y = (int(round(v)) for v in values[:2])
    width, height = values[2:]
    if width <= 0 or height <= 0:
        raise ValueError('sensor_roi width and height must be positive')
    x, y = x - width // 2, y - height // 2
    source_height, source_width = image.shape[:2]
    output = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + width, source_width), min(y + height, source_height)
    if left < right and top < bottom:
        output[top-y:bottom-y, left-x:right-x] = image[top:bottom, left:right]
    return output


class MvsCapture:
    def __init__(self, config, sdk=None):
        self.config = dict(config)
        self.sdk = sdk
        self.camera = None
        self._software_roi = None
        self.initialized = self.handle_created = self.opened = self.grabbing = False

    @staticmethod
    def _check(result, operation):
        if result != 0:
            raise RuntimeError(f'MVS {operation} failed: 0x{result & 0xffffffff:08x}')

    def _load_sdk(self):
        sdk_path = self.config.get('sdk_path', '/opt/MVS/Samples/64/Python/MvImport')
        runtime = self.config.get('sdk_runtime', '/opt/MVS/lib')
        os.environ.setdefault('MVCAM_COMMON_RUNENV', runtime)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            self.sdk = importlib.import_module('MvCameraControl_class')
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f'Cannot load MVS SDK from {sdk_path}. Install the MVS SDK and '
                'check sdk_path, sdk_runtime and MVCAM_COMMON_RUNENV.'
            ) from exc

    def _configure_roi(self):
        """Always acquire 1440x1080; sensor_roi now controls software cropping."""
        self._software_roi = None
        roi = self.config.get('sensor_roi', {})
        if roi.get('enabled', False):
            values = [roi.get('offset_x', 720), roi.get('offset_y', 540),
                      roi.get('width'), roi.get('height')]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values[:2]):
                raise ValueError('sensor_roi centers must be finite numbers')
            if any(isinstance(v, bool) or not isinstance(v, int) for v in values[2:]):
                raise ValueError('sensor_roi width and height must be integers')
            x, y = (int(round(v)) for v in values[:2])
            width, height = values[2:]
            if width <= 0 or height <= 0:
                raise ValueError('sensor_roi width and height must be positive')
            x, y = x - width // 2, y - height // 2
            if x >= 1440 or y >= 1080 or x + width <= 0 or y + height <= 0:
                raise ValueError('sensor_roi does not overlap the 1440x1080 capture frame')
            self._software_roi = dict(roi)
            self._software_roi.setdefault('offset_x', 720)
            self._software_roi.setdefault('offset_y', 540)
        # Clear any ROI retained from a previous camera program. User crop
        # coordinates are never sent to the SDK, so pixel offsets like 158 work.
        for name, value in [('OffsetX', 0), ('OffsetY', 0),
                            ('Width', 1440), ('Height', 1080)]:
            self._check(self.camera.MV_CC_SetIntValue(name, value), name)
        print(f'MVS capture: 1440x1080; software sensor_roi={self._software_roi}', flush=True)

    def open(self):
        if self.sdk is None:
            self._load_sdk()
        sdk = self.sdk
        try:
            self._check(sdk.MvCamera.MV_CC_Initialize(), 'initialize SDK')
            self.initialized = True
            devices = sdk.MV_CC_DEVICE_INFO_LIST()
            self._check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_USB_DEVICE, devices),
                        'enumerate USB cameras')
            
            selected = None
            serials = []
            for index in range(devices.nDeviceNum):
                info = ct.cast(devices.pDeviceInfo[index],
                               ct.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
                serial = bytes(info.SpecialInfo.stUsb3VInfo.chSerialNumber).split(b'\0', 1)[0].decode()
                serials.append(serial)
                if serial == self.config['serial']:
                    selected = info
            if selected is None:
                raise RuntimeError(f"MVS serial {self.config['serial']!r} not found; USB cameras: {serials}")
            
            self.camera = sdk.MvCamera()
            self._check(self.camera.MV_CC_CreateHandle(selected), 'create handle')
            self.handle_created = True
            self._check(self.camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
                        'open camera')
            self.opened = True
            self._enum('TriggerMode', sdk.MV_TRIGGER_MODE_OFF)
            self._configure_roi()

            # Unlike the original wrapper's software pacing, request the sensor
            # rate too, so a faster sensor cannot build an old-frame backlog.
            self._bool('AcquisitionFrameRateEnable', True)
            self._float('AcquisitionFrameRate', self.config['fps'])
            exposure = self.config.get('exposure_time')
            if exposure is not None:
                self._enum('ExposureAuto', sdk.MV_EXPOSURE_AUTO_MODE_OFF)
                self._float('ExposureTime', exposure)
            self._enum('GainAuto', sdk.MV_GAIN_MODE_CONTINUOUS
                       if self.config.get('enable_auto_gain', True) else sdk.MV_GAIN_MODE_OFF)
            self._enum('BalanceWhiteAuto', sdk.MV_BALANCEWHITE_AUTO_CONTINUOUS
                       if self.config.get('enable_auto_white_balance', True) else sdk.MV_BALANCEWHITE_AUTO_OFF)
            if 'black_level_enable' in self.config:
                self._bool('BlackLevelEnable', self.config['black_level_enable'])
            for key, feature in [('black_level', 'BlackLevel'), ('brightness', 'Brightness')]:
                if self.config.get(key) is not None:
                    self._check(self.camera.MV_CC_SetIntValue(feature, int(self.config[key])), feature)
            self._check(self.camera.MV_CC_StartGrabbing(), 'start grabbing')
            self.grabbing = True
            return self
        except BaseException:
            self.close()
            raise

    def _enum(self, name, value):
        self._check(self.camera.MV_CC_SetEnumValue(name, value), name)

    def _bool(self, name, value):
        self._check(self.camera.MV_CC_SetBoolValue(name, bool(value)), name)

    def _float(self, name, value):
        self._check(self.camera.MV_CC_SetFloatValue(name, float(value)), name)

    def read(self):
        sdk = self.sdk
        frame = sdk.MV_FRAME_OUT()
        self._check(self.camera.MV_CC_GetImageBuffer(
            frame, int(self.config.get('frame_timeout_ms', 1000))), 'get image buffer')
            
        received = time.time()
        try:
            width, height = frame.stFrameInfo.nWidth, frame.stFrameInfo.nHeight
            if (width, height) != (1440, 1080):
                raise RuntimeError(f'MVS returned {width}x{height}; expected 1440x1080 before software crop')
            buffer = (ct.c_ubyte * (width * height * 3))()
            conversion = sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
            conversion.nWidth, conversion.nHeight = width, height
            conversion.pSrcData = frame.pBufAddr
            conversion.nSrcDataLen = frame.stFrameInfo.nFrameLen
            conversion.enSrcPixelType = frame.stFrameInfo.enPixelType
            conversion.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
            conversion.pDstBuffer = buffer
            conversion.nDstBufferSize = len(buffer)
            self._check(self.camera.MV_CC_ConvertPixelTypeEx(conversion), 'convert pixels to BGR8')
            
            image = np.ctypeslib.as_array(buffer).reshape(height, width, 3)
            # Collection stores rotated frames; convert_dataset.py crops those.
            # Thus calibration center coordinates must be applied AFTER rotation.
            if self.config.get('rotate_180', True):
                image = cv2.rotate(image, cv2.ROTATE_180)
            if self._software_roi is not None:
                image = crop_with_black_padding(image, self._software_roi)
            image = cv2.resize(image, (self.config['width'], self.config['height']),
                               interpolation=cv2.INTER_LINEAR)
            return np.ascontiguousarray(image).copy(), received
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame)

    def close(self):
        # Attempt every cleanup operation even if a disconnected camera errors.
        for flag, method in [('grabbing', 'MV_CC_StopGrabbing'),
                             ('opened', 'MV_CC_CloseDevice'),
                             ('handle_created', 'MV_CC_DestroyHandle')]:
            if getattr(self, flag):
                try:
                    self._check(getattr(self.camera, method)(), method)
                except Exception as exc:
                    warnings.warn(str(exc), RuntimeWarning)
                finally:
                    setattr(self, flag, False)
        if self.initialized:
            try:
                self._check(self.sdk.MvCamera.MV_CC_Finalize(), "finalize SDK")
            except Exception as exc:
                warnings.warn(str(exc), RuntimeWarning)
            finally:
                self.initialized = False
