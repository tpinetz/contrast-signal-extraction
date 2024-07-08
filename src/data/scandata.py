from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass
import logging
import numpy as np
import os
import pydicom
import SimpleITK as sitk

from . import utils


__all__ = ["DCMTag", "ScanData"]


# type aliases
SImage = sitk.Image
Array = np.ndarray


# TODO: define ROOT UID (maybe own required)
ROOD_UID = "2.25."


@dataclass
class DCMTag:
    """Lookup table of required DICOM tags
    """
    MediaStorageSOPInstanceUID: str =                   "0002|0002"
    ImplementationClassUID: str =                       "0002|0012"
    ImplementationVersionName: str =                    "0002|0013"
    ImageType: str =                                    "0008|0008"
    InstanceCreationDate: str =                     	"0008|0012"
    InstanceCreationTime: str =                         "0008|0013"
    SOPInstanceUID: str =                    	        "0008|0018"
    ContentDate: str =                                  "0008|0023"
    ContentTime: str =                                  "0008|0033"
    Manufacturer: str =                                 "0008|0070"
    ManufacturerModelName: str =                        "0008|1090"
    SeriesDescription: str =                            "0008|103e"
    StudyDescription: str =                             "0008|1030"
    ProtocolName: str =                                 "0018|1030"
    PatientWeight: str =                       	        "0010|1030"
    MagneticFieldStrength: str =                        "0018|0087"
    SoftwareVersions: str =                             "0018|1020"
    SecondaryCaptureDeviceManufacturer: str =           "0018|1016"
    SecondaryCaptureDeviceManufacturerModelName: str =  "0018|1018"
    SecondaryCaptureDeviceSoftwareVersions: str =       "0018|1019"
    ContrastBolusAgent: str =                           "0018|0010"
    ContrastBolusVolume: str =                          "0018|1041"
    SeriesInstanceUID: str =                            "0020|000E"
    StudyInstanceUID: str =                             "0020|000D"
    SeriesNumber: str =                                 "0020|0011"
    WindowCenter: str =                                 "0028|1050"
    WindowWidth: str =                                  "0028|1051"
    RescaleIntercept: str =                             "0028|1052"
    RescaleSlope: str =                                 "0028|1053"


class ScanData:
    """Medical data io.
    """

    def __init__(self, path: str = None, other: ScanData = None,
                 simage: sitk.Image = None) -> ScanData:
        """Load data of a scan specified by `path`.

        Supported data formats are `DICOM` or `nii`.

        Args:
            path:   (optional) Path to the scan.
            other:  (optional) ScanData to copy from
            simage: (optional) sitk image to encapsulate
        """

        self._metadata = []
        self._format = "nib"

        if path is not None and other is not None and simage is not None:
            raise RuntimeError("Specify either path, simage, or other!")
        elif path is not None:
            if os.path.isdir(path):
                self._format = "dcm"
                # read as dicom
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(reader.GetGDCMSeriesFileNames(path))
                reader.SetOutputPixelType(sitk.sitkFloat32)
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()
                self._simg = reader.Execute()

                # This is neccessary for some vida scans.
                if self._simg.GetDimension() == 4 and self._simg.GetSize()[-1] == 1:
                    self._simg = self._simg[..., 0]

                # load all the meta data for saving later on
                warning_thrown = False
                for i in range(self._simg.GetDepth()):
                    try:
                        self._metadata.append({k: reader.GetMetaData(i, k) for k in reader.GetMetaDataKeys(i)})
                    except RuntimeError as e:
                        if len(self._metadata) == 0:
                            logging.error('Could not extract metadata')
                            raise e
                        else:
                            # Something went wrong with the ordering of the dicom files
                            # and only the first meta data slice is found.
                            # This did not affect the loading of the voxel values.
                            self._metadata.append(self._metadata[0])
                            if not warning_thrown:
                                logging.warning('Runtime error on extraction of metadata using information of first slice.')
                                warning_thrown = True

            elif path.endswith("nii.gz") or path.endswith(".nii"):
                # load the image in single precision
                self._simg = sitk.ReadImage(path, outputPixelType=sitk.sitkFloat32)
            else:
                raise RuntimeError(f"Loading data from '{path}' not supported." +
                                   " Neither nii-file nor directory.")
        elif other is not None:
            # clone image (view)
            self._simg = other.sitk()
            self._metadata = copy.deepcopy(other._metadata)
            self._format = other._format
        elif simage is not None:
            self._simg = simage

    def get_metadata_by_key(self, key, result_if_missing=None):
        try:
            if key in self._metadata[0]:
                return self._metadata[0][key]
            elif result_if_missing is not None:
                return result_if_missing
            else:
                raise Exception(f'Metadata {key} is not found!')
        except IndexError:
            return result_if_missing

    def sitk(self) -> SImage:
        return self._simg

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """Converts data to `np.ndarray`.

        Args:
            copy: (`True`) if `True` array is copied

        Returns:
            np.ndarray: 3D volume (z, y, x).
        """
        if copy:
            return sitk.GetArrayFromImage(self._simg)
        else:
            return sitk.GetArrayViewFromImage(self._simg)

    def to_isotropic(self, largest_spacing: float = 0.5, fft: bool = False) -> SImage:
        """Resamples the scan data to isotropic voxels for a given spacing.

        Params:
            largest_spacing: largest spacing for the resized image
            fft: if True the spacing and shape are adapted using corpping
                and padding in Fourier domain

        Returns:
            sitk image
        """
        old_spacing = np.asarray(self._simg.GetSpacing(), dtype=np.float32)
        old_size = np.asarray(self._simg.GetSize(), dtype=np.int32)

        new_spacing = np.ones((3,), dtype=np.float32) * largest_spacing
        scaling_factors = old_spacing / new_spacing
        new_size = np.ceil(old_size * scaling_factors)
        tmp = 8 - np.mod(new_size, 8)
        tmp[tmp == 8] = 0
        new_size += tmp  # ensure multiple of 8

        method = "fft" if fft else "splines"
        logging.info(f"Resampling image from {old_spacing}/{old_size} to {new_spacing}/{new_size} using {method}.")

        if fft:
            # correct the scaling factors and update the spacing
            scaling_factors = old_size / new_size
            new_spacing = old_spacing * scaling_factors

            diff = new_size[::-1] - old_size[::-1]  # dimensions are reversed in numpy
            pad_pre, pad_post = np.ceil(diff / 2).astype(np.int32), np.floor(diff / 2).astype(np.int32)
            pad_pre, pad_post = np.maximum(0, pad_pre), np.maximum(0, pad_post)
            crop_pre, crop_post = np.floor(-diff / 2).astype(np.int32), np.ceil(-diff / 2).astype(np.int32)
            crop_pre, crop_post = np.maximum(0, crop_pre), np.maximum(0, crop_post)

            x = sitk.GetArrayFromImage(self._simg)
            x_k = utils.fft(x)
            x_iso_k = np.pad(x_k, np.stack((pad_pre, pad_post), 1), mode="constant", constant_values=0)
            D, H, W = x_iso_k.shape[-3:]
            x_iso_k = x_iso_k[...,
                              crop_pre[0]:D-crop_post[0],
                              crop_pre[1]:H-crop_post[1],
                              crop_pre[2]:W-crop_post[2]]
            x_iso = utils.ifft(x_iso_k)
            x_iso *= np.prod(x_iso.shape[-3:]) / np.prod(x.shape[-3:])

            simg_iso = sitk.GetImageFromArray(x_iso.astype(np.float32))
            simg_iso.SetOrigin(self._simg.GetOrigin())
            simg_iso.SetDirection(self._simg.GetDirection())
            simg_iso.SetSpacing(new_spacing.astype(np.float64))

        else:
            simg_iso = sitk.Resample(
                self._simg,
                size=new_size.astype("int").tolist(),
                transform=sitk.Transform(),
                interpolator=sitk.sitkBSpline,
                outputOrigin=self._simg.GetOrigin(),
                outputSpacing=new_spacing.astype("float").tolist(),
                outputDirection=self._simg.GetDirection(), defaultPixelValue=0,
                outputPixelType=self._simg.GetPixelID()
            )
            tmp = sitk.GetArrayViewFromImage(self._simg)
            simg_iso = sitk.Clamp(
                simg_iso,
                simg_iso.GetPixelIDValue(),
                lowerBound=float(tmp.min()),
                upperBound=float(tmp.max())
            )

        return simg_iso, scaling_factors

    def resample_to(self, other: SImage, fft: bool = False) -> SImage:
        """Resamples the other image to the spacing and size of scan data.

        Params:
            other: sitk image at the same origin and direction but with different spacing
            fft: if True the spacing and shape are adapted using corpping
                and padding in Fourier domain

        Returns:
            sitk image
        """
        assert self._simg.GetOrigin() == other.GetOrigin()
        assert self._simg.GetDirection() == other.GetDirection()

        new_spacing = np.asarray(self._simg.GetSpacing(), dtype=np.float32)
        old_spacing = np.asarray(other.GetSpacing(), dtype=np.float32)

        new_size = np.asarray(self._simg.GetSize(), dtype=np.int32)
        old_size = np.asarray(other.GetSize(), dtype=np.int32)

        method = "fft" if fft else "splines"
        logging.info(f"Resampling image from {old_spacing}/{old_size} to {new_spacing}/{new_size} using {method}.")

        if fft:
            diff = old_size[::-1] - new_size[::-1]  # dimensions are reversed in numpy
            pad_pre, pad_post = np.ceil(diff / 2).astype(np.int32), np.floor(diff / 2).astype(np.int32)
            pad_pre, pad_post = np.maximum(0, pad_pre), np.maximum(0, pad_post)
            crop_pre, crop_post = np.floor(-diff / 2).astype(np.int32), np.ceil(-diff / 2).astype(np.int32)
            crop_pre, crop_post = np.maximum(0, crop_pre), np.maximum(0, crop_post)

            x = sitk.GetArrayFromImage(other)
            x_k = np.fft.fftshift(np.fft.fftn(x))
            x_lr_k = np.pad(x_k, np.stack((crop_pre, crop_post), 1), mode="constant", constant_values=0)
            D, H, W = x_lr_k.shape[-3:]
            x_lr_k = x_lr_k[...,
                            pad_pre[0]:D-pad_post[0],
                            pad_pre[1]:H-pad_post[1],
                            pad_pre[2]:W-pad_post[2]]
            x_lr = np.fft.ifftn(np.fft.ifftshift(x_lr_k)).real
            x_lr *= np.prod(x_lr.shape[-3:]) / np.prod(x.shape[-3:])

            simg_lr = sitk.GetImageFromArray(x_lr.astype(np.float32))
            simg_lr.SetOrigin(self._simg.GetOrigin())
            simg_lr.SetDirection(self._simg.GetDirection())
            simg_lr.SetSpacing(new_spacing.astype(np.float64))

        else:
            simg_lr = sitk.Resample(
                other,
                size=new_size.astype("int").tolist(),
                transform=sitk.Transform(),
                interpolator=sitk.sitkBSpline,
                outputOrigin=self._simg.GetOrigin(),
                outputSpacing=new_spacing.astype("float").tolist(),
                outputDirection=self._simg.GetDirection(), defaultPixelValue=0,
                outputPixelType=self._simg.GetPixelID()
            )
            tmp = sitk.GetArrayViewFromImage(other)
            simg_lr = sitk.Clamp(
                simg_lr,
                simg_lr.GetPixelIDValue(),
                lowerBound=float(tmp.min()),
                upperBound=float(tmp.max())
            )

        return simg_lr

    def to_resolution(self, new_spacing: np.ndarray, smooth: float = 0) -> SImage:
        """Resamples the scan data to isotropic voxels.

        Returns:
            sitk image
        """
        assert self._simg.GetDimension() == 3

        old_spacing = np.asarray(self._simg.GetSpacing(), dtype=np.float32)
        mask = np.isclose(old_spacing, new_spacing, rtol=0.15, atol=0)
        new_spacing[mask] = old_spacing[mask]

        if np.all(mask):
            logging.info("Skipping resampling.")
            return self._simg, (1.0, 1.0, 1.0)
        else:
            logging.info(f"Resampling image from {old_spacing} to isotropic voxels {new_spacing}.")
            scaling_factors = new_spacing / old_spacing
            new_size = (np.asarray(self._simg.GetSize()) / scaling_factors)

            if smooth:
                sigma = np.maximum(scaling_factors-1, 0) * smooth + 1e-3
                logging.info(f"Smoothing image using sigma={sigma}.")
                simg = sitk.SmoothingRecursiveGaussian(self._simg, sigma)
            else:
                simg = self._simg

            iso = sitk.Resample(simg, size=tuple(int(s) for s in new_size),
                                transform=sitk.Transform(), interpolator=sitk.sitkBSpline,
                                outputOrigin=self._simg.GetOrigin(), outputSpacing=new_spacing,
                                outputDirection=self._simg.GetDirection(), defaultPixelValue=0,
                                outputPixelType=self._simg.GetPixelID())
            tmp = sitk.GetArrayViewFromImage(self._simg)
            return (sitk.Clamp(iso, iso.GetPixelIDValue(), lowerBound=float(tmp.min()), upperBound=float(tmp.max())),
                    tuple(scaling_factors))

    def to_smaller_resolution(self, new_size: np.ndarray) -> SImage:
        """Resamples the scan data to isotropic voxels.

        Returns:
            sitk image
        """
        assert self._simg.GetDimension() == 3

        old_spacing = np.asarray(self._simg.GetSpacing(), dtype=np.float32)
        old_size = np.asarray(self._simg.GetSize())
        scaling_factors = np.array([old / new for old, new in zip(old_size, new_size)])
        new_spacing = np.array([space * scaling_factor for space, scaling_factor in zip(old_spacing, scaling_factors)])
        logging.info(f"Resampling image from {old_spacing} to isotropic voxels {new_spacing}.")
        iso = sitk.Resample(self._simg, size=tuple(int(s) for s in new_size),
                            transform=sitk.Transform(), interpolator=sitk.sitkBSpline,
                            outputOrigin=self._simg.GetOrigin(), outputSpacing=new_spacing,
                            outputDirection=self._simg.GetDirection(), defaultPixelValue=0,
                            outputPixelType=self._simg.GetPixelID())
        tmp = sitk.GetArrayViewFromImage(self._simg)
        return (sitk.Clamp(iso, iso.GetPixelIDValue(), lowerBound=float(tmp.min()), upperBound=float(tmp.max())),
                tuple(scaling_factors))

    def from_numpy(self, array: np.ndarray) -> ScanData:
        """Stores the `array` into the image data.

        Args:
            array: New pixel values of shape (z, y, x)
        """

        # generate sitk image
        simage = sitk.GetImageFromArray(array)
        # copy image properties (origin, direction, spacing)
        simage.CopyInformation(self._simg)
        self._simg = simage
        return self

    def from_sitk(self, simage: sitk.Image) -> ScanData:
        """Stores the `sitk.image` into the image data.

        Args:
            simage: New values
        """

        if self._simg.GetSize() != simage.GetSize():
            raise RuntimeError("Image can only be replaced by images of the same size!")
        self._simg = simage
        return self

    def save(self, path: str, format: str = None,
             description: str = "", series: int = 999, version: str = "",
             window: float = -1, study_uid=None,  extraTags: dict = {},
             **kwargs) -> ScanData:
        """Saves scan data to `path` where each z slice is saved as a `DICOM` file.

        The `DICOM`-fields are maintained from the original file set.

        Args:
            path: Target path.
            format: (None) image format {'nib', 'dcm'}
            description: ('') Detailed description of the series
            series: (999) Scan series
            version: ('') Version string to store along the data.
            kwargs: key/value pairs for `DICOM` files.
        """

        if format is None:
            # use original format if not specified
            format = self._format

        if format == "nib":
            sitk.WriteImage(self._simg, path + ".nii.gz", useCompression=True, compressionLevel=5)
        else:
            if self._format != "dcm":
                raise RuntimeError("Saving as DICOM is only supported if data is read from DICOM first!")

            if not os.path.exists(path):
                os.makedirs(path)

            # new series UID
            series_uid = pydicom.uid.generate_uid(prefix=ROOD_UID)

            exclude_from_copy = {DCMTag.ImageType,
                                 DCMTag.Manufacturer,
                                 DCMTag.SoftwareVersions}

            image_type = ["DERIVED", "SECONDARY"] + self._metadata[0][DCMTag.ImageType].split("\\")[2:]
            image_type = "\\".join(image_type)

            # determine series description
            series_description = [i for i in self._metadata[0][DCMTag.ImageType].split(" ") if len(i) > 0]
            series_description = " ".join(series_description[:-1] + [description])

            # determine the contrast window
            if window < 0.:
                tmp = sitk.GetArrayViewFromImage(self._simg)
                window = tmp.max() - tmp.min()

            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            for i in range(self._simg.GetDepth()):
                slice = self._simg[:, :, i]
                if len(self._metadata[i]) == 0:
                    raise RuntimeError("Meta data is required to write DICOMs.")
                # copy all tags from the original source
                for k, v in self._metadata[i].items():
                    if k not in exclude_from_copy:
                        slice.SetMetaData(k, v.encode('utf-8', 'replace').decode())

                for k, v in extraTags.items():
                    slice.SetMetaData(k, v)

                # set custom fields
                for k, v in kwargs.items():
                    slice.SetMetaData(k, v)

                if DCMTag.StudyDescription not in self._metadata:
                    slice.SetMetaData(DCMTag.StudyDescription, "MR-Hirn")

                slice.SetMetaData(DCMTag.SeriesDescription, series_description)
                slice.SetMetaData(DCMTag.ProtocolName, series_description)

                # update required fields
                slice.SetMetaData(DCMTag.ImageType, image_type)
                slice.SetMetaData(DCMTag.WindowCenter, str(window/2))
                slice.SetMetaData(DCMTag.WindowWidth, str(window))

                # manufacturer
                slice.SetMetaData(DCMTag.SecondaryCaptureDeviceManufacturer, "ukb")
                slice.SetMetaData(DCMTag.SecondaryCaptureDeviceManufacturerModelName, "Smart Contrast")
                slice.SetMetaData(DCMTag.SecondaryCaptureDeviceSoftwareVersions, version)

                # creation date/time
                dt = datetime.datetime.now()
                date = dt.strftime("%Y%m%d")
                time = dt.strftime("%H%M%S.%f")
                slice.SetMetaData(DCMTag.InstanceCreationDate, date)
                slice.SetMetaData(DCMTag.InstanceCreationTime, time)
                slice.SetMetaData(DCMTag.ContentDate, date)
                slice.SetMetaData(DCMTag.ContentTime, time)

                # set UID tags
                slice.SetMetaData(DCMTag.SeriesInstanceUID, series_uid)
                if study_uid is not None:
                    slice.SetMetaData(DCMTag.StudyInstanceUID, study_uid)
                sop_uid = pydicom.uid.generate_uid(prefix=ROOD_UID)
                slice.SetMetaData(DCMTag.SOPInstanceUID, sop_uid)
                slice.SetMetaData(DCMTag.MediaStorageSOPInstanceUID, sop_uid)

                # TODO: update version ID
                slice.SetMetaData(DCMTag.ImplementationClassUID, ROOD_UID + "0")
                slice.SetMetaData(DCMTag.ImplementationVersionName, version)

                slice.SetMetaData(DCMTag.SeriesNumber, f"{series:d}")

                # TODO: update private tags to store registration RMSE and other internal results

                writer.SetFileName(os.path.join(path, f"{i+1:08X}.dcm"))
                writer.Execute(slice)

        return self
