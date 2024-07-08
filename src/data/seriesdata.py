from __future__ import annotations

from dataclasses import field
import logging
import numpy as np
import os
import pydicom
import SimpleITK as sitk
from skimage.filters import gaussian, laplace, sobel, apply_hysteresis_threshold, median
from statsmodels.robust.scale import Huber
import skimage.morphology as morphology

from .scandata import DCMTag, ScanData
from .cgmm import fit_cgmm
from .utils import scan_mask, estimate_local_mean_std, sigmoid
from . import registration
from . import normalization


__all__ = ["ScanEntry", "SeriesData"]


ATLAS_SMOOTHING = 2.  # std of Gaussian smoothing kernel in [mm]
ATLAS_MASK_DILATION = 15  # dilation size in [mm]
RADIOMETRIC_REG_ALPHA = 0.05
RADIOMETRIC_REG_QUANTILE = 0.8


# Huber robust estimation
huber = Huber(c=1.5, maxiter=50)


# type aliases
SImage = sitk.Image


class ScanEntry:
    """ Entries for a specific scan of a series. """
    data: np.ndarray
    attrs: dict(str, object) = field(default_factory=lambda: {})

    def __init__(self, data: np.ndarray = None, attrs: dict = None,
                 simage: SImage = None, mask: np.ndarray = None):
        if data is not None and attrs is not None:
            self.data = data
            self.attrs = attrs
        elif simage is not None:
            self.data = sitk.GetArrayFromImage(simage).astype(np.float32)

            if mask is not None:
                vmax = np.quantile(self.data[mask > normalization.Nyul.ROI_THRESHOLD],
                                   normalization.Nyul.VMAX_QUANTILE).astype(np.float32)
            else:
                vmax = np.quantile(self.data[self.data > 1e-4], normalization.Nyul.VMAX_QUANTILE).astype(np.float32)

            self.attrs = {
                "origin": simage.GetOrigin(),
                "spacing": simage.GetSpacing(),
                "direction": simage.GetDirection(),
                "vmax": vmax,
            }
        else:
            raise RuntimeError("Specify either data and attrs or an sitk image!")

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def sitk(self) -> SImage:
        simage = sitk.GetImageFromArray(self.data)
        simage.SetOrigin(self.attrs["origin"])
        simage.SetSpacing(self.attrs["spacing"])
        simage.SetDirection(self.attrs["direction"])
        return simage


class SeriesData:
    """ Relevant data of a series.
    """

    # Scan name definitions
    zero = "T1_zero"
    low = "T1_low"
    target = "T1_full"
    flair = "Flair"
    t2 = "T2"
    diff = "Diff"
    mask = "mask"
    mask_atlas = "mask_atlas"
    diff_median = "diff_median"
    p_signal = "prob_signal"

    spacing_threshold = 0.75

    # Values taken from paper Shen et al. "T1 Relaxivities of Gadolinium-Based Magnetic Resonance
    # Contrast Agents in Human Whole Blood at 1.5, 3, and 7 T"
    contrastagent_to_relaxivity = {
        'dotarem':      [3.9, 3.4],
        'gadovist':     [4.6, 4.5],
        'gadobutrol':   [4.6, 4.5],   # same as gadovist
        'prohance':     [4.4, 3.5],
        'clariscan':    [3.9, 3.4]
    }

    # Flags indidacting whether image gradients are used during
    # rigid registration.
    rr_grad = {
        "T1_low":   False,
        "T1_full":  False,
        "Diff":     True,
        "T2":       True,
    }

    def __init__(self, mask_threshold: float = -1,
                 register_images: bool = False,
                 isotropic_regridding: bool = True,
                 fft_regridding: bool = False,
                 mandatory: list = ["T1_low"],
                 crop_to_brain: bool = False,
                 fast: bool = False):
        """ Collection of relevant data of a series.

        Args:
            mask_thresold: (0) Threshold to determine the initial mask on the nateive scan.
            register_images: if True the images are registered onto the native scan.
            isotropic_regridding: (True) if True the images are regirdded to 0.5mm^3 or 1mm^3 grids
            mandatory: (["T1_low"]) list of mandatory inputs
            crop_to_brain: (False) if True the entries are cropped to the brain region
            fast: Inference Code to speed up test script.
        """
        self._register_images = register_images
        self._mask_threshold = mask_threshold
        self._isotropic_regridding = isotropic_regridding
        self._fft_regridding = fft_regridding
        self._mandatory = mandatory
        self._crop_to_brain = crop_to_brain
        self.fast = fast

        atlas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas")
        self._atlas_t1 = sitk.ReadImage(os.path.join(atlas_path, "atlas_t1.nii"), outputPixelType=sitk.sitkFloat32)
        self._atlas_t1 = sitk.Multiply(self._atlas_t1, 1/sitk.GetArrayViewFromImage(self._atlas_t1).max())
        self._atlas_mask = sitk.ReadImage(os.path.join(atlas_path, "atlas_mask.nii"), outputPixelType=sitk.sitkUInt8)

        # Set of scan entries
        self._s = {}

    def load(self, path: str = None, input_files: dict[str, str] = None) -> SeriesData:
        """ Load required data.

        Args:
            path: (optinal) Path to DICOM series
            input_files: (optional) Dictionary that maps each required key to a filename.
        """

        if (path is None and input_files is None) or (path is not None and input_files is not None):
            raise RuntimeError("Specify either path or inputs_files!")

        if path:
            input_files = SeriesData.find_dicom_series_from_path(path)

        logging.info("Loading zero")
        # store zero scan (=refernce coordinate system) for later writing
        self._ref_data = ScanData(input_files[SeriesData.zero])

        if self._isotropic_regridding:
            # resample zero data to isotropic voxels
            s_zero, scaling_factors = self._ref_data.to_isotropic(fft=self._fft_regridding)
            self._ref_data_isotropic = s_zero
        else:
            s_zero = self._ref_data.sitk()
            scaling_factors = (1., 1., 1.)

        self._s[SeriesData.zero] = ScanEntry(simage=s_zero)
        self._s[SeriesData.zero].attrs["scaling_factors"] = scaling_factors

        logging.info("Computing initial mask")
        if not self.fast:
            if self._mask_threshold >= 0:
                mask = scan_mask(s_zero > self._mask_threshold)
            else:
                mask = np.ones(self._s[SeriesData.zero].shape, dtype=np.uint8)
                mask = sitk.GetImageFromArray(mask)
                mask.CopyInformation(s_zero)

        logging.info("Computing mask by atlas registration")
        # Register zero dose to mask first, then invert the transform.
        #   - This behaviour is motivated by the fact that the atlas defines the desired
        #     regions (it does not include the mouth and shoulders). Thus, sampling in the
        #     fixed image during registration is limited to these regions, which leads to
        #     more robust results.
        #   - The zero dose image is smoothed in order to refelct the smooth bone structure
        #     of the atlas.
        s_zero_smooth = sitk.SmoothingRecursiveGaussian(s_zero, ATLAS_SMOOTHING)
        transform_mask = registration.spatial(
            self._atlas_t1,
            s_zero_smooth,
            scale_fixed=1.0,
            scale_moving=self._s[SeriesData.zero].attrs["vmax"],
            transform="affine"
        )
        # use linear interpolation to avoid interpolation artifacts for masks
        mask_atlas = registration.resample(self._atlas_mask, s_zero, transform_mask.GetInverse(),
                                           interpolation="linear")
        # Increase atlas mask in each direction to ensure that brain edge is included.
        # This is helpful for the spatial registration.
        min_spacing = min(mask_atlas.GetSpacing())
        mask_atlas_reg = sitk.BinaryDilate(mask_atlas,
                                           [int(np.ceil(ATLAS_MASK_DILATION/min_spacing))]*mask_atlas.GetDimension(),
                                           sitk.sitkBall)

        self._s[SeriesData.mask_atlas] = ScanEntry(simage=mask_atlas > 1e-4)
        np_mask = sitk.GetArrayViewFromImage(mask_atlas) > 1e-4

        self._s[SeriesData.zero].attrs["vmax"] = np.quantile(self._s[SeriesData.zero].data[np_mask],
                                                             normalization.Nyul.VMAX_QUANTILE)

        self._s[SeriesData.zero].attrs["landmarks"] = normalization.Nyul.compute_landmarks(
            self._s[SeriesData.zero].data / self._s[SeriesData.zero].attrs["vmax"],
            np_mask
        )

        for scan in (self._mandatory if self.fast else self._mandatory + [SeriesData.target]):
            if scan in input_files:
                scan_data = ScanData(input_files[scan])
            else:
                if scan != SeriesData.target:
                    raise KeyError(scan)
                else:
                    continue
            logging.info(f"Loading {scan} " + "-" * 20)

            if scan_data.sitk().GetDimension() < 3:
                raise RuntimeError("Expected at least 3D scan!")

            if self._isotropic_regridding and self._fft_regridding:
                s_img, _ = scan_data.to_isotropic(fft=self._fft_regridding)
            else:
                s_img = scan_data.sitk()

            N = 1 if s_img.GetDimension() == 3 else s_img.GetSize()[-1]
            for i in range(N):
                scan_i = scan + f"{i}" if N > 1 else scan
                s_img_i = s_img[..., i] if N > 1 else s_img

                if "T1" in scan_i:
                    # allow 10% tolerance in spacing
                    if not np.allclose(scan_data.sitk().GetSpacing(), self._ref_data.sitk().GetSpacing(), atol=0, rtol=0.1):
                        logging.error(f"Spacing of {scan} '{scan_data.sitk().GetSpacing()}' differs from reference '" +
                                      f"{self._ref_data.sitk().GetSpacing()}'")
                        raise RuntimeError("Spacing missmatch")

                if self._register_images:
                    transform = registration.spatial(
                        s_zero, s_img_i,
                        fixed_mask=mask_atlas_reg,
                        use_gradient_abs=SeriesData.rr_grad[scan]
                    )
                else:
                    transform = sitk.Transform()
                s_img_i_reg = registration.resample(s_img_i, s_zero, transform)

                self._s[scan_i] = ScanEntry(simage=s_img_i_reg, mask=np_mask)

                if not self.fast:
                    tmp_mask = np.ones_like(sitk.GetArrayViewFromImage(s_img_i))
                    tmp_mask = sitk.GetImageFromArray(tmp_mask)
                    tmp_mask.CopyInformation(s_img_i)
                    # use linear interpolation to avoid interpolation artifacts for masks
                    tmp_mask = registration.resample(tmp_mask, s_zero, transform, interpolation="linear") > 1e-4
                    mask = sitk.And(mask, tmp_mask)

                np_zero = self._s[SeriesData.zero].data / self._s[SeriesData.zero].attrs["vmax"]
                if "T1" in scan_i:
                    # perform radiometric registration on T1 scans
                    np_reg = self._s[scan_i].data / self._s[SeriesData.zero].attrs["vmax"]
                    res = registration.radiometric(
                        np_zero, np_reg,
                        mask=np_mask,
                        alpha=RADIOMETRIC_REG_ALPHA,
                        diff_quantile=RADIOMETRIC_REG_QUANTILE
                    )
                    np_reg = res["out"]
                    self._s[scan_i].attrs["scale"] = res["params"]
                    self._s[scan_i].attrs["RR E0"] = res["energy"][0]
                    self._s[scan_i].attrs["RR E-1"] = res["energy"][-1]
                else:
                    np_reg = self._s[scan_i].data / self._s[scan_i].attrs["vmax"]

                # compute RMSE on overlapping regions
                rmse = np.sqrt(np.mean((np_reg[np_mask] - np_zero[np_mask])**2))
                self._s[scan_i].attrs["reg-RMSE"] = rmse
                logging.info(f"Final RMSE={rmse}")

                if scan_i.startswith("T1_low"):
                    self._s[scan_i].attrs["weight"] = float(scan_data.get_metadata_by_key(DCMTag.PatientWeight, 0))
                    if self._s[scan_i].attrs["weight"] == 0:
                        # I am using the mean weight 80 in case no weight is given.
                        self._s[scan_i].attrs["weight"] = 80
                        logging.warning(f'No patient weight found, set patient weight: {self._s[scan_i].attrs["weight"]}')
                    else:
                        logging.info(f'Found Patient weight: {self._s[scan_i].attrs["weight"]}')

                    self._s[scan_i].attrs["dose"] = float(scan_data.get_metadata_by_key(DCMTag.ContrastBolusVolume, 0.))
                    contrastagent = scan_data.get_metadata_by_key(DCMTag.ContrastBolusAgent, '').lower()

                    if contrastagent:
                        self._s[scan_i].attrs["contrastagent"] = contrastagent.strip().lower()
                        logging.info(f'Found contrast agent {contrastagent}')
                    else:
                        self._s[scan_i].attrs["contrastagent"] = "dotarem"
                        logging.info('Set contrastagent in absence of dose information to dotarem.')

                    # Sometimes the full dose is entered into the dicom and we handle this now.
                    if self._s[scan_i].attrs["dose"] == 0 or self._s[scan_i].attrs["dose"] > 0.06 * self._s[scan_i].attrs["weight"]:
                        # No one injects between 0.0ml and 0.1 ml so rounding to 1 dezimal makes sense.
                        self._s[scan_i].attrs["dose"] = max(np.round(self._s[scan_i].attrs["weight"] * 0.033, 1), 2.0)
                        if self._s[scan_i].attrs["contrastagent"] not in ['gadovist',  'gadobutrol']:
                            self._s[scan_i].attrs["dose"] *= 2

                        logging.warning(f'Set contrast agent dose in absence of dicom information to: {self._s[scan_i].attrs["dose"]}')
                    else:
                        logging.info(f'Found contrast agent dose: {self._s[scan_i].attrs["dose"]}')

                    # percent of standard dose is ml * 10 / weight. To get to percent we multiply by 100.
                    # gadovist is the only 0.1 mmol/ml and so the factor is doubled.
                    factor = 10 * 100 / (2 - int('gadovist' in self._s[scan_i].attrs["contrastagent"]) - int('gadobutrol' in self._s[scan_i].attrs["contrastagent"]))
                    self._s[scan_i].attrs["percent"] = (self._s[scan_i].attrs["dose"] * factor
                                                        / self._s[scan_i].attrs["weight"])
                    logging.info(f'Calculated contrast agent percent: {self._s[scan_i].attrs["percent"]}')
                    self._s[scan_i].attrs["field strength"] = float(scan_data.get_metadata_by_key(DCMTag.MagneticFieldStrength, 3.))
                    logging.info(f'Found field strength: {self._s[scan_i].attrs["field strength"]}')

                    self._s[scan_i].attrs["manufacturer"] = str(scan_data.get_metadata_by_key(DCMTag.Manufacturer, "unknown"))
                    self._s[scan_i].attrs["model"] = str(scan_data.get_metadata_by_key(DCMTag.ManufacturerModelName, "unknown"))
                    logging.info(f"Manufacturer: {self._s[scan_i].attrs['manufacturer']}, Model: {self._s[scan_i].attrs['model']}")

        if not self.fast:
            self._s[SeriesData.mask] = ScanEntry(simage=mask > 1e-4)

        if self._crop_to_brain:
            padding = 32
            # extract region of interest
            np_mask = self._s[SeriesData.mask_atlas].data
            rois_init = (np.argwhere(np.any(np_mask, axis=(1, 2))),
                         np.argwhere(np.any(np_mask, axis=(0, 2))),
                         np.argwhere(np.any(np_mask, axis=(0, 1))))
            # Change to even tuple as we only use even values for the maximum shape and the origin should be consistent.
            rois = tuple([slice(np.maximum(sl[0, 0]-padding, 0) // 2 * 2,
                                np.minimum(sl[-1, 0]+padding+1, np_mask.shape[i]) // 2 * 2)
                          for i, sl in enumerate(rois_init)])
            logging.info(f"Cropping from {np_mask.shape} to {rois}")
            # apply the region of interest
            for scan, entry in self._s.items():
                self._s[scan].data = entry.data[rois]

        # estimate the noise level in the brain
        self.estimate_noise_statistics()

        return self

    def estimate_noise_statistics(self) -> None:
        """ For every T1 scan estimate the noise statistics.

        Args:
        """

        def stats(x):
            lhs = x[x < 0]
            sigma = np.sqrt(np.mean(lhs**2))
            sigma_h = huber(lhs, mu=np.asarray(0), initscale=.9*sigma)[1]
            kurtosis = np.mean(lhs**4)/sigma_h**4 - 3
            return sigma, sigma_h, kurtosis

        T1_zero = sitk.Multiply(self._s[self.zero].sitk(), 1 / self._s[self.zero].attrs["vmax"])
        mask_atlas = self._s[self.mask_atlas].data

        for scan in [self.low, self.target]:
            if scan not in self._s.keys():
                continue

            T1_scan = sitk.Multiply(self._s[scan].sitk(),
                                    self._s[scan].attrs["scale"] / self._s[self.zero].attrs["vmax"])
            diff = T1_scan - T1_zero
            data = sitk.GetArrayViewFromImage(diff)[mask_atlas > 0]
            sigma, sigma_h, kurt = stats(data)
            self._s[scan].attrs["sigma"] = sigma
            self._s[scan].attrs["sigma_h"] = sigma_h
            self._s[scan].attrs["kurtosis"] = kurt
            logging.info(f"{scan} difference std estimated as {sigma:.3e}/{sigma_h:.3e} and kurtosis {kurt:.3e}")

            mean, std = estimate_local_mean_std(sitk.GetArrayViewFromImage(diff), sigma_est=sigma_h)
            mean = sitk.GetImageFromArray(mean)
            mean.CopyInformation(diff)
            self._s[scan + "_mean"] = ScanEntry(simage=mean)
            std = sitk.GetImageFromArray(std)
            std.CopyInformation(diff)
            self._s[scan + "_std"] = ScanEntry(simage=std)
            logging.info(f"{scan} local mean and standard deviation estimated")

    def load_h5(self, group, selector=None, target: bool = True) -> SeriesData:
        if selector is None:
            selector = lambda x: x[:]
        scans = [self.zero, self.mask, self.mask_atlas] + self._mandatory + \
            [s + "_mean" for s in [self.low, self.target]] + \
            [s + "_std" for s in [self.low, self.target]]
        if target:
            scans += [self.target]
        for scan in scans:
            try:
                self._s[scan] = ScanEntry(
                    data=selector(group[scan]),
                    attrs={k: v for k, v in group[scan].attrs.items()}
                )
            except KeyError as e:
                # This enables us to find bugs easier.
                logging.error(f'KeyError with {group} on scans {scans}')
                raise e
        self._ref_data = ScanData(simage=self._s[self.zero].sitk())
        self._ref_data_isotropic = None
        return self

    def isfinite(self) -> bool:
        for v in self._s.values():
            if not np.all(np.isfinite(v.data)):
                return False
        return True

    def __len__(self) -> int:
        return len(self._s)

    def __getitem__(self, k: str) -> ScanEntry:
        return self._s[k]

    def __contains__(self, key):
        return key in self._s

    def keys(self) -> list:
        return self._s.keys()

    def items(self) -> list:
        return self._s.items()

    def grad_mag(self, image):
        return np.sqrt(sum([sobel(image, axis=i)**2 for i in range(image.ndim)]) / image.ndim)

    def fill_data(self, data, patch_size):
        z, h, w = data.shape

        diffh = patch_size - h
        diffw = patch_size - w
        padh = 0
        padw = 0
        if diffh < 0:
            data[:, -diffh//2:diffh//2 + diffh % 2]
            diffh = 0
        else:
            padh = diffh // 2
        if diffw < 0:
            data[:, :, -diffw//2:diffw//2 + diffw % 2]
            diffw = 0
        else:
            padw = diffw // 2

        data = np.pad(data, ((0, 0), (padh, padh + diffh % 2), (padw, padw + diffw % 2)))

        return data.astype(np.float32)

    def to_input(self, cfg: dict, training=False) -> dict:
        """
        Args:
            cfg: Data configuration dictionary with the following keys:
                - inputs: List of inputs to the model.
                - radio_reg_input (optional): If True low-dose input is scaled by radiometric registration scale.
                - sub_target (optinal): If True the target is defined as the subtraction image.
                - sub_threshold (optional): Threshold for the target difference image. Defaults to $-\\infty$.
                - radio_reg_target (optional): If True full-dose target is scaled by radiometric registration scale.
                - add_grad (optional): Adds the smoothed gradient of the target image to the sample.
                - add_smoth (optional): Adds weights to focus loss on large scale CA signals.
                - adaptive_threshold (optional): Adds an adaptive loss threshold image to the sample. All values above
                    the threshold are mapped to zero and there is a linear transition from the treshold towards zero.
            training: Flag controls if this is called during training for additional augmentation or during inference. 
        Return:
            Dictionary of numpy arrays and keys:
                - data: Input data to the model
                - mask: Mask indicating the valid image regions
                - target (optional): Target
                - grad (optinal): Gradient of the target image
                - smooth (optinal): Weights for large scale CA signals
                - threshold (optional): Adaptive loss threshold
        """

        logging.debug('Mask Generation')
        sample = {}
        if SeriesData.mask in self._s:
            sample["mask"] = self._s[SeriesData.mask].data[..., np.newaxis].astype(np.float32)
        if SeriesData.mask_atlas in self._s:
            sample["mask_atlas"] = self._s[SeriesData.mask_atlas].data[..., np.newaxis].astype(np.float32)

        np_reference = self._s[SeriesData.zero].data[..., np.newaxis].copy()
        # define the contrast
        reference_scale = self._s[SeriesData.zero].attrs["vmax"]
        if training:
            if "augmentation_probability" in cfg and np.random.uniform() < cfg["augmentation_probability"]:
                reference_scale += np.random.normal(0, reference_scale * 0.05)
        np_reference /= reference_scale

        if "nyul_normalization" in cfg and cfg["nyul_normalization"]:
            standard_scale = self.get_standard_scale(cfg)
            normalize = lambda x: normalization.Nyul.normalize(x, standard_scale, landmarks=self._s[SeriesData.zero].attrs["landmarks"])
        else:
            normalize = lambda x: x

        np_reference = normalize(np_reference)
        np_inputs = []
        base_index = cfg["inputs"].index(SeriesData.low)

        logging.debug('Generating input arrays')
        for inp in cfg["inputs"]:
            if inp in self._s.keys():
                np_input = self._s[inp].data[..., np.newaxis].copy()
                if "T1" in inp and "std" not in inp:
                    if "radio_reg_input" in cfg and cfg["radio_reg_input"]:
                        np_input = np_input * self._s[inp].attrs["scale"]
                    np_input /= reference_scale
                    np_input = normalize(np_input)
                elif "std" in inp:
                    np_input *= 10  # increase range since the noise standard deviation is usually rather small
                else:
                    np_input /= self._s[inp].attrs["vmax"]
                np_inputs.append(np_input)
            else:
                raise KeyError(f"Cannot identify '{inp}'!")

        # prepend initial diff image
        if "preprocessed_low" in cfg and cfg["preprocessed_low"]:
            np_diff = (
                np.maximum(median(np_inputs[base_index][..., 0], morphology.ball(1)), np_inputs[base_index][..., 0]) -
                np.minimum(median(np_reference[..., 0], morphology.ball(1)), np_reference[..., 0])
            )[..., np.newaxis]
        else:
            np_diff = np_inputs[base_index] - np_reference

        if "sub_low" in cfg and cfg["sub_low"]:
            np_diff = np.maximum(np_diff, 0)

        # The LD to the SD signal is approximately the same.
        np_inputs.insert(0, np_diff)
        base_index += 1

        sample["data"] = np.concatenate(np_inputs, axis=3)

        logging.debug('Generating embedding.')
        condition = []
        for inp in cfg["condition"]:
            if inp == "Dose":
                # consistency with other contrast agents
                factor = (2 - int("gado" in self._s["T1_low"].attrs["contrastagent"]))
                dose = self._s["T1_low"].attrs["dose"] / (10 * factor)
                condition.append(dose)
            elif inp == "Percent":
                percent = self._s["T1_low"].attrs["percent"] / 100
                condition.append(percent)
            elif inp == "Field strength":
                field_strength = (self._s["T1_low"].attrs["field strength"] - 2.25) / 0.75
                condition.append(field_strength)
            elif inp == "contrastagent":
                contrastagent = self._s["T1_low"].attrs["contrastagent"]
                if contrastagent not in SeriesData.contrastagent_to_relaxivity.keys():
                    logging.warning(f"No relaxivity entry for '{contrastagent}'. Falling back to 'dotarem'.")
                    contrastagent = "dotarem"
                contrastagent = SeriesData.contrastagent_to_relaxivity[contrastagent]
                contrastagent = contrastagent[np.isclose(self._s["T1_low"].attrs["field strength"], 3)] / 10.
                condition.append(contrastagent)
            elif inp == "scaling factors":
                for factor in self._s["T1_zero"].attrs["scaling_factors"]:
                    condition.append(factor)
            elif inp == "Noise level":
                sigma = self._s["T1_low"].attrs["sigma"] * 10  # original sigma range is in [0, 0.2]
                condition.append(sigma)
            elif inp == "Noise level robust":
                sigma_h = self._s["T1_low"].attrs["sigma_h"] * 10  # original sigma_h range is in [0, 0.14]
                condition.append(sigma_h)
            elif inp == "Kurtosis":
                kurtosis = self._s["T1_low"].attrs["kurtosis"] / 100  # original kurtosis range is in [0, 140]
                condition.append(kurtosis)

        sample["condition"] = np.asarray(condition, dtype=np.float32)

        sample["mean"] = self._s[self.low + "_mean"].data[..., np.newaxis].copy()
        sample["std"] = self._s[self.low + "_std"].data[..., np.newaxis].copy()

        # get the target
        logging.debug('Generating target specific values.')
        if SeriesData.target in self._s.keys():
            np_target = self._s[SeriesData.target].data[..., np.newaxis].copy()
            if "radio_reg_target" in cfg and cfg["radio_reg_target"]:
                np_target *= self._s[SeriesData.target].attrs["scale"]
            np_target /= reference_scale
            np_target = normalize(np_target)

            # estimate signal probability based on local mean and standard deviation
            # using a discriminative model
            w = (np.log(1/cfg.psignal.p[0] - 1) - np.log(1/cfg.psignal.p[1] - 1)) / (cfg.psignal.x[1] - cfg.psignal.x[0])
            b = - w * cfg.psignal.x[0] - np.log(1/cfg.psignal.p[0] - 1)
            local_mean = self._s[SeriesData.target + "_mean"].data[..., np.newaxis].copy()
            local_std = self._s[SeriesData.target + "_std"].data[..., np.newaxis].copy()
            normalized_diff = ((np_target - np_reference) - local_mean) / np.maximum(local_std, 1e-6)
            sample["p_signal"] = sigmoid(normalized_diff * w + b).astype(np.float32)

            if "sub_target" in cfg and cfg["sub_target"]:
                # target is given by the subtraction image
                np_target = np_target - np_inputs[base_index]
                if "sub_threshold" in cfg:
                    np_target = np.maximum(np_target, cfg["sub_threshold"])

                if "sub_adaptive" in cfg and cfg["sub_adaptive"]:
                    # Mask out pixels which have a low probability of holding CA signal
                    np_target *= np.minimum(sample["p_signal"] * cfg["sub_adaptive"], 1)

            sample["target"] = np_target

            np_full = self._s[SeriesData.target].data[..., np.newaxis] \
                * self._s[SeriesData.target].attrs["scale"] / reference_scale
            np_zero = self._s[SeriesData.zero].data[..., np.newaxis] \
                / reference_scale
            diff_full = np.maximum(np_full - np_zero, 0)
            if "add_grad" in cfg and cfg["add_grad"]:
                spacing = min(list(self._s[SeriesData.zero].attrs['spacing']))
                # sigma=0.5mm
                sample['grad'] = np.maximum(laplace((gaussian(diff_full, 0.5 * 1/spacing))), 0.).astype(np.float32)

            if "add_smooth" in cfg and cfg["add_smooth"]:
                spacing = min(list(self._s[SeriesData.zero].attrs['spacing']))
                threshold = cfg["adaptive_threshold"] * self._s[SeriesData.target].attrs["sigma"]
                # sigma=0.5mm
                sample['smooth'] = apply_hysteresis_threshold(
                    gaussian(np.maximum(diff_full, 0), 1/spacing), threshold/2, threshold).astype(diff_full.dtype)

        return sample

    def get_standard_scale(self, cfg: dict, add_key="") -> np.ndarray:
        if "standard_scale" in cfg:
            return np.asarray(cfg["standard_scale"], dtype=np.float32)
        else:
            raise RuntimeError("Standard scale not in config!")

    def to_reference(self, x: np.ndarray, cfg: dict = None, return_contrast_window=False) -> ScanData:
        # only 3D image volumes are supported
        assert x.ndim == 3

        # so that a single outlier cant break the contrast window
        if return_contrast_window:
            # Contrast is too limited so I use 50 percent more than observed in the brain
            max_val = np.quantile(x[self._s[SeriesData.mask_atlas].data > 1e-4], 0.9999)
        # revert nyul normalizationmask
        if cfg and "nyul_normalization" in cfg and cfg["nyul_normalization"]:
            landmarks = self._s[SeriesData.zero].attrs["landmarks"]
            standard_scale = self.get_standard_scale(cfg)
            x = normalization.Nyul.reverse(x, landmarks=landmarks, standard_scale=standard_scale)
            if return_contrast_window:
                max_val = normalization.Nyul.reverse(max_val, landmarks=landmarks, standard_scale=standard_scale)
        # convert to sitk image
        simage = sitk.GetImageFromArray(x * self._s[SeriesData.zero].attrs["vmax"])
        if return_contrast_window:
            max_val *= self._s[SeriesData.zero].attrs["vmax"]
        if self._isotropic_regridding:
            logging.info("Regridding to original anisotropic resolution")
            # regrid to original voxel size
            simage.CopyInformation(self._ref_data_isotropic)
            simage = self._ref_data.resample_to(simage, fft=self._fft_regridding)
            simage = sitk.Clamp(simage, simage.GetPixelIDValue(),
                                lowerBound=0.)
        elif self._ref_data:
            simage.CopyInformation(self._ref_data.sitk())
        if return_contrast_window:
            return ScanData(other=self._ref_data).from_sitk(simage), max_val
        return ScanData(other=self._ref_data).from_sitk(simage)

    def to_anisotropic(self, x: np.ndarray) -> ScanData:
        simage = sitk.GetImageFromArray(x)
        if self._isotropic_regridding:
            logging.info("Regridding to original anisotropic resolution")
            # regrid to original voxel size
            simage.CopyInformation(self._ref_data_isotropic)
            simage = self._ref_data.resample_to(simage, fft=self._fft_regridding)
            # do not use negative values.
            simage = sitk.Clamp(simage, simage.GetPixelIDValue(),
                                lowerBound=0.)
        else:
            # Do not throw an exception if we do non-isotropic
            logging.info('Regridding was not turned on!')
        return ScanData(other=self._ref_data).from_sitk(simage)

    def add_to_reference(self, x: np.ndarray, cfg: dict = None, return_contrast_window=False) -> ScanData:
        if self._isotropic_regridding:
            simage = sitk.GetImageFromArray(x)
            logging.info("Regridding to original anisotropic resolution")
            # regrid to original voxel size
            simage.CopyInformation(self._ref_data_isotropic)
            simage = self._ref_data.resample_to(simage, fft=self._fft_regridding)
            x = sitk.GetArrayFromImage(simage)

        # get reference image
        r = self._ref_data.to_numpy() / self._s[SeriesData.zero].attrs["vmax"]
        if cfg and "nyul_normalization" in cfg and cfg["nyul_normalization"]:
            standard_scale = self.get_standard_scale(cfg)
            # apply nyul normalization to reference image
            r = normalization.Nyul.normalize(r, standard_scale,
                                             landmarks=self._s[SeriesData.zero].attrs["landmarks"])

        # add to reference
        x = np.maximum(r + x, 0)

        if cfg and "nyul_normalization" in cfg and cfg["nyul_normalization"]:
            # undo nyul normalization
            standard_scale = self.get_standard_scale(cfg)
            x = normalization.Nyul.reverse(x, landmarks=self._s[SeriesData.zero].attrs["landmarks"],
                                           standard_scale=standard_scale)

        # scale to original intensities
        x *= self._s[SeriesData.zero].attrs["vmax"]

        if return_contrast_window:
            # use brain mask to estimate the contrast window (robust towards outliers)
            mask = registration.resample(self._s[SeriesData.mask_atlas].sitk(), self._ref_data.sitk(),
                                         sitk.Transform(), interpolation="linear")
            max_val = np.quantile(x[sitk.GetArrayViewFromImage(mask) > 1e-4], 0.9999)
            return ScanData(other=self._ref_data).from_numpy(x), max_val
        else:
            return ScanData(other=self._ref_data).from_numpy(x)

    def find_dicom_series_from_path(path: str):
        path_dict = {}
        dcmdir = os.path.join(path, 'DICOMDIR')
        if not os.path.exists(dcmdir):
            logging.error('Could not find DICOMDIR: ' + dcmdir)
            raise Exception('Not found!')

        ref_spacing = -np.ones((3, ))

        ds = pydicom.dcmread(dcmdir)
        for patient in ds.patient_records:
            studies = [
                ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
            ]
            for study in studies:
                descr = study.StudyDescription if hasattr(study, 'StudyDescription') else None
                logging.info(
                    f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                    f"StudyDate={study.StudyDate}, StudyDescription={descr}"
                )

                # Find all the SERIES records in the study
                all_series = [
                    ii for ii in study.children if ii.DirectoryRecordType == "SERIES" and ii.SeriesNumber
                ]
                # We sort by series number as they are increasing and reconstructions have higher series numbers
                # as the initial series.
                for series in sorted(all_series, key=lambda x: x.SeriesNumber):
                    # Find all the IMAGE records in the series
                    images = [
                        ii for ii in series.children
                        if ii.DirectoryRecordType == "IMAGE"
                    ]
                    plural = ('', 's')[len(images) > 1]

                    descr = getattr(
                        series, "SeriesDescription", None
                    )
                    logging.info(
                        f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                        f"Modality={series.Modality}, SeriesDescription={descr} - "
                        f"{len(images)} SOP Instance{plural}"
                    )

                    # Get the absolute file path to each instance
                    #   Each IMAGE contains a relative file path to the root directory
                    elems = [ii["ReferencedFileID"] for ii in images]
                    # Make sure the relative file path is always a list of str
                    paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                    paths = [os.path.join(path, *p) for p in paths]

                    # compute the pixel spacing
                    try:
                        file = os.path.join(path, paths[-1])
                        tmp = pydicom.dcmread(file, specific_tags=["PixelSpacing", "SliceThickness"])
                        pixel_spacing = np.concatenate((
                            np.asarray(tmp.SliceThickness, dtype=np.float32)[None, ],
                            np.asarray(tmp.PixelSpacing, dtype=np.float32),
                            ),
                            axis=0
                        )

                        if descr is None:   # This is None for patients from the DKFZ.
                            descr = elems[0][1]

                        dirname = os.path.dirname(os.path.abspath(paths[-1]))
                    except (IndexError, AttributeError) as e:
                        logging.warning(f"Skipping '{file}' due to exception {e}")
                        continue    # This branch is for sequences we do not care about.

                    # mpr is multiplanar reconstruction.
                    # We want to use the original.
                    # We had this before, but because of Essen I had to use a longer version
                    # because they use mpr to refer to mprage, the protocol.
                    if 'sag_km_mpr' in descr.lower() or 'sag_mpr_km' in descr.lower() or 'mpr cs t1_3d_tfe_tra' in descr.lower() or 'mpr cs t1_3d_tfe_cor' in descr.lower():
                        continue

                    # Excluding flair, only occured for 1 patient.
                    if (('t1' in descr.lower() and 'fl2d' not in descr.lower()) or "3DTFE" in descr) and 't1w_ffe' not in descr.lower():
                        if '10p' in descr or '20p' in descr or '30p' in descr or '33p' in descr or 'low' in descr.lower():
                            if np.min(ref_spacing) < 0:
                                # no reference defined yet
                                # the nativ has to have a lower series number and has to fit the low-dose.
                                continue
                            diff_spacing = np.linalg.norm(pixel_spacing - ref_spacing, ord=2)
                            if SeriesData.low not in path_dict:
                                if diff_spacing < 1e-1:
                                    path_dict[SeriesData.low] = os.path.join(path, dirname)
                                else:
                                    logging.warning(f"Ignoring low-dose sequences; {os.path.join(path, dirname)} with distance {diff_spacing}")
                            else:
                                logging.warning(f"Found multiple low-dose sequences; {os.path.join(path, dirname)} with distance {diff_spacing}")

                        elif 'KM' in descr or 'full' in descr.lower() or "mpr_s_sag" in descr.lower():  # mpr_s_sag is used for Essen.
                            if np.min(ref_spacing) < 0:
                                # no reference defined yet
                                # the nativ has to have a lower series number and has to fit the low-dose.
                                continue
                            diff_spacing = np.linalg.norm(pixel_spacing - ref_spacing, ord=2)
                            if SeriesData.target not in path_dict:
                                if diff_spacing < 1e-1:
                                    path_dict[SeriesData.target] = os.path.join(path, dirname)
                                else:
                                    logging.warning(f"Ignoring full-dose sequences; {os.path.join(path, dirname)} with distance {diff_spacing}")
                            else:
                                logging.warning(f"Found multiple full-dose sequences; {os.path.join(path, dirname)} with distance {diff_spacing}")

                        # If the resolution is too bad we either have the wrong sequence or an error.
                        elif len(images) > 50:
                            if SeriesData.zero not in path_dict:
                                path_dict[SeriesData.zero] = os.path.join(path, dirname)
                                ref_spacing = pixel_spacing
                            else:
                                logging.warning(f"Found multiple zero-dose sequences: {os.path.join(path, dirname)}")

                    elif 'flair' in descr.lower() or ('t2' in descr.lower() and ('fs' in descr.lower() or 'dark-fluid' in descr.lower())):
                        if 'Flair' not in path_dict:
                            path_dict[SeriesData.flair] = os.path.join(path, dirname)
                        else:
                            logging.warning(f"Found multiple sequences: {os.path.join(path, dirname)}")

                    elif 't2' in descr.lower():
                        path_dict[SeriesData.t2] = os.path.join(path, dirname)
                    elif 'diff' in descr.lower():
                        path_dict[SeriesData.diff] = os.path.join(path, dirname)

        logging.info('Found the following path dict:' + str(path_dict))
        return path_dict
