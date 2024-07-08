import logging
import multiprocessing as mp
import numpy as np
import os
from timeit import default_timer as timer

import itk
import SimpleITK as sitk

# type hinting aliases
IImage = itk.Image[itk.F, 3]
SImage = sitk.Image


__all__ = ["spatial", "resample"]


def get_interpolator(interpolation: str):
    """ Get simple ITK interpolator by string.
    """
    if interpolation == "linear":
        return sitk.sitkLinear
    elif interpolation == "spline":
        return sitk.sitkBSpline
    elif interpolation == "Lanczos":
        return sitk.sitkLanczosWindowedSinc
    else:
        raise RuntimeError(f"Interpolation type '{interpolation}' not supported!")


def resample(moving: SImage, fixed: SImage, transform: sitk.Transform,
             interpolation: str = "spline"):
    """ Resmaple a volume given the transform into the coordinate system
        of fixed image.

        Args:
            moving: image to transform
            fixed: static image defining the coordinate system
            transform: transform to apply
            interpolation: string specifying the interpolator
        Returns:
            transformed moving image
    """
    interpolator = get_interpolator(interpolation)
    out = sitk.Resample(moving, fixed, transform, interpolator, 0,
                        moving.GetPixelID())
    if interpolation == 'spline':
        tmp = sitk.GetArrayViewFromImage(out)
        return sitk.Clamp(out, out.GetPixelIDValue(),
                          lowerBound=float(tmp.min()),
                          upperBound=float(tmp.max()))
    else:
        return out


def to_itk(simg: SImage) -> IImage:
    img = itk.GetImageFromArray(sitk.GetArrayViewFromImage(simg), is_vector=simg.GetNumberOfComponentsPerPixel() > 1)
    img.SetOrigin(simg.GetOrigin())
    img.SetSpacing(simg.GetSpacing())
    img.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(simg.GetDirection()), (3, 3))))
    return img


def from_itk(img: IImage) -> SImage:
    simg = sitk.GetImageFromArray(itk.GetArrayViewFromImage(img), is_vector=img.GetNumberOfComponentsPerPixel() > 1)
    simg.SetOrigin(img.GetOrigin())
    simg.SetSpacing(img.GetSpacing())
    simg.SetDirection(itk.GetArrayFromMatrix(img.GetDirection()).flatten())
    return simg


def elastix_to_rigid(pm):
    transform = sitk.Euler3DTransform()
    center = np.asarray(pm['CenterOfRotationPoint'], dtype=np.float64)
    center = np.append(center, 1.0)
    transform.SetFixedParameters(center)
    params = np.asarray(pm['TransformParameters'], dtype=np.float64)
    transform.SetParameters(params)
    return transform


def elastix_to_affine(pm):
    transform = sitk.AffineTransform(3)
    center = np.asarray(pm['CenterOfRotationPoint'], dtype=np.float64)
    transform.SetFixedParameters(center)
    params = np.asarray(pm['TransformParameters'], dtype=np.float64)
    transform.SetParameters(params)
    return transform


def spatial(fixed: SImage, moving: SImage,
            fixed_mask: SImage = None,
            transform: str = "rigid",
            scale_fixed: float = None,
            scale_moving: float = None,
            use_gradient_abs: bool = False) -> tuple:
    """ Performs the actual 3D registration.

        Args:
            fixed: static volume.
            moving: volume to be transformed.
            fixed_mask: optional binary image defining the ROI
            transfor: type of transform either 'rigid' or 'affine'
            scale_fixed: optional scale of the fixed image (defaults to 0.95 quantile)
            scale_moving: optional scale of the moving image (defaults to 0.95 quantile)
            use_gradient_abs: Use the absolute of the gradient.
        Returns:
            registered volume
            final itk transform
            mse (optinal) if get_error is True.
    """
    assert transform in {"affine", "rigid"}

    if scale_fixed is None:
        np_fixed = sitk.GetArrayViewFromImage(fixed)
        scale_fixed = np.quantile(np_fixed[np_fixed > 1e-4], 0.95)
    if scale_moving is None:
        np_moving = sitk.GetArrayViewFromImage(moving)
        scale_moving = np.quantile(np_moving[np_moving > 1e-4], 0.95)

    if use_gradient_abs:
        log_filter = sitk.LaplacianRecursiveGaussianImageFilter()
        log_filter.SetSigma(1.0)

        fixed = sitk.Abs(log_filter.Execute(fixed))
        moving = sitk.Abs(log_filter.Execute(moving))

    fixed = sitk.Multiply(fixed, 1/scale_fixed)
    moving = sitk.Multiply(moving, 1/scale_moving)

    parameter_object = itk.ParameterObject.New()
    pm_file_path = os.path.dirname(os.path.abspath(__file__))
    parameter_object.AddParameterFile(os.path.join(pm_file_path, "rigid.txt"))
    if transform == "affine":
        # perform an affine registration after the rigid pre-registration
        parameter_object.AddParameterFile(os.path.join(pm_file_path, f"{transform}.txt"))
    # parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap(transform))

    # setup elastix registration
    elastix_object = itk.ElastixRegistrationMethod[IImage, IImage].New()
    elastix_object.SetFixedImage(to_itk(fixed))
    elastix_object.SetMovingImage(to_itk(moving))
    if fixed_mask:
        elastix_object.SetFixedMask(to_itk(fixed_mask))
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetNumberOfThreads(min(32, mp.cpu_count()))  # Use all CPUs (at most 32)
    # elastix_object.SetLogToFile(True)
    # elastix_object.SetLogFileName("elastix.log")
    # elastix_object.SetOutputDirectory("./out")
    # elastix_object.SetLogToConsole(False)

    # start registration
    logging.info("Registration: start")
    start = timer()
    elastix_object.Update()
    end = timer()
    logging.info(f"Registration: ended ({end-start}s)")

    # parse transform
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    t_rigid = elastix_to_rigid(result_transform_parameters.GetParameterMap(0))
    if transform == "affine":
        t_affine = elastix_to_affine(result_transform_parameters.GetParameterMap(1))
        t_final = sitk.CompositeTransform([t_affine, t_rigid])
    else:
        t_final = t_rigid

    return t_final
