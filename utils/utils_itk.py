import numpy as np
import itk
import SimpleITK as sitk

def compute_itk_origin_size_crop(image, crop_start, crop_end):

    start = np.array(crop_start)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    offset = direction @ (spacing * start)
    new_origin = origin + offset
    new_size = np.array(crop_end) - np.array(crop_start)
    return new_origin, new_size


def crop_itk_metadata(origin, spacing, direction, crop_start, crop_end):

    offset = direction @ (spacing * crop_start)
    new_origin = origin + offset
    new_size = crop_end - crop_start
    return new_origin, new_size


def scale_spacing_and_origin(itk_image, scale_factor):

    # TODO needs to be tested

    old_spacing = itk_image.GetSpacing()
    old_origin = itk_image.GetOrigin()

    # The start of image's domain, origin-spacing/2.0, should be the same
    new_spacing = old_spacing * scale_factor
    new_origin = (new_spacing - old_spacing) * 0.5

    itk_image.SetSpacing(new_spacing)
    itk_image.SetOrigin(new_origin)

def get_itk_metadata(image):
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    size = itk.size(image)

    return origin, spacing, direction, size

def set_itk_metadata(image, origin, spacing, direction):
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image

def crop_itk_image(image, crop_start, crop_end):
    crop_filter = itk.CropImageFilter.New(Input=image)
    crop_filter.SetLowerBoundaryCropSize(crop_start)  # pixels cropped from lower side
    crop_filter.SetUpperBoundaryCropSize(crop_end)  # pixels cropped from upper side
    crop_filter.Update()
    cropped_image = crop_filter.GetOutput()
    return cropped_image


def create_itk_view(array):
    # Convert to an ITK image (automatically deduces pixel type)
    itk_image = itk.image_view_from_array(array)

    # Optionally, set spacing and origin
    #print("ITK view shape", itk_image.GetImageDimension())
    itk_image.SetOrigin([0.0] * itk_image.GetImageDimension())
    itk_image.SetSpacing([1.0] * itk_image.GetImageDimension())

    return itk_image


def resample_image_sitk(image, scale_factor, interpolator=sitk.sitkLinear):
    """
    Resample (downsample or upsample) a SimpleITK image by a given scale factor.

    Parameters:
        image (sitk.Image): The input image to resample.
        scale_factor (float or list/tuple): Scaling factor(s) for each dimension.
                                            E.g., 0.5 to downsample by half.
        interpolator (sitk.InterpolatorEnum): Interpolation method (default: sitkLinear).

    Returns:
        sitk.Image: The resampled image.
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * image.GetDimension()

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [s * f for s, f in zip(original_spacing, scale_factor)]
    new_size = [int(sz / f) for sz, f in zip(original_size, scale_factor)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def resample_image_itk(image, scale_factor, interpolator=None):
    """
    Resample (downsample or upsample) an ITK image by a given scale factor.

    Parameters:
        image (itk.Image): The input ITK image.
        scale_factor (float or list/tuple): Scaling factor(s) for each dimension.
        interpolator: Optional ITK interpolator. Defaults to LinearInterpolateImageFunction.

    Returns:
        itk.Image: The resampled image.
    """
    dim = image.GetImageDimension()

    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * dim

    original_spacing = image.GetSpacing()
    original_size = itk.size(image)

    # Compute new spacing and size
    new_spacing = [original_spacing[i] * scale_factor[i] for i in range(dim)]
    new_size = [int(original_size[i] / scale_factor[i]) for i in range(dim)]

    # Default interpolator if none provided
    if interpolator is None:
        interpolator = itk.LinearInterpolateImageFunction.New(image)

    # Identity transform
    transform = itk.IdentityTransform[itk.D, dim].New()

    # Set up resampler
    resampler = itk.ResampleImageFilter.New(Input=image)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)

    resampler.Update()
    return resampler.GetOutput()


def itk_checkerboard(fixed_image, moving_image, checker_pattern=(10, 10, 10)):

    checkerboard_filter = itk.CheckerBoardImageFilter[fixed_image].New()
    checkerboard_filter.SetInput1(fixed_image)
    checkerboard_filter.SetInput2(moving_image)
    checkerboard_filter.SetCheckerPattern(checker_pattern)
    checkerboard_filter.Update()

    return checkerboard_filter.GetOutput()

def voxel2world(image, point):

    x, y, z = point

    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    world_point = origin + direction @ (spacing * [x, y, z])

    return world_point


def extract_points(filename, field_name):
    result = []
    with open(filename, 'r') as file:
        for line in file:
            key = f"{field_name} = ["
            start = line.find(key)
            if start != -1:
                start += len(key)
                end = line.find("]", start)
                numbers_str = line[start:end].strip()
                numbers = [float(n) for n in numbers_str.split()]
                result.append(numbers)
    return result

def apply_registration_transform(moving_image, transform_parameter_object):

    transformix_object = itk.TransformixFilter.New(moving_image)  # Load Transformix Object
    transformix_object.SetTransformParameterObject(transform_parameter_object)  # set the transform parameter object
    transformix_object.UpdateLargestPossibleRegion()  # Update object (required)
    result_image = transformix_object.GetOutput()  # Results of Transformation

    return result_image

