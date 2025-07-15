import itk
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from utils.utils_plot import viz_multiple_images, viz_registration


# Class reference for ElastixRegistrationMethod:
# https://elastix.dev/doxygen/classitk_1_1ElastixRegistrationMethod.html

def get_itk_translation_transform(translation_vec=[0.0, 0.0, 0.0], save_path=None):

    transform = itk.TranslationTransform[itk.D, 3].New()
    transform.SetOffset(translation_vec)
    if save_path is not None:
        itk.transformwrite(transform, save_path)

    return transform


def get_itk_rigid_transform(rotation_angles=[0.0, 0.0, 0.0], translation_vec=[0.0, 0.0, 0.0], save_path=None):
    transform = itk.Euler3DTransform[itk.D].New()
    parameters = rotation_angles + translation_vec  # [rx, ry, rz, tx, ty, tz]
    transform.SetParameters(parameters)
    if save_path is not None:
        itk.transformwrite(transform, save_path)

    return transform


def get_default_parameter_map(parameter_object, registration_model='translation', resolutions=4, max_iterations=1024, metric='AdvancedMattesMutualInformation', no_samples=4096,
                      write_result_image=True, log_mode=False):

    parameter_map = parameter_object.GetDefaultParameterMap(registration_model, resolutions)
    parameter_map['AutomaticTransformInitialization'] = ['true']
    parameter_map['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
    parameter_map['HowToCombineTransforms'] = ['Compose']

    # Write result image
    if write_result_image:
        parameter_map['WriteResultImage'] = ['true']
    else:
        parameter_map['WriteResultImage'] = ['false']

    # Logging
    if log_mode:
        parameter_map["WriteIterationInfo"] = ["true"]
        parameter_map["WriteTransformParametersEachIteration"] = ["true"]
        parameter_map["LogToFile"] = ["true"]

    # Loss metric
    parameter_map['Metric'] = [str(metric)]

    # Number of samples
    parameter_map['NumberOfSpatialSamples'] = [str(no_samples)]
    parameter_map['NewSamplesEveryIteration'] = ['true']  # Default

    # Max iterations
    parameter_map['MaximumNumberOfIterations'] = (str(max_iterations),)

    return parameter_map


def get_default_parameter_object(registration_model='translation', resolutions=4, max_iterations=1024, metric='AdvancedMattesMutualInformation', no_registration_samples=4096, write_result_image=True, save_path=None, log_mode=False):

    """

    :param registration_model: Can be translation, rigid, affine, bspline, spline, groupwise
    :param resolutions:
    :param max_iterations:
    :return:
    """

    # The Elastix manual is here: https://courses.compute.dtu.dk/02502/docs/elastix-5.2.0-manual.pdf
    parameter_object = itk.ParameterObject.New()

    # Get default parameter map for the model and resolution
    parameter_map = get_default_parameter_map(parameter_object, registration_model, resolutions, max_iterations, metric, no_registration_samples, write_result_image, log_mode)

    # Add the parameter map
    parameter_object.AddParameterMap(parameter_map)

    if save_path is not None:
        # Save custom parameter map
        parameter_object.WriteParameterFile(parameter_map, save_path)

    return parameter_object, parameter_map


def get_default_parameter_object_list(registration_models, resolution_list, max_iteration_list, metric_list, no_registration_samples_list, write_result_image_list, save_path=None):

    """

    :param registration_model: Can be translation, rigid, affine, bspline, spline, groupwise
    :param resolutions:
    :param max_iterations:
    :return:
    """
    if no_registration_samples_list is None:
        no_registration_samples_list = [4096] * len(registration_models)

    # The Elastix manual is here: https://courses.compute.dtu.dk/02502/docs/elastix-5.2.0-manual.pdf
    parameter_object = itk.ParameterObject.New()

    for i in range(len(registration_models)):
        registration_model = registration_models[i]
        resolutions = resolution_list[i]
        max_iterations = max_iteration_list[i]
        metric = metric_list[i]
        no_registration_samples = no_registration_samples_list[i]
        write_result_image = write_result_image_list[i]

        parameter_map = get_default_parameter_map(parameter_object, registration_model, resolutions, max_iterations, metric, no_registration_samples, write_result_image, log_mode=False)

        # # Get default parameter map for the current model and resolution
        # parameter_map = parameter_object.GetDefaultParameterMap(registration_model, resolutions)
        # parameter_map['MaximumNumberOfIterations'] = str(max_iterations)

        # Add the parameter map to the object
        parameter_object.AddParameterMap(parameter_map)

    if save_path is not None:
        # Save custom parameter map
        parameter_object.WriteParameterFile(parameter_map, save_path)

    return parameter_object, parameter_map


def get_elastix_registration_object(fixed_image, moving_image, parameter_object, log_mode="console"):

    # Define the elastix registration object and set the parameter object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)

    # Set additional options
    if log_mode == "file":
        elastix_object.SetLogToFile(True)
    elif log_mode == "console":
        elastix_object.SetLogToConsole(True)
    else:
        elastix_object.SetLogToConsole(False)

    return elastix_object


def get_spaced_coords(shape, spacing):
    """
    Generates evenly spaced 3D coordinates within a 3D image volume.

    Parameters:
        shape (tuple of int): The shape of the 3D image as (depth, height, width).
        spacing (int or tuple of int): The spacing between points. Can be a single int or a tuple (dz, dy, dx).

    Returns:
        numpy.ndarray: Array of shape (N, 3) with (z, y, x) coordinates.
    """
    if isinstance(spacing, int):
        dz = dy = dx = spacing
    else:
        dz, dy, dx = spacing

    z = np.arange(0, shape[0], dz)
    y = np.arange(0, shape[1], dy)
    x = np.arange(0, shape[2], dx)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    coords = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)

    return coords

def get_positional_grid(center_mm, grid_spacing_mm, grid_size):

    offsets = []
    for i in range(3):
        if grid_size[i] % 2 == 0:  # even
            start = center_mm[i] + ((grid_size[i] - 1) / 2) * grid_spacing_mm[i]
            stop = center_mm[i] - ((grid_size[i] - 1) / 2) * grid_spacing_mm[i]
            offsets.append(np.linspace(start, stop, grid_size[i]))
        else:  # odd
            start = center_mm[i] + (grid_size[i] // 2) * grid_spacing_mm[i]
            stop = center_mm[i] - (grid_size[i]// 2) * grid_spacing_mm[i]
            offsets.append(np.linspace(start, stop, grid_size[i]))
    grid = np.meshgrid(offsets[0], offsets[1], offsets[2], indexing='ij')
    grid = np.stack(grid, axis=-1).reshape(-1, 3)

    return grid


def get_spaced_coords_around_point(center_mm, grid_spacing_mm, grid_size, spacing_mm):
    """
    Generates a grid of evenly spaced 3D coordinates around a center point.

    Parameters:
        center (tuple of float): The (z, y, x) center point.
        shape (tuple of int): The shape of the 3D image volume (depth, height, width).
        spacing (int or tuple): The spacing between coordinates. Can be int or (dz, dy, dx).
        size (int or tuple): Number of samples in each direction. Can be int (applied to all dims) or (nz, ny, nx).

    Returns:
        numpy.ndarray: Array of shape (N, 3) with valid (z, y, x) coordinates within image bounds.
    """
    center = np.round(np.array(center_mm) / spacing_mm).astype(int)
    grid_spacing = np.round(np.array(grid_spacing_mm) / spacing_mm).astype(int)

    # Unpack parameters
    dz, dy, dx = (grid_spacing, grid_spacing, grid_spacing) if isinstance(grid_spacing, int) else grid_spacing
    nz, ny, nx = (grid_size, grid_size, grid_size) if isinstance(grid_size, int) else grid_size

    # Symmetric offsets around 0
    offsets_z = dz * (np.arange(nz) - (nz - 1) / 2)
    offsets_y = dy * (np.arange(ny) - (ny - 1) / 2)
    offsets_x = dx * (np.arange(nx) - (nx - 1) / 2)

    # Create grid of offsets
    zz, yy, xx = np.meshgrid(offsets_z, offsets_y, offsets_x, indexing='ij')
    offsets = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)

    # Add offsets to center
    center = np.array(center).reshape(1, 3)
    coords = center + offsets

    # Convert to back to mm
    coords = coords * spacing_mm

    # Clip to valid bounds
    #shape = np.array(shape).reshape(1, 3)
    #coords = coords[(coords >= 0).all(axis=1) & (coords < shape).all(axis=1)]

    return coords



def elastix_coarse_registration_sweep(fixed_image_sparse, moving_image_sparse, center_mm, initial_rotation_angles=(0.0, 0.0, 0.0), grid_spacing_mm=(1, 1, 1), grid_size=(2, 2, 2),
                                      resolutions=4, max_iterations=1024, metric='AdvancedMattesMutualInformation', no_registration_samples=4096, write_result_image=True,
                                      log_mode=None, visualize=False, fig_name=None):

    print("Running coarse registration")

    parameter_object, parameter_map = get_default_parameter_object('translation', resolutions, max_iterations, metric, no_registration_samples, write_result_image)
    elastix_object = get_elastix_registration_object(fixed_image_sparse, moving_image_sparse, parameter_object, log_mode=log_mode)

    if center_mm is None:  # defaults to aligning upper slices and center in x and y
        center_mm = (0.0, 0.0, 0.0)
    else:
        center_mm = tuple(0.0 if center_val is None else center_val for center_val in center_mm)  # ensure all are not None

    translation_coords = get_positional_grid(center_mm, grid_spacing_mm, grid_size)
    print("translation_coords", translation_coords)

    best_metric = -1

    #mse = np.mean((fixed_array - result_array) ** 2)

    progress_bar = tqdm(translation_coords, desc="Running coarse registration")

    for translation in progress_bar:
        progress_bar.set_postfix({'Translation': translation, 'Best Metric': best_metric})

        # Get elastix registration object
        #transform = get_itk_translation_transform(translation, save_path=None)
        transform = get_itk_rigid_transform(rotation_angles=initial_rotation_angles, translation_vec=translation, save_path=None)
        elastix_object.SetInitialTransform(transform)

        # Run the registration
        elastix_object.UpdateLargestPossibleRegion()  # Update filter object (required)
        result_image = elastix_object.GetOutput()
        result_transform_object = elastix_object.GetTransformParameterObject()

        result_array = itk.array_view_from_image(result_image)
        fixed_array = itk.array_view_from_image(fixed_image_sparse)

        metric = np.corrcoef(fixed_array.reshape(-1), result_array.reshape(-1))[0, 1]

        if metric > best_metric:
            best_metric = metric
            best_result = itk.ImageDuplicator(result_image)  # ensure deep copy
            best_transform_object = result_transform_object

        if visualize:
            # Visualize the results
            axis = 2
            dim = min(fixed_image_sparse.shape[axis], best_result.shape[axis])
            off = int(dim * 0.05)  # offset for visualization
            diff = fixed_image_sparse[:] - best_result[:]
            viz_multiple_images([fixed_image_sparse, best_result, diff], [dim-i*off-5 for i in range(3)], savefig=True, title=fig_name + "_coarse", axis=axis)
            #viz_registration(fixed_image_sparse, best_result, [dim-i*off-1 for i in range(3)])

    return best_result, best_transform_object, best_metric



def elastix_refined_registration(fixed_image_sparse, moving_image_sparse, coarse_transform_object, registration_models, resolution_list,
                                 max_iteration_list, write_result_image_list, metric_list='AdvancedMattesMutualInformation',
                                 no_registration_samples_list=None, log_mode=None, visualize=False, fig_name=None):

    print("Running refined registration with models: ", registration_models)

    parameter_object, parameter_map = get_default_parameter_object_list(registration_models, resolution_list, max_iteration_list, metric_list, no_registration_samples_list, write_result_image_list)

    elastix_object = get_elastix_registration_object(fixed_image_sparse, moving_image_sparse, parameter_object, log_mode=log_mode)

    # Set transform parameter object from coarse registration as the initial tranformation for the refinement
    elastix_object.SetInitialTransformParameterObject(coarse_transform_object)

    # Run the registration
    elastix_object.UpdateLargestPossibleRegion()  # Update filter object (required)
    result_image_small_refined = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    if visualize:
        # Visualize the results
        axis = 2
        dim = min(fixed_image_sparse.shape[axis], result_image_small_refined.shape[axis])
        off = int(dim * 0.05)  # offset for visualization
        diff = fixed_image_sparse[:] - result_image_small_refined[:]
        viz_multiple_images([fixed_image_sparse, result_image_small_refined, diff], [dim-i*off-5 for i in range(3)], axis=axis, savefig=True, title=fig_name + "_refined")
        viz_registration(fixed_image_sparse, result_image_small_refined, [dim-i*off-5 for i in range(3)], axis=axis, savefig=True, title=fig_name + "_eval")

    return result_image_small_refined, result_transform_parameters


