from tqdm import tqdm
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_elastix import elastix_coarse_registration_sweep, get_elastix_registration_object, get_default_parameter_object_list
from utils.utils_plot import viz_slices, viz_multiple_images, viz_registration
from utils.utils_itk import resample_image_itk, create_itk_view, scale_spacing_and_origin
from utils.utils_tiff import load_tiff, center_crop


# Define paths
base_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = base_path + "Larch_A_bin1x1/"
moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

# Load downsampled images
factor = 1.0
moving_array_sparse = load_tiff(sample_path + "Larch_A_LFOV_crop.tiff")
fixed_array_sparse = load_tiff(sample_path + "Larch_A_4x_crop_pos1.tiff")
moving_array_sparse = moving_array_sparse[:, 50:-50, 50:-50]
#fixed_array_sparse = fixed_array_sparse


if __name__ == "__main__":

    # Convert to ITK images and set spacing and origin
    moving_image_sparse = create_itk_view(moving_array_sparse)
    scale_spacing_and_origin(moving_image_sparse, factor)

    fixed_image_sparse = create_itk_view(fixed_array_sparse)
    scale_spacing_and_origin(fixed_image_sparse, factor)

    best_result, result_transform_object, metric = elastix_coarse_registration_sweep(fixed_image_sparse,
                                                                                     moving_image_sparse,
                                                                                     center=None,
                                                                                     spacing=(50, 20, 20),
                                                                                     size=(2, 2, 2),
                                                                                     log_mode=None)

    # Visualize the results
    viz_multiple_images([fixed_image_sparse, best_result], [319, 310, 305], axis=2)
    viz_registration(fixed_image_sparse, best_result, [319, 310, 305], axis=2)


    ################## Refine registration ###################

    # Create default parameter object list
    registration_models = ['affine', 'bspline']
    resolution_list = [4, 4]
    max_iteration_list = [256, 256]
    parameter_object, parameter_map = get_default_parameter_object_list(registration_models, resolution_list, max_iteration_list)

    elastix_object = get_elastix_registration_object(fixed_image_sparse, moving_image_sparse, parameter_object, log_mode=None)

    # Set transform parameter object from coarse registration as the initial tranformation for the refinement
    elastix_object.SetInitialTransformParameterObject(result_transform_object)

    # Run the registration
    elastix_object.UpdateLargestPossibleRegion()  # Update filter object (required)
    result_image_small_refined = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    result_array = itk.array_view_from_image(result_image_small_refined)
    fixed_array = itk.array_view_from_image(fixed_image_sparse)

    # Visualize the results
    viz_multiple_images([fixed_image_sparse, result_image_small_refined], [319, 310, 305], axis=2)
    viz_registration(fixed_image_sparse, result_image_small_refined, [319, 310, 305], axis=2)

    print("Done")
