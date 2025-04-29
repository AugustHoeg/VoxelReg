import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_elastix import get_itk_translation_transform, get_default_parameter_object, get_elastix_registration_object
from utils.utils_plot import viz_slices
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
moving_array_sparse = moving_array_sparse[925:, :, :]
#fixed_array_sparse = fixed_array_sparse


if __name__ == "__main__":

    # Convert to ITK images and set spacing and origin
    moving_image_sparse = create_itk_view(moving_array_sparse)
    scale_spacing_and_origin(moving_image_sparse, factor)

    fixed_image_sparse = create_itk_view(fixed_array_sparse)
    scale_spacing_and_origin(fixed_image_sparse, factor)

    parameter_object, parameter_map = get_default_parameter_object('translation', 4)

    if False:
        # Call registration function
        result_image_small, result_transform_parameters = itk.elastix_registration_method(
            fixed_image_sparse, moving_image_sparse,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name='transforms/link.txt',
            log_to_console=False)
    else:
        # Get elastix registration object
        elastix_object = get_elastix_registration_object(fixed_image_sparse, moving_image_sparse, parameter_object, log_to_console=True)

        shape_diff = np.subtract(moving_image_sparse.shape, fixed_image_sparse.shape).astype(np.float64)

        # Set initial transform
        translation_vec = [shape_diff[2]/2, shape_diff[1]/2, shape_diff[0]/2]  # 3D Translation vector
        transform = get_itk_translation_transform(translation_vec, save_path=None)
        elastix_object.SetInitialTransform(transform)

        # Run the registration
        elastix_object.UpdateLargestPossibleRegion()     # Update filter object (required)
        result_image_small = elastix_object.GetOutput()
        result_transform_parameters = elastix_object.GetTransformParameterObject()

    # visualize the results
    viz_slices(fixed_image_sparse, [319, 310, 305], axis=2)
    viz_slices(result_image_small, [319, 310, 305], axis=2)

    viz_slices(fixed_image_sparse, [150], axis=1)
    viz_slices(result_image_small, [150], axis=1)

    print("Done")


