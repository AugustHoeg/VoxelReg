from tqdm import tqdm
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_elastix import get_itk_translation_transform, get_default_parameter_object, get_elastix_registration_object, get_spaced_coords, get_spaced_coords_around_point
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
#moving_array_sparse = moving_array_sparse[:, 50:-50, 50:-50]
#fixed_array_sparse = fixed_array_sparse


if __name__ == "__main__":

    # Convert to ITK images and set spacing and origin
    moving_image_sparse = create_itk_view(moving_array_sparse)
    scale_spacing_and_origin(moving_image_sparse, factor)

    fixed_image_sparse = create_itk_view(fixed_array_sparse)
    scale_spacing_and_origin(fixed_image_sparse, factor)

    parameter_object, parameter_map = get_default_parameter_object('translation', 4)
    elastix_object = get_elastix_registration_object(fixed_image_sparse, moving_image_sparse, parameter_object, log_mode=None)

    shape_diff = np.subtract(moving_image_sparse.shape, fixed_image_sparse.shape).astype(np.float64)
    center = [shape_diff[0], shape_diff[1]/2, shape_diff[2]/2]
    translation_coords = get_spaced_coords_around_point(center, shape=shape_diff + 1, spacing=(50, 20, 20), size=(2, 2, 2))

    best_result = None
    best_transform_parameters = None
    best_metric = -1

    #mse = np.mean((fixed_array - result_array) ** 2)

    progress_bar = tqdm(translation_coords, desc="Running coarse registration")

    for translation in progress_bar:
        progress_bar.set_postfix({'Translation': translation[::-1], 'Best Metric': best_metric})

    #for i, translation in enumerate(tqdm(translation_coords, desc=f"Running coarse registration")):
    #    print("Translation coordinates: ", translation[::-1])

        # Get elastix registration object
        transform = get_itk_translation_transform(translation[::-1], save_path=None)
        elastix_object.SetInitialTransform(transform)

        # Run the registration
        elastix_object.UpdateLargestPossibleRegion()  # Update filter object (required)
        result_image = elastix_object.GetOutput()
        result_transform_parameters = elastix_object.GetTransformParameterObject()

        result_array = itk.array_view_from_image(result_image)
        fixed_array = itk.array_view_from_image(fixed_image_sparse)

        metric = np.corrcoef(fixed_array.reshape(-1), result_array.reshape(-1))[0, 1]

        if metric > best_metric:
            print(f"New best metric: {metric}")
            best_metric = metric
            best_result = itk.ImageDuplicator(result_image)
            best_transform_parameters = result_transform_parameters

            # visualize the results
            viz_slices(fixed_image_sparse, [319, 310, 305], axis=2)
            viz_slices(best_result, [319, 310, 305], axis=2)

            viz_slices(fixed_image_sparse, [150], axis=1)
            viz_slices(best_result, [150], axis=1)

    print("Done")


