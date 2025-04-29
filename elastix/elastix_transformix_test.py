import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_elastix import get_itk_translation_transform, get_default_parameter_object, get_elastix_registration_object
from utils.utils_plot import viz_slices
from utils.utils_itk import resample_image_itk, create_itk_view, scale_spacing_and_origin
from utils.utils_tiff import load_tiff, center_crop
from utils.utils_elastix import get_itk_translation_transform, get_default_parameter_object, get_elastix_registration_object

# Define paths
base_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = base_path + "Larch_A_bin1x1/"
moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

# Load downsampled images
factor = 1.0
moving_array_sparse = load_tiff(sample_path + "Larch_A_LFOV_crop.tiff")
moving_array_sparse = moving_array_sparse
fixed_array_sparse = load_tiff(sample_path + "Larch_A_4x_crop_pos1.tiff")

# Convert to ITK images and set spacing and origin
moving_image_sparse = create_itk_view(moving_array_sparse)
scale_spacing_and_origin(moving_image_sparse, factor)

fixed_image_sparse = create_itk_view(fixed_array_sparse)
scale_spacing_and_origin(fixed_image_sparse, factor)

# Create transform
# pos X moves image left, pos Y moves image up in the slice plane, while pos Z moves slices
translation_vec = [0.0, 0.0, 1000.0]  # 3D Translation vector
transform = get_itk_translation_transform(translation_vec, save_path=None)

parameter_map = {
                 "Direction": ("1", "0", "0", "0", "1", "0", "0", "0", "1"),
                 "Index": ("0", "0", "0"),
                 "Origin": ("0", "0", "0"),
                 "Size": ("493", "504", "1320"),
                 "Spacing": ("1", "1", "1")
                }

parameter_object = itk.ParameterObject.New()
parameter_object.AddParameterMap(parameter_map)

# Load Transformix Object
transformix_object = itk.TransformixFilter.New(moving_image_sparse)
transformix_object.SetTransformParameterObject(parameter_object)
transformix_object.SetTransform(transform)
transformix_object.Update()

# Update object (required)
transformix_object.UpdateLargestPossibleRegion()

# Results of Transformation
result_image_transformix = transformix_object.GetOutput()

viz_slices(moving_image_sparse, [150], axis=1)
viz_slices(fixed_image_sparse, [150], axis=1)
viz_slices(result_image_transformix, [150], axis=1)

print("Done")