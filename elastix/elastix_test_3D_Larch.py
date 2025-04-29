import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

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

if False:  # does not work with elastix
    # Suppose you want to translate the moving image
    translation = [900.0, 0, 0]  # 3D Translation vector
    transform = itk.TranslationTransform[itk.D, 3].New()
    transform.SetOffset(translation)
    itk.transformwrite(transform, "transforms/pos_1_init_transform.0.txt")

# Convert to ITK images and set spacing and origin
moving_image_sparse = create_itk_view(moving_array_sparse)
scale_spacing_and_origin(moving_image_sparse, factor)

fixed_image_sparse = create_itk_view(fixed_array_sparse)
scale_spacing_and_origin(fixed_image_sparse, factor)

if False:
    # Load the full resolution moving image
    moving_array_dense = load_tiff(moving_path)
    center_roi = (1800, 1800, 1800)
    moving_array_dense = center_crop(moving_array_dense, center_roi)
    moving_image_dense = create_itk_view(moving_array_dense)

# Import Default Parameter Map
# The Elastix manual is here: https://courses.compute.dtu.dk/02502/docs/elastix-5.2.0-manual.pdf

parameter_object = itk.ParameterObject.New()
parameter_map = parameter_object.GetDefaultParameterMap('translation', 4)
parameter_map['AutomaticTransformInitialization'] = ['true']
parameter_map['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
#parameter_map['MaximumNumberOfIterations'] = (1000,) # not working
parameter_object.AddParameterMap(parameter_map)


# Call registration function
result_image_small, result_transform_parameters = itk.elastix_registration_method(
    fixed_image_sparse, moving_image_sparse,
    parameter_object=parameter_object,
    #initial_transform_parameter_object=transform,
    #initial_transform_parameter_file_name="transforms/pos_1_init_transform.0.txt",
    log_to_console=False)

# Check registration of sparse images
viz_slices(fixed_image_sparse, [319, 310, 305], axis=2)
viz_slices(result_image_small, [319, 310, 305], axis=2)

dense_origin = moving_image_dense.GetOrigin()
sparse_origin = moving_image_sparse.GetOrigin()

dense_spacing = moving_image_dense.GetSpacing()
sparse_spacing = moving_image_sparse.GetSpacing()

for dim in range(moving_image_dense.GetImageDimension()):
    assert dense_origin[dim] - 0.5*dense_spacing[dim] == sparse_origin[dim] - 0.5*sparse_spacing[dim]

dense_size = itk.size(moving_image_dense)
sparse_size = itk.size(moving_image_sparse)

for dim in range(moving_image_dense.GetImageDimension()):
    assert dense_size[dim] * dense_spacing[dim] == sparse_size[dim] * sparse_spacing[dim]

####### Verify the transform parameters #######

plt.figure()
plt.imshow(fixed_image_sparse[150,:,:])
plt.title("Fixed Image Sparse")

plt.figure()
plt.imshow(moving_image_sparse[150,:,:])
plt.title("Moving Image Sparse")

plt.figure()
plt.imshow(result_image_small[150,:,:])
plt.title("Result Image Small")
plt.show()

# Save all the images as Nifti files:
itk.imwrite(result_image_small, "results/Larch/result_image_small.nii.gz")
itk.imwrite(fixed_image_sparse, "results/Larch/fixed_image_sparse.nii.gz")
itk.imwrite(moving_image_sparse, "results/Larch/moving_image_sparse.nii.gz")

######## ChatGPT suggestion here: ########

# Update the transform parameters to use the moving image's spacing, origin, and size
result_transform_parameters.SetParameter("Size", [str(s) for s in itk.size(moving_image_dense)])
result_transform_parameters.SetParameter("Spacing", [str(s) for s in moving_image_dense.GetSpacing()])
result_transform_parameters.SetParameter("Origin", [str(o) for o in moving_image_dense.GetOrigin()])

##########################################

result_image_large = itk.transformix_filter(
    moving_image_dense,
    transform_parameter_object=result_transform_parameters,
    log_to_console=False)

# Save all the images as Nifti files:
#itk.imwrite(result_image_small, "results/result_image_small.nii.gz")
#itk.imwrite(result_image_large, "results/result_image_large.nii.gz")
#itk.imwrite(fixed_image_sparse, "results/fixed_image_sparse.nii.gz")
#itk.imwrite(moving_image_sparse, "results/moving_image_sparse.nii.gz")


print("Done")