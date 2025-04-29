import itk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_plot import viz_slices, viz_multiple_images, viz_registration

# To generate a downsampled image, the spacing of the image should be increased 10-fold in both directions,
# when the number of pixels is decreased 10-fold in both directions.
def image_generator(z1, z2, x1, x2, y1, y2, downsampled=False):
    if downsampled:
        image = np.zeros([100, 100, 100], np.float32)
    else:
        image = np.zeros([400, 400, 400], np.float32)
    image[z1:z2, y1:y2, x1:x2] = 1
    image_itk = itk.image_view_from_array(image)
    if downsampled:
        old_spacing = 1.0
        factor = 400 / 100
        new_spacing = old_spacing * factor
        image_itk.SetSpacing([new_spacing, new_spacing, new_spacing])
        old_origin = 0.0
        # The start of image's domain, origin-spacing/2.0, should be the same
        new_origin = (new_spacing - old_spacing) / 2
        image_itk.SetOrigin([new_origin, new_origin, new_origin])
    return image_itk

size_small = 100
size_large = 400
factor = int(size_large / size_small)

# Create sparsely sampled images (fewer pixels) for registration
fixed_image_sparse = image_generator(25,75,25,75,25,75, downsampled=True)
moving_image_sparse = image_generator(15,65,15,65,25,65, downsampled=True)

# .. and a densely sampled moving image (more pixels) for transformation
moving_image_dense = image_generator(15*factor,65*factor,15*factor,65*factor,25*factor,65*factor)
fixed_image_dense = image_generator(25*factor,75*factor,25*factor,75*factor,25*factor,75*factor)

# Import Default Parameter Map
parameter_object = itk.ParameterObject.New()
default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['0']
parameter_object.AddParameterMap(default_affine_parameter_map)

# Call registration function
result_image_small, result_transform_parameters = itk.elastix_registration_method(
    fixed_image_sparse, moving_image_sparse,
    parameter_object=parameter_object,
    log_to_console=False)

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


viz_multiple_images([fixed_image_dense, result_image_large], [size_large//2, size_large//2, size_large//2], axis=2)
viz_registration(fixed_image_dense, result_image_large, [size_large//2, size_large//2, size_large//2], axis=2)

# plt.figure()
# plt.imshow(fixed_image_sparse[:,:,size_small//2])
# plt.title("Fixed Image Sparse")
#
# plt.figure()
# plt.imshow(moving_image_sparse[:,:,size_small//2])
# plt.title("Moving Image Sparse")
#
# plt.figure()
# plt.imshow(moving_image_dense[:,:,size_large//2])
# plt.title("Moving Image Dense")
#
# plt.figure()
# plt.imshow(result_image_small[:,:,size_small//2])
# plt.title("Result Image Small")
#
# plt.figure()
# plt.imshow(result_image_large[:,:,size_large//2])
# plt.title("Result Image Large")
#
# plt.show()

# Save all the images as Nifti files:
#itk.imwrite(result_image_small, "results/result_image_small.nii.gz")
#itk.imwrite(result_image_large, "results/result_image_large.nii.gz")
#itk.imwrite(fixed_image_sparse, "results/fixed_image_sparse.nii.gz")
#itk.imwrite(fixed_image_dense, "results/fixed_image_dense.nii.gz")
#itk.imwrite(moving_image_sparse, "results/moving_image_sparse.nii.gz")
#itk.imwrite(moving_image_dense, "results/moving_image_dense.nii.gz")

print("Done")