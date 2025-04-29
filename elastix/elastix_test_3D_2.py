import itk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_plot import viz_slices, viz_multiple_images, viz_registration

# To generate a downsampled image, the spacing of the image should be increased 10-fold in both directions,
# when the number of pixels is decreased 10-fold in both directions.
def image_generator(box_coords, size, spacing):
    z1, z2, x1, x2, y1, y2 = box_coords
    image = np.zeros([size, size, size], np.float32)
    image[z1:z2, y1:y2, x1:x2] = 1
    image_itk = itk.image_view_from_array(image)
    image_itk.SetSpacing([spacing, spacing, spacing])
    origin = (spacing - 1) / 2
    image_itk.SetOrigin([origin, origin, origin])
    return image_itk

size_hr = 400
f = 4
d = 1
size_lr = int(size_hr / (d*f))

box_coords = (25,75,25,75,25,75)
box_coords_hr = [val * f * d for val in box_coords]

fixed_hr = image_generator(box_coords_hr, size_hr, 1.0)
fixed_lr = image_generator(box_coords, size_lr, f*d)

moving_box_coords = (25-10,75-10,25-10,75-10,25-10,75-10)
moving_hr = image_generator([val for val in moving_box_coords], size_hr, f)
moving_lr = image_generator([int(val/f) for val in moving_box_coords], size_lr, f*d)

plt.figure()
plt.imshow(fixed_hr[:,:,size_hr//2])
plt.show()

plt.figure()
plt.imshow(fixed_lr[:,:,size_lr//2])
plt.show()

plt.figure()
plt.imshow(moving_hr[:,:,size_hr//8])
plt.show()

plt.figure()
plt.imshow(moving_lr[:,:,size_lr//8])
plt.show()

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