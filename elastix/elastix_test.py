import itk
import numpy as np
import matplotlib.pyplot as plt

# To generate a downsampled image, the spacing of the image should be increased 10-fold in both directions,
# when the number of pixels is decreased 10-fold in both directions.
def image_generator(x1, x2, y1, y2, downsampled=False):
    if downsampled:
        image = np.zeros([100, 100], np.float32)
    else:
        image = np.zeros([1000, 1000], np.float32)
    image[y1:y2, x1:x2] = 1
    image_itk = itk.image_view_from_array(image)
    if downsampled:
        old_spacing = 1.0
        factor = 1000 / 100
        new_spacing = old_spacing * factor
        image_itk.SetSpacing([new_spacing, new_spacing])
        old_origin = 0.0
        # The start of image's domain, origin-spacing/2.0, should be the same
        new_origin = (new_spacing - old_spacing) / 2
        image_itk.SetOrigin([new_origin, new_origin])
    return image_itk


# Create sparsely sampled images (fewer pixels) for registration
fixed_image_sparse = image_generator(25,75,25,75, downsampled=True)
moving_image_sparse = image_generator(15,65,25,65, downsampled=True)

# .. and a densely sampled moving image (more pixels) for transformation
moving_image_dense = image_generator(15*10,65*10,25*10,65*10)

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

plt.figure()
plt.imshow(fixed_image_sparse)
plt.title("Fixed Image Sparse")

plt.figure()
plt.imshow(moving_image_sparse)
plt.title("Moving Image Sparse")

plt.figure()
plt.imshow(moving_image_dense)
plt.title("Moving Image Dense")

plt.figure()
plt.imshow(result_image_small)
plt.title("Result Image Small")

plt.figure()
plt.imshow(result_image_large)
plt.title("Result Image Large")

plt.show()



print("Done")