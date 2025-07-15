import itk
import numpy as np

from utils.utils_elastix import get_itk_rigid_transform, get_default_parameter_object, get_elastix_registration_object
from utils.utils_itk import voxel2world

project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/"

moving_path = project_path + "moving_scale_1.nii.gz"
fixed_path = project_path + "fixed_scale_4.nii.gz"
mask_path = project_path + "fixed_scale_4_mask.nii.gz"
out_name = "Femur_01_transformed"  # Name of the output file

moving_image = itk.imread(moving_path)
fixed_image = itk.imread(fixed_path)

rotation_angles_deg = [55, 145, -54]  # Example rotation angles in degrees
translation_vec = [20, 31, 20]  # Example translation vector
fixed_image_center = voxel2world(fixed_image, np.array(fixed_image.shape) / 2)

transform = get_itk_rigid_transform(rotation_angles_deg, translation_vec, rot_center=fixed_image_center, order="ZXY")

parameter_object, parameter_map = get_default_parameter_object('translation')
elastix_object = get_elastix_registration_object(fixed_image, moving_image, parameter_object, log_mode=None)

elastix_object.SetInitialTransform(transform)

# Run the registration
elastix_object.UpdateLargestPossibleRegion()  # Update filter object (required)
result_image = elastix_object.GetOutput()
result_transform_object = elastix_object.GetTransformParameterObject()

from utils.utils_plot import viz_slices, viz_multiple_images

viz_slices(fixed_image, slice_indices=[50, 100, 200, 300], axis=1, savefig=False)
viz_slices(moving_image, slice_indices=[50, 100, 200, 300], axis=1, savefig=False)
viz_slices(result_image, slice_indices=[50, 100, 200, 300], axis=1, savefig=False)

print("Done")