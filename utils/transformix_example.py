import itk
import numpy as np

from utils.utils_elastix import get_itk_rigid_transform

project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/"

moving_path = project_path + "moving_scale_1.nii.gz"
fixed_path = project_path + "fixed_scale_4.nii.gz"
mask_path = project_path + "fixed_scale_4_mask.nii.gz"
out_name = "Femur_01_transformed"  # Name of the output file

moving_image = itk.imread(moving_path)
fixed_image = itk.imread(fixed_path)

rotation_angles_deg = [-55, -145, -54]  # Example rotation angles in degrees
translation_vec = [20, 31, 20]  # Example translation vector

transform = get_itk_rigid_transform(rotation_angles_deg, translation_vec, order="ZXY")

# Load Transformix Object
ImageType = itk.Image[itk.F, 3]
transformix_filter = itk.TransformixFilter[ImageType].New()

parameter_map = {
                 "Direction": ("-1", "0", "0", "0", "-1", "0", "0", "0", "1"),
                 "Index": ("0", "0", "0"),
                 "Origin": ("0", "0", "0"),
                 "Size": ("600", "600", "600"),
                 "Spacing": ("0.232", "0.232", "0.232")
                }

parameter_object = itk.ParameterObject.New()
parameter_object.AddParameterMap(parameter_map)

transformix_filter.SetMovingImage(moving_image)
transformix_filter.SetTransformParameterObject(parameter_object)
transformix_filter.SetTransform(transform)
transformix_filter.Update()

output_image = transformix_filter.GetOutput()

from utils.utils_plot import viz_slices, viz_multiple_images

viz_slices(fixed_image, slice_indices=[50, 100, 200, 300, 400], axis=1, savefig=False)
viz_slices(moving_image, slice_indices=[50, 100, 200, 300, 400], axis=1, savefig=False)
viz_slices(output_image, slice_indices=[50, 100, 200, 300, 400], axis=1, savefig=False)

print("Done")