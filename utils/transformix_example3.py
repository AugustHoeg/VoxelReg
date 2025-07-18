
tx2 = get_itk_affine_transform([-0.15945393738619706, 0.25756359580693305, 0.8474877202455747, 0.7493381845372026, 0.4983798814694924, -0.010477544604729855, -0.47229940402940174, 0.7037602488500311, -0.3027454129991309, -99.39023435010945, 49.48104092401816, 81.94456138124868])

result_image_transform = itk.resample_image_filter(moving_image_sparse,
                          transform=tx2,
                          use_reference_image=True,
                          reference_image=fixed_image_sparse)
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [201], savefig=False)
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [221], savefig=False, axis=1)
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [141], savefig=False, axis=2)

center_mm = [72.190, 2.610, -9.391]
initial_rotation_angles = [51.44, 122.66, -27.33]
fixed_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/fixed_scale_4.nii.gz")
moving_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/moving_scale_1.nii.gz")
transform.SetCenter(fixed_image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2 for sz in moving_image.GetSize()]))
transform.GetCenter()

# parameter_object = itk.ParameterObject.New()
# affine_map = parameter_object.GetDefaultParameterMap('affine')
# parameter_object.AddParameterMap(affine_map)
# transformix_object = itk.TransformixFilter.New(moving_image_sparse)
# transformix_object.SetTransformParameterObject(parameter_object)
# #transformix_object.SetTransform(transform)
# transformix_object.UpdateLargestPossibleRegion()
# output_transformix = transformix_object.GetOutput()


import SimpleITK as sitk
fixed_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/fixed_scale_4.nii.gz")
moving_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/moving_scale_1.nii.gz")
transform_file_name = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/transform.txt"
tx = sitk.ReadTransform(transform_file_name)

resampled_moving_image = sitk.Resample(moving_image, fixed_image, tx)
sitk.WriteImage(resampled_moving_image, "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/sitk_resample_test.nii.gz")



rotation_center = fixed_image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2 for sz in moving_image.GetSize()]) #[-147.84199476,  -92.27799749,   92.91599798]
theta_x = 51.44 * np.pi / 180
theta_y = 122.66 * np.pi / 180
theta_z = -27.33 * np.pi / 180
translation = [72.19,   2.61,  -9.391]

rigid_euler = sitk.Euler3DTransform(
    rotation_center, theta_x, theta_y, theta_z, translation
)

similarity = sitk.Similarity3DTransform()
similarity.SetMatrix(rigid_euler.GetMatrix())
similarity.SetTranslation(rigid_euler.GetTranslation())
similarity.SetCenter(rigid_euler.GetCenter())
similarity.SetScale(0.9)

fixed_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/fixed_scale_4.nii.gz")
moving_image = sitk.ReadImage("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/moving_scale_1.nii.gz")
resampled_moving_image = sitk.Resample(moving_image, fixed_image, similarity)
sitk.WriteImage(resampled_moving_image, "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/sitk_resample_test.nii.gz")
