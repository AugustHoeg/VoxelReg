
params = [-0.16068530359974353, 0.26684892042779546, 0.9502483290559255, 0.8518848094982622, 0.5237204946142994, -0.003019083811859058, -0.4984701641554564, 0.8090169943749475, -0.311478728421956, -175.08933638116275, 59.422493129608654, 95.87855101816379]

transform = get_itk_similarity_transform((51.44, 122.66, -27.33), (72.190, 2.610, -9.391), rot_center, 0.9, order="ZXY")
result_image_transform = itk.resample_image_filter(moving_image_sparse,
                          transform=transform,
                          use_reference_image=True,
                          reference_image=fixed_image_sparse)

from utils.utils_plot import viz_slices
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [201], savefig=False)
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [221], savefig=False, axis=1)
viz_multiple_images([fixed_image_sparse, moving_image_sparse, result_image_transform], [141], savefig=False, axis=2)

# parameter_object = itk.ParameterObject.New()
# affine_map = parameter_object.GetDefaultParameterMap('affine')
# parameter_object.AddParameterMap(affine_map)
# transformix_object = itk.TransformixFilter.New(moving_image_sparse)
# transformix_object.SetTransformParameterObject(parameter_object)
# #transformix_object.SetTransform(transform)
# transformix_object.UpdateLargestPossibleRegion()
# output_transformix = transformix_object.GetOutput()