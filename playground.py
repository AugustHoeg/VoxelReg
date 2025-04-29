import itk
import numpy as np


if __name__ == "__main__":

    #base_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Bamboo_A_bin1x1/"
    base_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    fixed_image_path = base_path + "Bamboo_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_recon_000.tiff"
    moving_image_path = base_path + "Bamboo_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon_000.tiff"

    # Load images with itk floats (itk.F). Necessary for elastix
    print("Loading fixed image...")
    fixed_image = itk.imread(fixed_image_path, itk.F)
    print("Loading moving image...")
    moving_image = itk.imread(moving_image_path, itk.F)

    # we will use a default parametermap of elastix, for more info about parametermaps, see:
    # https://github.com/InsightSoftwareConsortium/ITKElastix/blob/e9d8d553c6179a3376a89843a1bc47880dd7ca85/examples/ITK_Example02_CustomOrMultipleParameterMaps.ipynb#section_id2

    parameter_object = itk.ParameterObject.New()
    default_similarity_parameter_map = parameter_object.GetDefaultParameterMap('similarity')
    parameter_object.AddParameterMap(default_similarity_parameter_map)

    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    # elastix_object.SetFixedImage(fixed_image)
    # elastix_object.SetMovingImage(moving_image)
    elastix_object.SetParameterObject(parameter_object)

    # Set additional options
    elastix_object.SetLogToConsole(False)

    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()

    # Results of Registration
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    # Save image with itk
    itk.imwrite(result_image, base_path + "registration_output.tiff")