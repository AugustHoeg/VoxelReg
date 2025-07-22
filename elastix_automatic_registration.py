import os
import argparse
import itk
import numpy as np
from utils.utils_elastix import elastix_coarse_registration_sweep, elastix_refined_registration
from utils.utils_itk import create_itk_view, scale_spacing_and_origin
from utils.utils_tiff import load_tiff, write_tiff
from utils.utils_preprocess import norm, masked_norm


# Define paths
project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = project_path + "Larch_A_bin1x1/processed/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

moving_path = sample_path + "Larch_A_LFOV_crop_full_height.npy"
fixed_path = sample_path + "Larch_A_4x_pos1_down_4.npy"
out_name = "Larch_A_LFOV_registered"  # Name of the output file


# Define paths
sample_path = project_path + "unregistered/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

sample_path = project_path + "Femur_74/"
moving_path = sample_path + "moving_scale_1.nii.gz"
fixed_path = sample_path + "fixed_scale_4.nii.gz"
mask_path = sample_path + "fixed_scale_4_mask.nii.gz"
out_name = "Femur_74_registered"  # Name of the output file


# Load downsampled images
f = 4  # Resolution factor between the two images
d = 2  # Downsampling factor
tdf = f * d  # Total downsampling factor

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--fixed_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--moving_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")

    parser.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Initial guess for coarse registration, formatted as [D, H, W]")
    parser.add_argument("--rotation_angles_deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Initial guess for coarse registration rotation angles in degrees")
    parser.add_argument("--scale", type=float, default=1.0, help="Initial guess for coarse registration scale factor.")
    parser.add_argument("--affine_transform_file", type=str, default=None, help="Path to an affine transform file (txt file).")

    parser.add_argument("--coarse_resolutions", type=int, default=4, help="Resolutions for coarse registration.")
    parser.add_argument("--fine_resolutions", type=int, default=4, help="Resolutions for fine registration.")

    parser.add_argument("--fine_registration_models", type=str, nargs='*', default=['rigid', 'affine'], help="Models for fine registration. Default is ['rigid', 'affine']")

    parser.add_argument("--size", type=int, nargs=3, default=(1, 1, 1), help="Number of coords around initial guess in (x,y,z) to apply coarse registration")
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.25, 0.25, 0.25), help="Voxel spacing in (x,y,z) between coarse registration coords")

    parser.add_argument("--mask_path", type=str, required=False, default=None, help="Path to the mask image.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU_HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path)
    if args.moving_path is not None:
        moving_path = os.path.join(sample_path, args.moving_path)
    if args.fixed_path is not None:
        fixed_path = os.path.join(sample_path, args.fixed_path)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # out_name = os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)
    if args.mask_path is not None:
        mask_path = os.path.join(sample_path, args.mask_path)
        print("Mask path: ", mask_path)
    if args.affine_transform_file is not None:
        args.affine_transform_file = os.path.join(sample_path, args.affine_transform_file)
        print("Affine transform file: ", args.affine_transform_file)

    filename, file_extension = os.path.basename(moving_path).split('.', 1)

    if file_extension == "nii" or file_extension == "nii.gz":
        moving_image_sparse = itk.imread(moving_path)
        fixed_image_sparse = itk.imread(fixed_path)
        if args.mask_path is not None:
            mask_image_sparse = itk.imread(mask_path)
            mask_array_sparse = itk.array_from_image(mask_image_sparse)

    elif file_extension == "tiff" or file_extension == "tif":
        moving_array_sparse = load_tiff(moving_path)
        fixed_array_sparse = load_tiff(fixed_path)

        # Convert to ITK images and set spacing and origin
        moving_image_sparse = create_itk_view(moving_array_sparse)
        scale_spacing_and_origin(moving_image_sparse, 1.0)

        fixed_image_sparse = create_itk_view(fixed_array_sparse)
        scale_spacing_and_origin(fixed_image_sparse, 1.0)

        if args.mask_path is not None:
            mask_array_sparse = load_tiff(mask_path)
            mask_image_sparse = create_itk_view(mask_array_sparse)
            scale_spacing_and_origin(mask_image_sparse, 1.0)

    elif file_extension == "npy":
        moving_array_sparse = np.load(moving_path)
        fixed_array_sparse = np.load(fixed_path)

        # Convert to ITK images and set spacing and origin
        moving_image_sparse = create_itk_view(moving_array_sparse)
        scale_spacing_and_origin(moving_image_sparse, 1.0)

        fixed_image_sparse = create_itk_view(fixed_array_sparse)
        scale_spacing_and_origin(fixed_image_sparse, 1.0)

        if args.mask_path is not None:
            mask_array_sparse = np.load(mask_path)
            mask_image_sparse = create_itk_view(mask_array_sparse)
            scale_spacing_and_origin(mask_image_sparse, 1.0)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


    # Coarse registration parameters
    #center = [731, 65, 65]  # None
    #center = [1459, 161, 161]  # If any value is None, will pick geometric center for xy and top for z
    #spacing = (25, 20, 20)
    #size = (1, 1, 1)
    center = args.center
    spacing = args.spacing
    size = args.size

    #ImageTypeOut = itk.Image[itk.F, 3]  # e.g., float32, 3D
    #moving_image_sparse = itk.cast_image_filter(moving_image_sparse, ttype=[type(moving_image_sparse), ImageTypeOut])
    #fixed_image_sparse = itk.cast_image_filter(fixed_image_sparse, ttype=[type(fixed_image_sparse), ImageTypeOut])

    # Run coarse registration via sweep
    result_coarse, coarse_trans_obj, metric = elastix_coarse_registration_sweep(
        fixed_image_sparse,
        moving_image_sparse,
        center_mm=center,
        initial_rotation_angles=args.rotation_angles_deg,
        initial_scale=args.scale,
        affine_transform_file=args.affine_transform_file,
        grid_spacing_mm=spacing,
        grid_size=size,
        resolutions=args.coarse_resolutions,
        max_iterations=512,  # 256, 512, 1024
        metric='AdvancedMattesMutualInformation',  # 'AdvancedNormalizedCorrelation', 'AdvancedMattesMutualInformation'
        no_registration_samples=4096,  # 2048, 4096
        log_mode=None,  # None, "console"
        visualize=True,
        fig_name=out_name
    )

    # Refined registration parameters
    registration_models = args.fine_registration_models  # ['rigid', 'affine']  # 'bspline'
    resolution_list = [args.fine_resolutions] * len(registration_models)
    max_iteration_list = [512, 512, 512]
    metric_list = ['AdvancedMattesMutualInformation',
                   'AdvancedMattesMutualInformation',
                   'AdvancedMattesMutualInformation']
    no_registration_samples_list = [4096, 4096, 4096]
    write_result_image_list = [False, True, True]

    # Refine registration
    result_image, refined_trans_obj = elastix_refined_registration(
        fixed_image_sparse,
        moving_image_sparse,
        coarse_trans_obj,
        registration_models,
        resolution_list,
        max_iteration_list,
        write_result_image_list,
        metric_list,
        no_registration_samples_list,
        log_mode=None,  # None, "console"
        visualize=True,
        fig_name=out_name
    )

    # Save custom parameter map
    output_transform_parameter_file = "refined_transform_parameters.txt"
    itk.WriteParameterFile(refined_trans_obj, os.path.join(sample_path, output_transform_parameter_file))
    #parameter_object.WriteParameterFile(parameter_map_custom, 'exampleoutput/parameters_custom.txt')

    print(f"Registration completed successfully. \n")

    # Extract metadata
    origin = result_image.GetOrigin()
    spacing = result_image.GetSpacing()
    direction = result_image.GetDirection()

    # Get array for normalization
    result_array = itk.array_view_from_image(result_image)  # .astype(np.float32)

    # Enforce normalization to [0, 1]
    if args.mask_path is not None:
        masked_norm(result_array, mask_array_sparse)  # in-place
    else:
        norm(result_array)  # in-place

    # Convert to ITK image
    result_image = itk.image_view_from_array(result_array)
    result_image.SetOrigin(origin)
    result_image.SetSpacing(spacing)
    result_image.SetDirection(direction)

    full_out_path = os.path.join(out_path, out_name + ".npy")
    np.save(full_out_path, result_array)
    print(f"Output saved to {full_out_path}")

    full_out_path = os.path.join(out_path, out_name + ".tiff")
    write_tiff(result_array, full_out_path)
    print(f"Output saved to {full_out_path}")

    full_out_path = os.path.join(out_path, out_name + ".nii.gz")
    itk.imwrite(result_image, full_out_path)
    #write_nifti(result_array, affine=get_affine_from_itk_image(result_refined), output_path=full_out_path)
    print(f"Output saved to {full_out_path}")