import os
import glob
import argparse
import itk
import numpy as np
from utils.utils_elastix import elastix_coarse_registration_sweep, elastix_refined_registration
from utils.utils_itk import create_itk_view, scale_spacing_and_origin, apply_registration_transform
from utils.utils_tiff import load_tiff, write_tiff
from utils.utils_preprocess import norm, masked_norm
from utils.utils_plot import viz_slices, viz_multiple_images


# Define paths
project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = project_path + "Larch_A_bin1x1/processed/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"

moving_path = sample_path + "Larch_A_LFOV_crop_full_height.npy"
out_name = "Larch_A_LFOV_registered"  # Name of the output file


# Define paths
sample_path = project_path + "unregistered/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"

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
    parser.add_argument("--moving_path", type=str, required=False, help="Path to moving image.")
    parser.add_argument("--fixed_path", type=str, required=False, help="Path to fixed image. Used to infer size of output image.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")

    parser.add_argument("--transform_parameter_map_name", type=str, required=False, default="refined_parameters")
    parser.add_argument("--mask_path", type=str, required=False, default=None, help="Path to the mask image.")
    parser.add_argument("--moving_image_is_binary", default=False, help="Flag to indicate if the moving image is binary.")

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

    filename, file_extension = os.path.basename(moving_path).split('.', 1)

    if file_extension == "nii" or file_extension == "nii.gz":
        moving_image = itk.imread(moving_path)
        if args.fixed_path is not None:
            fixed_image = itk.imread(fixed_path)
        if args.mask_path is not None:
            mask_image = itk.imread(mask_path)
            mask_array = itk.array_from_image(mask_image)

    elif file_extension == "tiff" or file_extension == "tif":
        moving_array = load_tiff(moving_path)

        # Convert to ITK images and set spacing and origin
        moving_image = create_itk_view(moving_array)
        scale_spacing_and_origin(moving_image, 1.0)

        if args.fixed_path is not None:
            fixed_array = load_tiff(fixed_path)
            fixed_image = create_itk_view(fixed_array)
            scale_spacing_and_origin(fixed_image, 1.0)

        if args.mask_path is not None:
            mask_array = load_tiff(mask_path)
            mask_image = create_itk_view(mask_array)
            scale_spacing_and_origin(mask_image, 1.0)

    elif file_extension == "npy":
        moving_array = np.load(moving_path)

        # Convert to ITK images and set spacing and origin
        moving_image = create_itk_view(moving_array)
        scale_spacing_and_origin(moving_image, 1.0)

        if args.fixed_path is not None:
            fixed_array = np.load(fixed_path)
            fixed_image = create_itk_view(fixed_array)
            scale_spacing_and_origin(fixed_image, 1.0)

        if args.mask_path is not None:
            mask_array = np.load(mask_path)
            mask_image = create_itk_view(mask_array)
            scale_spacing_and_origin(mask_image, 1.0)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


    # Read custom parameter map
    refined_trans_obj = itk.ParameterObject().New()

    # Load parameter maps
    output_transform_parameter_files = glob.glob(os.path.join(sample_path, "parameter_maps", f"{args.transform_parameter_map_name}*.txt"))
    refined_trans_obj.ReadParameterFiles(output_transform_parameter_files)
    print(f"Loaded parameter files: {output_transform_parameter_files}")

    # Adjust parameter file with spacing and size of moving image (may be larger!).
    if args.moving_image_is_binary:
        refined_trans_obj.SetParameter('FinalBSplineInterpolationOrder', '0')  # Set if transforming a binary image

    if args.fixed_path is not None:  # If fixed image is provided, use its size for transformation output
        refined_trans_obj.SetParameter("Size", [f'{val}' for val in list(itk.size(fixed_image))])
    else:
        refined_trans_obj.SetParameter("Size", [f'{val}' for val in list(itk.size(moving_image))])

    refined_trans_obj.SetParameter("Spacing", [f'{val}' for val in list(moving_image.GetSpacing())])

    # Apply registration transform to moving image
    print("Applying registration transform to moving image...")
    result_image = apply_registration_transform(moving_image, refined_trans_obj)

    if args.fixed_path is not None:
        # Visualize the results
        for axis in range(3):
            dim = min(fixed_image.shape[axis], result_image.shape[axis])
            off = int(dim * 0.05)  # offset for visualization
            diff = fixed_image[:] - result_image[:]
            viz_multiple_images([fixed_image, result_image, diff],
                                [dim - i * off - 5 for i in range(3)],
                                axis=axis, savefig=True, title=out_name + f"_transformed_axis_{axis}")

    # Extract metadata
    origin = result_image.GetOrigin()
    spacing = result_image.GetSpacing()
    direction = result_image.GetDirection()

    # Get array for normalization
    result_array = itk.array_view_from_image(result_image)  # .astype(np.float32)

    # Enforce normalization to [0, 1]
    if args.mask_path is not None:
        masked_norm(result_array, mask_array)  # in-place
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