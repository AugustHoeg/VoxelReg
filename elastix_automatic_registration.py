import os
import argparse
import itk
import numpy as np
from utils.utils_elastix import elastix_coarse_registration_sweep, elastix_refined_registration
from utils.utils_itk import create_itk_view, scale_spacing_and_origin
from utils.utils_tiff import load_tiff


# Define paths
project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = project_path + "Larch_A_bin1x1/processed/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

moving_path = sample_path + "Larch_A_LFOV_crop_full_height.npy"
fixed_path = sample_path + "Larch_A_4x_pos1_down_4.npy"
out_name = "Larch_A_LFOV_registered"  # Name of the output file

# Load downsampled images
f = 4  # Resolution factor between the two images
d = 2  # Downsampling factor
tdf = f * d  # Total downsampling factor

#moving_array_sparse = np.load(sample_path + "Larch_A_LFOV_crop_full_height.npy")
#fixed_array_sparse = np.load(sample_path + "Larch_A_4x_pos1_down_4.npy")

# even newer, now works better
#moving_array_sparse = load_tiff(sample_path + "Larch_A_LFOV_crop_full_height.tiff")
#fixed_array_sparse = load_tiff(sample_path + "Larch_A_4x_pos1_down_4.tiff")

# Newer, also works, but slightly different alignment
#moving_array_sparse = load_tiff(sample_path + "test2/" + "Larch_A_LFOV_crop.tiff")
#fixed_array_sparse = load_tiff(sample_path + "test2/" + "Larch_A_4x_crop_pos1.tiff")
#fixed_array_sparse = fixed_array_sparse[:, 50:-50, 50:-50]
#fixed_array_sparse = fixed_array_sparse

# Old, works
#moving_array_sparse = load_tiff(sample_path + "Larch_A_LFOV_crop.tiff")
#fixed_array_sparse = load_tiff(sample_path + "Larch_A_4x_crop_pos1.tiff")
#fixed_array_sparse = fixed_array_sparse[:, 25:-25, 25:-25]

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--fixed_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--moving_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")

    parser.add_argument("--center", type=float, default=[1450, 161, 161], help="Initial guess for coarse registration, formatted as [D, H, W]")
    parser.add_argument("--size", type=float, default=(1, 1, 1), help="Number of coords around initial guess in (x,y,z) to apply coarse registration")
    parser.add_argument("--spacing", type=float, default=(25, 20, 20), help="Voxel spacing in (x,y,z) between coarse registration coords")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path, "processed/")
    if args.moving_path is not None:
        moving_path = os.path.join(sample_path, args.moving_path)
    if args.fixed_path is not None:
        fixed_path = os.path.join(sample_path, args.fixed_path)
    if args.out_name is not None:
        out_name = args.out_name  # out_name = os.path.join(sample_path, args.out_name)

    moving_array_sparse = np.load(moving_path)
    fixed_array_sparse = np.load(fixed_path)

    #moving_array_sparse = np.load(sample_path + "Larch_A_LFOV_crop_full_height.npy")
    #fixed_array_sparse = np.load(sample_path + "Larch_A_4x_pos1_down_4.npy")

    # Convert to ITK images and set spacing and origin
    moving_image_sparse = create_itk_view(moving_array_sparse)
    scale_spacing_and_origin(moving_image_sparse, 1.0)

    fixed_image_sparse = create_itk_view(fixed_array_sparse)
    scale_spacing_and_origin(fixed_image_sparse, 1.0)

    # Coarse registration parameters
    #center = [731, 65, 65]  # None
    #center = [1459, 161, 161]  # If any value is None, will pick geometric center for xy and top for z
    #spacing = (25, 20, 20)
    #size = (1, 1, 1)
    center = args.center
    spacing = args.spacing
    size = args.size

    # Run coarse registration via sweep
    result_coarse, coarse_trans_obj, metric = elastix_coarse_registration_sweep(
        fixed_image_sparse,
        moving_image_sparse,
        center=center,
        spacing=spacing,
        size=size,
        log_mode=None,
        visualize=True,
        fig_name=out_name
    )

    # Refined registration parameters
    registration_models = ['affine', 'bspline']
    resolution_list = [4, 4]
    max_iteration_list = [256, 256]

    # Refine registration
    result_refined, refined_trans_obj = elastix_refined_registration(
        fixed_image_sparse,
        moving_image_sparse,
        coarse_trans_obj,
        registration_models,
        resolution_list,
        max_iteration_list,
        log_mode=None,
        visualize=True,
        fig_name=out_name
    )

    # Save results
    result_array = itk.array_view_from_image(result_refined)

    # Convert to float
    result_array = result_array.astype(np.float32)

    # Enforce normalization to [0, 1]
    result_array = (result_array - np.min(result_array)) / (np.max(result_array) - np.min(result_array))

    #np.save(sample_path + "Larch_LFOV_pos1_registered.npy", result_array)

    out_path = os.path.join(sample_path, out_name + ".npy")
    np.save(out_path, result_array)

    print(f"Registration complete, output saved to {out_path}")