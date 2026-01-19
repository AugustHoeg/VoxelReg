import os
import argparse

import dask.array as da
import itk
import numpy as np
from skimage.exposure import match_histograms
from utils.utils_elastix import elastix_coarse_registration_sweep, elastix_refined_registration
from utils.utils_itk import create_itk_view, scale_spacing_and_origin, crop_itk_image, cast_itk, set_itk_metadata_from_affine
from utils.utils_tiff import load_tiff, write_tiff
from utils.utils_preprocess import apply_image_mask, compute_crop_bounds, minmax_scaler
from utils.utils_image import load_image
from utils.utils_plot import viz_multiple_images, viz_slices
from utils.utils_nifti import get_affine_from_itk_image
from utils.utils_zarr import create_ome_group, write_ome_level

# Define paths
project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = project_path + "Larch_A_bin1x1/processed/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

moving_path = sample_path + "Larch_A_LFOV_crop_full_height.npy"
fixed_path = sample_path + "Larch_A_4x_pos1_down_4.npy"
out_name = "Larch_A_LFOV_registered"  # Name of the output file


# Define paths
# sample_path = project_path + "unregistered/"
#moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"
#fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"

sample_path = project_path + "Oak_A/"
moving_path = sample_path + "moving_scale_1.nii.gz"
fixed_path = sample_path + "fixed_scale_4.nii.gz"
mask_path = sample_path + "fixed_scale_4_mask.nii.gz"
out_path = sample_path
out_name = "Oak_A"  # Name of the output file


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
    parser.add_argument("--out_name", type=str, required=False, default="output", help="Output name for the registered output image.")
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

    parser.add_argument("--moving_image_roi", type=int, nargs=3, default=(np.inf, np.inf, np.inf), help="Region of interest for the moving image in (D, H, W) format. Default is (inf, inf, inf).")
    parser.add_argument("--top_index", type=str, default='last', help="Index of the top slice for cropping. Default is 'last'. Can be 'first' or 'last'.")

    parser.add_argument("--match_slices", required=False, default=True, help="Match registered image slices to fixed image slices")
    parser.add_argument("--ome_level", required=False, default=0, help="Resolution level to write registered vol to in OME-Zarr store")

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
    if args.affine_transform_file is not None:  # args.affine_transform_file = "transform.txt"
        args.affine_transform_file = os.path.join(sample_path, args.affine_transform_file)
        print("Affine transform file: ", args.affine_transform_file)

    #####
    print("REMOVE THIS")
    args.affine_transform_file = os.path.join(sample_path, "test_transform_v2.txt")  # REMOVE THIS
    args.mask_path = mask_path # REMOVE THIS
    args.moving_image_roi = [np.inf, 240, 240]
    #####

    filename, file_extension = os.path.basename(moving_path).split('.', 1)

    print(f"Loading Moving image: {moving_path}")
    moving_array, moving_metadata = load_image(moving_path, dtype=np.uint16, return_metadata=True)
    moving_image = itk.image_from_array(moving_array)    # Convert to ITK image
    set_itk_metadata_from_affine(moving_image, moving_metadata.affine)

    print(f"Loading Fixed image: {fixed_path}")
    fixed_array_sparse, fixed_metadata = load_image(fixed_path, dtype=np.uint16, return_metadata=True)
    fixed_image_sparse = itk.image_from_array(fixed_array_sparse)     # Convert to ITK image
    set_itk_metadata_from_affine(fixed_image_sparse, fixed_metadata.affine)

    if args.mask_path is not None:
        print(f"Loading Mask image: {mask_path}")
        mask_array, mask_metadata = load_image(mask_path, dtype=np.uint8, return_metadata=True)
        mask_image_sparse = itk.image_from_array(mask_array)  # Convert to ITK image
        set_itk_metadata_from_affine(mask_image_sparse, mask_metadata.affine)
        mask_array = itk.array_view_from_image(mask_image_sparse)

    # if file_extension == "nii" or file_extension == "nii.gz":
    #     moving_image = itk.imread(moving_path)
    #     fixed_image_sparse = itk.imread(fixed_path)
    #     if args.mask_path is not None:
    #         mask_image_sparse = itk.imread(mask_path)
    #         mask_array_sparse = itk.array_from_image(mask_image_sparse)
    #
    # elif file_extension == "tiff" or file_extension == "tif":
    #     moving_array_sparse = load_tiff(moving_path)
    #     fixed_array_sparse = load_tiff(fixed_path)
    #
    #     # Convert to ITK images and set spacing and origin
    #     moving_image = create_itk_view(moving_array_sparse)
    #     scale_spacing_and_origin(moving_image, 1.0)
    #
    #     fixed_image_sparse = create_itk_view(fixed_array_sparse)
    #     scale_spacing_and_origin(fixed_image_sparse, 1.0)
    #
    #     if args.mask_path is not None:
    #         mask_array_sparse = load_tiff(mask_path)
    #         mask_image_sparse = create_itk_view(mask_array_sparse)
    #         scale_spacing_and_origin(mask_image_sparse, 1.0)
    #
    # elif file_extension == "npy":
    #     moving_array_sparse = np.load(moving_path)
    #     fixed_array_sparse = np.load(fixed_path)
    #
    #     # Convert to ITK images and set spacing and origin
    #     moving_image = create_itk_view(moving_array_sparse)
    #     scale_spacing_and_origin(moving_image, 1.0)
    #
    #     fixed_image_sparse = create_itk_view(fixed_array_sparse)
    #     scale_spacing_and_origin(fixed_image_sparse, 1.0)
    #
    #     if args.mask_path is not None:
    #         mask_array_sparse = np.load(mask_path)
    #         mask_image_sparse = create_itk_view(mask_array_sparse)
    #         scale_spacing_and_origin(mask_image_sparse, 1.0)
    # else:
    #     raise ValueError(f"Unsupported file extension: {file_extension}")


    # Set FOV of fixed and moving image
    print(f"Cropping moving image to ROI: {args.moving_image_roi} with top_index: '{args.top_index}'")
    start_crop, end_crop = compute_crop_bounds(moving_image, args.moving_image_roi[::-1], args.top_index, slice_axis=2)
    moving_image_sparse = crop_itk_image(moving_image, start_crop[::-1], end_crop[::-1])  # should be long

    # Coarse registration parameters
    #center = [731, 65, 65]  # None
    #center = [1459, 161, 161]  # If any value is None, will pick geometric center for xy and top for z
    #spacing = (25, 20, 20)
    #size = (1, 1, 1)
    center = args.center
    spacing = args.spacing
    size = args.size

    # Convert to float for registration accuracy
    fixed_image_sparse = cast_itk(fixed_image_sparse, input_dtype=itk.US, output_dtype=itk.F)
    moving_image_sparse = cast_itk(moving_image_sparse, input_dtype=itk.US, output_dtype=itk.F)

    # from utils.utils_plot import viz_slices
    # viz_slices(moving_image_sparse, [100], axis=2, savefig=False)
    # viz_slices(fixed_image_sparse, [100], axis=2, savefig=False)

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
        fig_name=out_name,
        save_dir=out_path
    )

    # Refined registration parameters
    registration_models = args.fine_registration_models  # ['rigid', 'affine']  # 'bspline'
    resolution_list = [args.fine_resolutions] * len(registration_models)
    max_iteration_list = [512, 512, 512]
    metric_list = ['AdvancedMattesMutualInformation',
                   'AdvancedMattesMutualInformation',
                   'AdvancedMattesMutualInformation']
    no_registration_samples_list = [4096, 4096, 4096]
    write_result_image_list = [False] * len(registration_models)  # Do not write intermediate results
    write_result_image_list[-1] = True  # Write the final result image

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
        fig_name=out_name,
        save_dir=out_path
    )

    print(f"Registration completed successfully. \n")

    # Save custom parameter map
    os.makedirs(os.path.join(sample_path, "parameter_maps"), exist_ok=True)
    output_parameter_file_name = "refined_parameters"
    output_transform_parameter_files = []
    for i in range(refined_trans_obj.GetNumberOfParameterMaps()):
        parameter_map_path = os.path.join(sample_path, "parameter_maps", f"{output_parameter_file_name}_{i}.txt")
        output_transform_parameter_files.append(parameter_map_path)

    refined_trans_obj.WriteParameterFile(refined_trans_obj, output_transform_parameter_files)
    print(f"Refined transform parameter maps saved to: {output_transform_parameter_files}")

    # Extract metadata
    origin = result_image.GetOrigin()
    spacing = result_image.GetSpacing()
    direction = result_image.GetDirection()

    # Get array for normalization
    result_array = itk.array_view_from_image(result_image)  # .astype(np.float32)
    minmax_scaler(result_array, vmin=0, vmax=65535)  #
    result_array = result_array.astype(np.uint16)

    # Apply fixed mask
    if args.mask_path is not None:
        #apply_image_mask(result_array, mask_array_sparse)  # zero values outside mask in-place
        apply_image_mask(result_array, mask_array)  # zero values outside mask in-place
        #apply_image_mask(fixed_image_sparse, mask_array)

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
    # write_nifti(result_array, affine=get_affine_from_itk_image(result_refined), output_path=full_out_path)
    print(f"Output saved to {full_out_path}")

    H, W, D = result_array.shape  # slice axis is last in ITK
    # viz_multiple_images([result_image, fixed_image_sparse], D // 5 * np.arange(1,4), axis=2, savefig=False)

    if args.match_slices:
        if args.mask_path is not None:
            source_vals = np.where(mask_array.astype(bool), result_array, np.nan)
            reference_vals = np.where(mask_array.astype(bool), itk.array_view_from_image(fixed_image_sparse), np.nan)
            # Check source and reference vals
            viz_multiple_images([source_vals, reference_vals], D // 5 * np.arange(1, 4), axis=2, save_dir=out_path, title=out_name + "_src_ref", savefig=True)

            for slice_idx in range(D):
                if slice_idx % 100 == 0:
                    print(f"Matching slice {slice_idx}/{D}")
                matched_slice = match_histograms(source_vals[:, :, slice_idx], reference_vals[:, :, slice_idx])
                matched_slice = np.nan_to_num(matched_slice, nan=0)  # fill nans with 0
                result_array[:, :, slice_idx] = matched_slice
        else:
            reference_vals = itk.array_view_from_image(fixed_image_sparse)
            for slice_idx in range(D):
                if slice_idx % 100 == 0:
                    print(f"Matching slice {slice_idx}/{D}")
                matched_slice = match_histograms(result_array[:, : slice_idx], reference_vals[:, :, slice_idx])
                result_array[:, :, slice_idx] = matched_slice

    # Convert to ITK image
    result_image = itk.image_view_from_array(result_array)
    result_image.SetOrigin(origin)
    result_image.SetSpacing(spacing)
    result_image.SetDirection(direction)

    viz_multiple_images([result_image, fixed_image_sparse], D // 5 * np.arange(1, 4), axis=2, save_dir=out_path, title=out_name + "_matched", savefig=True)

    full_out_path = os.path.join(out_path, out_name + "_matched.npy")
    np.save(full_out_path, result_array)
    print(f"Output saved to {full_out_path}")

    full_out_path = os.path.join(out_path, out_name + "_matched.tiff")
    write_tiff(result_array, full_out_path)
    print(f"Output saved to {full_out_path}")

    full_out_path = os.path.join(out_path, out_name + "_matched.nii.gz")
    itk.imwrite(result_image, full_out_path)
    #write_nifti(result_array, affine=get_affine_from_itk_image(result_refined), output_path=full_out_path)
    print(f"Output saved to {full_out_path}")

    # Write registered vol to OME-Zarr
    group_name = "REG"
    ome_path = os.path.join(out_path, args.out_name + "_ome.zarr")  # TODO ensure correct ome path

    # Create OME group with metadata
    store, group = create_ome_group(ome_path, group_name=group_name, pyramid_depth=2)  # TODO what should be pyramid depth?

    # Write fixed to level 0
    result_array = da.from_array(result_array)
    result_array = result_array.reshape(D, H, W)
    result_array = result_array.rechunk((160, 160, 160))
    registered = write_ome_level(result_array, store, group_name, level=args.ome_level)

    print("Done")