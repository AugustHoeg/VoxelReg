import os
import argparse

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar

import zarr
from ome_zarr.io import parse_url

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_preprocess import get_image_and_affine, save_image_pyramid, get_image_pyramid, define_image_space, get_dtype, scale_n_clip, mask_with_threshold, mask_with_cylinder, dtype_min_max
from utils.utils_nifti import voxel2world, set_origin
from utils.utils_dask import threshold_dask
from utils.utils_zarr import write_ome_metadata

# Define paths
project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/"

# Define paths
sample_path = project_path + "femur_001/"
moving_path = sample_path + "clinical/volume/f_001.nii"
fixed_path = sample_path + "micro/volume/f_001.nii"
out_name = "f_001_prepropress"  # Name of the output file

# args.moving_mask_method = "cylinder"
# args.moving_cylinder_radius = 400
# args.moving_cylinder_center_offset = (50, 200)
# args.apply_moving_mask = True

#project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Oak_A/"
#sample_path = project_path
#moving_path = sample_path + "Oak_A_bin1x1_LFOV_retake_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_Stitch_scale_4.tif"
#fixed_path = sample_path + "Oak_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_Stitch_scale_4.tif"

project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/"
sample_path = project_path
moving_path = sample_path + "Cardboard_A_LFOV_80kV_7W_air_4s_8mu_bin1_pos1_Stitch_scale_4.tif"
fixed_path = sample_path + "Cardboard_A_4X_80kV_7W_air_3s_2mu_bin1_pos1_Stitch_scale_4.tif"

out_name = "test"  # Name of the output file

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")

    parser.add_argument("--moving_path", type=str, required=False, help="Path to the scan file.")
    parser.add_argument("--fixed_path", type=str, required=False, help="Path to the scan file.")

    parser.add_argument("--moving_out_path", type=str, required=False, default="", help="Path to the output file.")
    parser.add_argument("--fixed_out_path", type=str, required=False, default="", help="Path to the output file.")
    parser.add_argument("--moving_out_name", type=str, required=False, default="moving", help="Output name for the processed image.")
    parser.add_argument("--fixed_out_name", type=str, required=False, default="fixed", help="Output name for the processed image.")

    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--dtype", type=str, default="UINT16", help="Data type of fixed/moving input images")

    parser.add_argument("--moving_min_size", type=int, nargs=3, default=(0, 480, 480), help="Minimum size for cropping.")
    parser.add_argument("--moving_max_size", type=int, nargs=3, default=(9999, 1920, 1920), help="Maximum size for cropping.")
    parser.add_argument("--fixed_min_size", type=int, nargs=3, default=(0, 480, 480), help="Minimum size for cropping.")
    parser.add_argument("--fixed_max_size", type=int, nargs=3, default=(9999, 1920, 1920), help="Maximum size for cropping.")

    parser.add_argument("--moving_pixel_size", type=float, nargs=3, default=(None, None, None), help="Pixel size in mm for moving image.")
    parser.add_argument("--fixed_pixel_size", type=float, nargs=3, default=(None, None, None), help="Pixel size in mm for fixed image.")

    parser.add_argument("--margin_percent", type=float, default=0.5, help="Margin percentage for cropping.")

    parser.add_argument("--moving_divis_factor", type=int, default=8, help="Divisibility factor for cropping highest resolution moving image.")
    parser.add_argument("--fixed_divis_factor", type=int, default=8, help="Divisibility factor for cropping highest resolution fixed image.")

    parser.add_argument("--moving_pyramid_depth", type=int, default=4, help="Depth of saved image pyramid.")
    parser.add_argument("--fixed_pyramid_depth", type=int, default=4, help="Depth of saved image pyramid.")

    parser.add_argument("--f", type=int, default=4, help="LR resolution factor.")
    parser.add_argument("--moving_clip_percentiles", type=float, nargs=2, default=(1.0, 99.0), help="Lower and upper percentiles for image normalization")
    parser.add_argument("--fixed_clip_percentiles", type=float, nargs=2, default=(1.0, 99.0), help="Lower and upper percentiles for image normalization")

    parser.add_argument("--moving_clip_range", type=float, nargs=2, default=(0, 65535), help="Lower and upper percentiles for image normalization")
    parser.add_argument("--fixed_clip_range", type=float, nargs=2, default=(0, 65535), help="Lower and upper percentiles for image normalization")

    parser.add_argument("--moving_mask_path", default=None, help="Path to moving mask image, default is None.")
    parser.add_argument("--moving_mask_method", default=None, help="Method for creating moving mask. Currently supports 'threshold' and 'cylinder'. Default is None, which skips mask creation.")
    parser.add_argument("--moving_mask_threshold", default=None, help="Threshold for binary mask image. If unspecified, otsu thresholding will be used. default is None.")
    parser.add_argument("--moving_cylinder_radius", type=int, default=None, help="Radius of the cylinder for moving mask in voxels.")
    parser.add_argument("--moving_cylinder_center_offset", type=int, nargs=2, default=(0, 0), help="Offset for the center of the cylinder mask in voxels, default is 0 (centered in H, W).")
    parser.add_argument("--apply_moving_mask", action="store_true", help="Apply moving mask to the image.")

    parser.add_argument("--fixed_mask_path", default=None, help="Path to fixed mask image, default is None.")
    parser.add_argument("--fixed_mask_method", default=None, help="Method for creating fixed mask. Currently supports 'threshold' and 'cylinder'. Default is None, which skips mask creation.")
    parser.add_argument("--fixed_mask_threshold", default=None, help="Threshold for binary mask image, default is None.")
    parser.add_argument("--fixed_cylinder_radius", type=int, default=None, help="Radius of the cylinder for fixed mask in voxels.")
    parser.add_argument("--fixed_cylinder_center_offset", type=int, nargs=2, default=(0, 0), help="Offset for the center of the cylinder mask in voxels, default is 0 (centered in H, W).")
    parser.add_argument("--apply_fixed_mask", action="store_true", help="Apply fixed mask to the image.")

    parser.add_argument("--top_index", type=str, default="last", help="Index for the top slice of the image, default is 'last'")

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
        print("Sample path: ", sample_path)
    if args.moving_path is not None:
        moving_path = os.path.join(sample_path, args.moving_path)
        print("Moving path: ", moving_path)
    if args.fixed_path is not None:
        fixed_path = os.path.join(sample_path, args.fixed_path)
        print("Fixed path: ", fixed_path)
    if args.moving_mask_path is not None:
        moving_mask_path = os.path.join(sample_path, args.moving_mask_path)
        print("Moving mask path: ", moving_mask_path)
    if args.fixed_mask_path is not None:
        fixed_mask_path = os.path.join(sample_path, args.fixed_mask_path)
        print("Fixed mask path: ", fixed_mask_path)
    if args.moving_out_path is not None:
        moving_out_path = os.path.join(sample_path, args.moving_out_path)  # os.path.join(sample_path, args.out_name)
        print("Moving output path: ", moving_out_path)
    if args.fixed_out_path is not None:
        fixed_out_path = os.path.join(sample_path, args.fixed_out_path)  # os.path.join(sample_path, args.out_name)
        print("Fixed output path: ", fixed_out_path)

    visualize = False  # True
    print("Visualization is set to: ", visualize)

    ##################### MOVING IMAGE ######################

    # Load moving image
    args.moving_pixel_size = (4, 4, 4)  # REMOVE THIS
    args.moving_divis_factor = 160  # REMOVE THIS
    args.moving_mask_method = 'threshold' # REMOVE THIS
    args.moving_mask_threshold = 100 # REMOVE THIS
    args.apply_moving_mask = True # REMOVE THIS

    input_dtype = get_dtype(args.dtype)
    moving, moving_affine = get_image_and_affine(moving_path,
                                                 custom_origin=(0, 0, 0),
                                                 pixel_size_mm=args.moving_pixel_size,
                                                 dtype=input_dtype,
                                                 backend="Dask")

    # Define moving image space
    moving, moving_affine, moving_crop_start, moving_crop_end = define_image_space(moving, moving_affine, f=args.f,
                                                                                   min_size=args.moving_min_size,
                                                                                   max_size=args.moving_max_size,
                                                                                   margin_percent=args.margin_percent,
                                                                                   divis_factor=args.moving_divis_factor,
                                                                                   top_index=args.top_index)

    # scale and clip
    moving = scale_n_clip(moving, args.moving_clip_range)

    # Test

    moving_ome_path = os.path.join(moving_out_path, args.moving_out_name + "_ome.zarr")
    # Create/open a Zarr array in write mode
    store = parse_url(moving_ome_path, mode="w").store
    # store = zarr.storage.LocalStore(moving_ome_path)
    root = zarr.group(store=store)

    group_name = "LR"
    out_path = moving_ome_path
    if os.path.exists(os.path.join(moving_ome_path, group_name)):
        print(f"Group {group_name} already exists in {out_path}. Skipping...")
    else:
        # Create image group for the volume
        image_group = root.create_group(group_name)

        write_ome_metadata(group=image_group, num_levels=args.moving_pyramid_depth, scale=2)

        group = image_group.create_group("0")
        da.to_zarr(moving, group=group, overwrite=True)

        print(f"Done writing OME-Zarr data to {out_path}/{group_name}")

    # End test



    # Create disk checkpoint to clean dask graph
    with ProgressBar(dt=1):
        path = os.path.join(moving_out_path, args.moving_out_name + ".zarr")
        #moving = checkpoint_as_zarr(moving, path, chunks=(1, *moving.shape[1:]))
        da.to_zarr(moving, path, overwrite=True)
        moving = da.from_zarr(path, chunks=moving.chunksize)

    # Get moving image mask
    moving_mask = None
    if args.moving_mask_path is not None:
        print(f"Using moving mask from: {moving_mask_path}")
        moving_mask, moving_mask_affine = get_image_and_affine(moving_mask_path,
                                                             custom_origin=(0, 0, 0),
                                                             pixel_size_mm=args.moving_pixel_size,
                                                             dtype=np.uint8,
                                                             backend="Dask")

        moving_mask, _, _, _ = define_image_space(moving_mask, moving_mask_affine, f=1,
                                                 min_size=args.moving_min_size,
                                                 max_size=args.moving_max_size,
                                                 margin_percent=0.0,
                                                 divis_factor=args.moving_divis_factor,
                                                 top_index=args.top_index)
        moving_mask_affine = moving_affine  # Defined, but currently unused

    elif args.moving_mask_method == "threshold":
        moving_mask = threshold_dask(moving, threshold=args.moving_mask_threshold, high=1, low=0, dtype=np.uint8)
        #moving_mask = mask_with_threshold(moving, mask_threshold=args.moving_mask_threshold)

    elif args.moving_mask_method == "cylinder":
        print(f"Creating moving mask using method: {args.moving_mask_method}")
        moving_mask = mask_with_cylinder(moving, cylinder_radius=args.moving_cylinder_radius, cylinder_offset=args.moving_cylinder_center_offset)
    else:
        print("No moving mask will be used.")

    # scale and clip
    moving_mask = scale_n_clip(moving_mask)

    # Create disk checkpoint to clean dask graph
    with ProgressBar(dt=1):
        path = os.path.join(moving_out_path, args.moving_out_name + "_mask.zarr")
        #moving_mask = checkpoint_as_zarr(moving, path, chunks=(1, *moving.shape[1:]))
        da.to_zarr(moving_mask, path, overwrite=True)
        moving_mask = da.from_zarr(path, chunks=moving_mask.chunksize)


    # Get & save moving image pyramid
    # args.moving_mask_method = 'threshold' # REMOVE THIS
    # args.moving_mask_threshold = 100 # REMOVE THIS
    # args.apply_moving_mask = True # REMOVE THIS
    pyramid, mask_pyramid, affines = get_image_pyramid(moving, moving_affine,
                                                       args.moving_pyramid_depth,
                                                       args.moving_clip_percentiles,
                                                       args.moving_clip_range,
                                                       moving_mask,
                                                       args.moving_mask_method,
                                                       args.moving_mask_threshold,
                                                       args.moving_cylinder_radius,
                                                       args.moving_cylinder_center_offset,
                                                       args.apply_moving_mask)

    print("Preparing to write moving image pyramid...")
    save_image_pyramid(pyramid, mask_pyramid, affines, moving_path, moving_out_path, args.moving_out_name)

    # Visualize
    if visualize:
        for i, image in enumerate(pyramid):
            slices = [image.shape[0]//2, image.shape[1]//2, image.shape[2]//2]
            viz_slices(image, slices[0], save_dir=moving_out_path, title=args.moving_out_name + f"_scale_{2 ** i}_pre_axis_0", axis=0)
            viz_slices(image, slices[1], save_dir=moving_out_path, title=args.moving_out_name + f"_scale_{2 ** i}_pre_axis_1", axis=1)
            viz_slices(image, slices[2], save_dir=moving_out_path, title=args.moving_out_name + f"_scale_{2 ** i}_pre_axis_2", axis=2)

    # Record moving image top center position
    p1 = [moving.shape[0], moving.shape[1] / 2, moving.shape[2] / 2]

    # Clear memory
    del moving, pyramid, mask_pyramid

    ##################### FIXED IMAGE ######################

    # Load fixed image
    # args.fixed_pixel_size = (1, 1, 1) # REMOVE THIS
    # args.fixed_divis_factor = 160  # REMOVE THIS
    fixed, fixed_affine = get_image_and_affine(fixed_path, custom_origin=(0, 0, 0), pixel_size_mm=args.fixed_pixel_size, dtype=input_dtype)

    # Define fixed image space
    fixed, fixed_affine, fixed_crop_start, fixed_crop_end = define_image_space(fixed, fixed_affine, f=1,
                                                                               min_size=args.fixed_min_size,
                                                                               max_size=args.fixed_max_size,
                                                                               margin_percent=0.0,
                                                                               divis_factor=args.fixed_divis_factor,
                                                                               top_index=args.top_index)

    # Set fixed origin to moving image top slice, centered in H, W
    p2 = [fixed.shape[0], fixed.shape[1] / 2, fixed.shape[2] / 2]
    pos_diff = voxel2world(moving_affine, p1) - voxel2world(fixed_affine, p2)
    set_origin(fixed_affine, new_origin=pos_diff)
    print("nifti affine after set pos\n", fixed_affine)

    # Get fixed image mask
    fixed_mask = None
    if args.fixed_mask_path is not None:
        print(f"Using fixed mask from: {fixed_mask_path}")
        fixed_mask, fixed_mask_affine = get_image_and_affine(fixed_mask_path,
                                                             custom_origin=(0, 0, 0),
                                                             pixel_size_mm=args.fixed_pixel_size,
                                                             dtype=np.uint8)

        fixed_mask, _, _, _ = define_image_space(fixed_mask, fixed_mask_affine, f=1,
                                                 min_size=args.fixed_min_size,
                                                 max_size=args.fixed_max_size,
                                                 margin_percent=0.0,
                                                 divis_factor=args.fixed_divis_factor,
                                                 top_index=args.top_index)
        fixed_mask_affine = fixed_affine  # Defined, but currently unused


    # Get & save moving image pyramid
    # args.fixed_mask_method = 'threshold' # REMOVE THIS
    # args.fixed_mask_threshold = 100 # REMOVE THIS
    # args.apply_fixed_mask = True # REMOVE THIS
    pyramid, mask_pyramid, affines = get_image_pyramid(fixed, fixed_affine,
                                                       args.fixed_pyramid_depth,
                                                       args.fixed_clip_percentiles,
                                                       args.fixed_clip_range,
                                                       fixed_mask,
                                                       args.fixed_mask_method,
                                                       args.fixed_mask_threshold,
                                                       args.fixed_cylinder_radius,
                                                       args.fixed_cylinder_center_offset,
                                                       args.apply_fixed_mask)

    print("Preparing to write fixed image pyramid...")
    save_image_pyramid(pyramid, mask_pyramid, affines, fixed_path, fixed_out_path, args.fixed_out_name)

    # Visualize'
    if visualize:
        for i, image in enumerate(pyramid):
            slices = [image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2]
            viz_slices(image, slices[0], save_dir=fixed_out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_0", axis=0)
            viz_slices(image, slices[1], save_dir=fixed_out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_1", axis=1)
            viz_slices(image, slices[2], save_dir=fixed_out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_2", axis=2)

    print("Done")
