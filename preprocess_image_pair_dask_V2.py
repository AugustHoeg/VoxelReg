import os
import argparse
import shutil

import numpy as np
import dask.array as da
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar

from utils.utils_image import plot_histogram
from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_preprocess import get_image_and_affine, save_image_pyramid, get_dtype, mask_with_cylinder, compute_affine_scale
from utils.utils_nifti import voxel2world, set_origin
from utils.utils_dask import threshold_dask, otsu_threshold_dask, crop_pad_vol
from utils.utils_zarr import create_ome_group, write_ome_level

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


project_path = "../../Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Oak_A/"
sample_path = project_path
moving_path = sample_path + "Oak_A_bin1x1_LFOV_retake_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_pos1_Stitch_scale_4.tif"
fixed_path = sample_path + "Oak_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_Stitch_scale_4.tif"
out_name = "Oak_A"  # Name of the output file

project_path = "../../Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/forams_A/"
sample_path = project_path
moving_path = sample_path + "forams_A_LR.zarr/raw"
fixed_path = sample_path + "forams_A_LR.zarr/raw"
out_name = "forams_A"  # Name of the output file


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")

    parser.add_argument("--moving_path", type=str, required=False, help="Path to the scan file.")
    parser.add_argument("--fixed_path", type=str, required=False, help="Path to the scan file.")

    parser.add_argument("--out_path", type=str, required=False, default="", help="Path to the output file.")
    #parser.add_argument("--moving_out_path", type=str, required=False, default="", help="Path to the output file.")
    #parser.add_argument("--fixed_out_path", type=str, required=False, default="", help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, default="output", help="Output name for the processed image.")
    parser.add_argument("--moving_out_name", type=str, required=False, default="moving", help="Output name for the processed image.")
    parser.add_argument("--fixed_out_name", type=str, required=False, default="fixed", help="Output name for the processed image.")

    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--dtype", type=str, default="UINT16", help="Data type of fixed/moving input images")

    parser.add_argument("--moving_d_range", type=int, nargs=2, help="Start and end indices for moving first axis.")
    parser.add_argument("--moving_h_range", type=int, nargs=2, help="Start and end indices for moving second axis.")
    parser.add_argument("--moving_w_range", type=int, nargs=2, help="Start and end indices for moving third axis.")
    parser.add_argument("--fixed_d_range", type=int, nargs=2, help="Start and end indices for fixed first axis.")
    parser.add_argument("--fixed_h_range", type=int, nargs=2, help="Start and end indices for fixed second axis.")
    parser.add_argument("--fixed_w_range", type=int, nargs=2, help="Start and end indices for fixed third axis.")

    parser.add_argument("--moving_pixel_size", type=float, nargs=3, default=(None, None, None), help="Pixel size in mm for moving image.")
    parser.add_argument("--fixed_pixel_size", type=float, nargs=3, default=(None, None, None), help="Pixel size in mm for fixed image.")

    parser.add_argument("--moving_divis_factor", type=int, default=8, help="Divisibility factor for cropping highest resolution moving image.")
    parser.add_argument("--fixed_divis_factor", type=int, default=8, help="Divisibility factor for cropping highest resolution fixed image.")

    parser.add_argument("--moving_pyramid_depth", type=int, default=4, help="Depth of saved image pyramid.")
    parser.add_argument("--fixed_pyramid_depth", type=int, default=4, help="Depth of saved image pyramid.")

    parser.add_argument("--moving_clip_percentiles", type=float, nargs=2, default=(1.0, 99.0), help="Lower and upper percentiles for image normalization")
    parser.add_argument("--fixed_clip_percentiles", type=float, nargs=2, default=(1.0, 99.0), help="Lower and upper percentiles for image normalization")

    parser.add_argument("--moving_clip_range", type=float, nargs=2, default=(0, 65535), help="Lower and upper percentiles for image normalization")
    parser.add_argument("--fixed_clip_range", type=float, nargs=2, default=(0, 65535), help="Lower and upper percentiles for image normalization")

    parser.add_argument("--moving_mask_path", default=None, help="Path to moving mask image, default is None.")
    parser.add_argument("--moving_mask_method", default=None, help="Method for creating moving mask. Currently supports 'threshold' and 'cylinder'. Default is None, which skips mask creation.")
    parser.add_argument("--moving_mask_threshold", default=None, type=int, help="Threshold for binary mask image. If unspecified, otsu thresholding will be used. default is None.")
    parser.add_argument("--moving_cylinder_radius", type=int, default=None, help="Radius of the cylinder for moving mask in voxels.")
    parser.add_argument("--moving_cylinder_center_offset", type=int, nargs=2, default=(0, 0), help="Offset for the center of the cylinder mask in voxels, default is 0 (centered in H, W).")
    parser.add_argument("--apply_moving_mask", action="store_true", help="Apply moving mask to the image.")

    parser.add_argument("--fixed_mask_path", default=None, help="Path to fixed mask image, default is None.")
    parser.add_argument("--fixed_mask_method", default=None, help="Method for creating fixed mask. Currently supports 'threshold' and 'cylinder'. Default is None, which skips mask creation.")
    parser.add_argument("--fixed_mask_threshold", default=None, type=int, help="Threshold for binary mask image, default is None.")
    parser.add_argument("--fixed_cylinder_radius", type=int, default=None, help="Radius of the cylinder for fixed mask in voxels.")
    parser.add_argument("--fixed_cylinder_center_offset", type=int, nargs=2, default=(0, 0), help="Offset for the center of the cylinder mask in voxels, default is 0 (centered in H, W).")
    parser.add_argument("--apply_fixed_mask", action="store_true", help="Apply fixed mask to the image.")

    parser.add_argument("--top_index", type=str, default="last", help="Index for the top slice of the image, default is 'last'")
    parser.add_argument("--write_nifti", action="store_true", help="Write preprocessed images to nifti files for registration (note: very slow).")

    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for Dask cluster.")
    parser.add_argument("--memory_limit", type=str, default="12GB", help="Memory limit for each Dask worker.")

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
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)

    visualize = False  # True
    print("Visualization is set to: ", visualize)

    ##################### DASK CLUSTER ######################
    if args.run_type == "DTU_HPC":
        # Set up Dask cluster for DTU HPC
        cluster = LocalCluster(args.num_workers,
                               threads_per_worker=1,
                               memory_limit=args.memory_limit)
        client = Client(cluster)

    ##################### MOVING IMAGE ######################

    # Load moving image
    # args.moving_pixel_size = (4, 4, 4)  # REMOVE THIS
    # args.moving_divis_factor = 160  # REMOVE THIS
    # args.moving_mask_method = None # 'threshold' # REMOVE THIS
    # args.moving_mask_threshold = 'otsu' # REMOVE THIS  #TODO ADD THRESHOLD EXPLORER!
    # # args.apply_fixed_mask = True # REMOVE THIS

    input_dtype = get_dtype(args.dtype)
    moving, moving_affine = get_image_and_affine(moving_path,
                                                 custom_origin=(0, 0, 0),
                                                 pixel_size_mm=args.moving_pixel_size,
                                                 dtype=input_dtype,
                                                 backend="Dask")

    moving = crop_pad_vol(moving, args.moving_d_range, args.moving_h_range, args.moving_w_range, pad_value=0)

    # Scale moving affines
    moving_affines = [moving_affine]
    for depth in range(0, args.moving_pyramid_depth - 1):
        affine = compute_affine_scale(moving_affines[depth], scale=2)
        moving_affines.append(affine)

    # rechunk moving
    moving = moving.rechunk((160, 160, 160))

    # Define moving path
    ome_path = os.path.join(out_path, args.out_name + "_ome.zarr")

    # Get moving image mask
    moving_mask = None
    mask_pyramid = None
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

    elif args.moving_mask_method == "threshold":
        moving_threshold = args.moving_mask_threshold
        print(f"Creating moving mask with threshold = {moving_threshold}")
        if moving_threshold == "otsu":
            moving_threshold = otsu_threshold_dask(moving, bins=65535, value_range=(0, 65535), remove_zero_bin=False)
        moving_mask = threshold_dask(moving, threshold=moving_threshold, high=255, low=0, dtype=np.uint8)
        #moving_mask = mask_with_threshold(moving, mask_threshold=args.moving_mask_threshold)

    elif args.moving_mask_method == "cylinder":
        print(f"Creating moving mask using method: {args.moving_mask_method}")
        moving_mask = mask_with_cylinder(moving, cylinder_radius=args.moving_cylinder_radius, cylinder_offset=args.moving_cylinder_center_offset)
    else:
        print("No moving mask will be used.")

    # Write mask only if it exists
    if moving_mask is not None:
        moving_mask_affines = moving_affines

        # rechunk mask
        moving_mask = moving_mask.rechunk((160, 160, 160))

        # Write moving mask
        group_name = "LR_mask"
        store, group_tmp = create_ome_group(ome_path, group_name=group_name, pyramid_depth=args.moving_pyramid_depth)

        mask_pyramid = [moving_mask]
        for level in range(args.moving_pyramid_depth):

            mask_pyramid[level] = write_ome_level(mask_pyramid[level], store, group_name, level=level, cname='lz4')

            if level < args.moving_pyramid_depth - 1:
                down = da.coarsen(np.mean, mask_pyramid[level], {0: 2, 1: 2, 2: 2}, trim_excess=True).astype(np.uint8)
                down = da.where(down < 255, 0, 255)  # ensure mask is binary
                down = down.rechunk((160, 160, 160))

                mask_pyramid.append(down)

        # from ome_zarr.writer import write_label_metadata
        # write_label_metadata(group, 'LR')

        viz_slices(moving_mask, [200, 400, 600], save_dir=out_path, title=args.out_name + f"_moving_mask", axis=0, vmin=0, vmax=255)

        # moving = da.where(moving_mask.astype(bool), moving, da.nan)
        moving = da.where(moving_mask.astype(bool), moving, 0)  # Avoids promotion to float due to nan

    hist, bins = da.histogram(moving, bins=65535, range=(0, 65535))
    with ProgressBar(dt=1):
        print("Computing histogram for percentile clipping...")
        hist = hist.compute()  # to numpy
        hist, bins = hist[1:], bins[1:]  # remove zero bin
        cdf = np.cumsum(hist) / np.sum(hist)
        lower, upper = args.moving_clip_percentiles
        low = int(np.searchsorted(cdf, lower / 100))
        high = int(np.searchsorted(cdf, upper / 100))
        plot_histogram(hist, bins, low=low, high=high, save_dir=out_path, title=args.out_name + f"_moving_histogram")
    print(f"Percentile clipping values: low = {low}, high = {high}")

    # scale and clip
    moving = da.clip(moving, low, high, dtype=np.float32)  # clip, and promote to float for scaling
    moving = (moving - low) / (high - low)
    moving = (moving * 65535).astype(input_dtype)
    moving = moving.persist()  # keep in memory/dask cluster

    if moving_mask is not None:
        moving = da.where(moving_mask.astype(bool), moving, 0)

    moving = moving.rechunk((160, 160, 160))

    # Write moving image ome-zarr level 0
    group_name = "LR"
    store, group = create_ome_group(ome_path, group_name=group_name, pyramid_depth=args.moving_pyramid_depth)

    moving_pyramid = [moving]
    for level in range(args.moving_pyramid_depth):

        moving_pyramid[level] = write_ome_level(moving_pyramid[level], store, group_name, level=level, cname='lz4')

        if level < args.moving_pyramid_depth - 1:
            down = da.coarsen(np.mean, moving_pyramid[level], {0: 2, 1: 2, 2: 2}, trim_excess=True).astype(input_dtype)
            down = down.rechunk((160, 160, 160))

            # apply mask
            if mask_pyramid is not None:
                down = da.where(mask_pyramid[level + 1].astype(bool), down, 0)
            moving_pyramid.append(down)


    for i, image in enumerate(moving_pyramid):
        print(f"Level {i} shape: {image.shape}, chunks: {image.chunksize}")
        slices = [image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2]
        viz_slices(image, slices[0], save_dir=out_path, title=args.out_name + f"_scale_{2 ** i}_pre_axis_0", axis=0, vmin=0, vmax=65535)
        viz_slices(image, slices[1], save_dir=out_path, title=args.out_name + f"_scale_{2 ** i}_pre_axis_1", axis=1, vmin=0, vmax=65535)
        viz_slices(image, slices[2], save_dir=out_path, title=args.out_name + f"_scale_{2 ** i}_pre_axis_2", axis=2, vmin=0, vmax=65535)

    # Save to nifti for registration here
    if args.write_nifti:
        print("Preparing to write moving image pyramid...")
        save_image_pyramid(moving_pyramid, mask_pyramid, moving_affines, moving_path, out_path, args.moving_out_name, start_level=0)

    # Record moving image top center position
    p1 = [moving.shape[0], moving.shape[1] / 2, moving.shape[2] / 2]

    ##################### FIXED IMAGE ######################

    # Load moving image
    # args.fixed_pixel_size = (1, 1, 1)  # REMOVE THIS
    # args.fixed_divis_factor = 160  # REMOVE THIS
    # args.fixed_mask_method = None  # 'threshold'  # REMOVE THIS
    # args.fixed_mask_threshold = 0  # REMOVE THIS #TODO ADD THRESHOLD EXPLORER!
    # # args.apply_moving_mask = True  # REMOVE THIS

    fixed, fixed_affine = get_image_and_affine(fixed_path,
                                               custom_origin=(0, 0, 0),
                                               pixel_size_mm=args.fixed_pixel_size,
                                               dtype=input_dtype,
                                               backend="Dask")

    fixed = crop_pad_vol(fixed, args.fixed_d_range, args.fixed_h_range, args.fixed_w_range, pad_value=0)

    # Set fixed origin to moving image top slice, centered in H, W
    p2 = [fixed.shape[0], fixed.shape[1] / 2, fixed.shape[2] / 2]
    pos_diff = voxel2world(moving_affine, p1) - voxel2world(fixed_affine, p2)
    set_origin(fixed_affine, new_origin=pos_diff)
    print("nifti affine after set pos\n", fixed_affine)

    # Scale fixed affines
    fixed_affines = [fixed_affine]
    for depth in range(0, args.fixed_pyramid_depth - 1):
        affine = compute_affine_scale(fixed_affines[depth], scale=2)
        fixed_affines.append(affine)

    # rechunk fixed
    fixed = fixed.rechunk((160, 160, 160))

    # Get fixed image mask
    fixed_mask = None
    mask_pyramid = None
    if args.fixed_mask_path is not None:
        print(f"Using fixed mask from: {fixed_mask_path}")
        fixed_mask, fixed_mask_affine = get_image_and_affine(fixed_mask_path,
                                                             custom_origin=(0, 0, 0),
                                                             pixel_size_mm=args.fixed_pixel_size,
                                                             dtype=np.uint8,
                                                             backend="Dask")

        # fixed_mask, _, _, _ = define_image_space(fixed_mask, fixed_mask_affine, f=1,
        #                                          min_size=args.fixed_min_size,
        #                                          max_size=args.fixed_max_size,
        #                                          margin_percent=0.0,
        #                                          divis_factor=args.fixed_divis_factor,
        #                                          top_index=args.top_index)

    elif args.fixed_mask_method == "threshold":
        fixed_threshold = args.fixed_mask_threshold
        print(f"Creating fixed mask with threshold = {fixed_threshold}")
        if fixed_threshold == "otsu":
            fixed_threshold = otsu_threshold_dask(fixed, bins=65535, value_range=(0, 65535), remove_zero_bin=False)
        fixed_mask = threshold_dask(fixed, threshold=fixed_threshold, high=255, low=0, dtype=np.uint8)

    elif args.fixed_mask_method == "cylinder":
        print(f"Creating fixed mask using method: {args.fixed_mask_method}")
        fixed_mask = mask_with_cylinder(fixed, cylinder_radius=args.fixed_cylinder_radius, cylinder_offset=args.fixed_cylinder_center_offset)
    else:
        print("No fixed mask will be used.")

    fixed_mask_affines = fixed_affines

    # Write mask only if it exists
    if fixed_mask is not None:

        # rechunk mask
        fixed_mask = fixed_mask.rechunk((160, 160, 160))

        # Write fixed mask
        group_name = "HR_mask"
        # fixed_ome_path = os.path.join(fixed_out_path, args.fixed_out_name + "_ome.zarr")
        store, group_tmp = create_ome_group(ome_path, group_name=group_name, pyramid_depth=args.fixed_pyramid_depth)

        mask_pyramid = [fixed_mask]
        for level in range(args.fixed_pyramid_depth):

            mask_pyramid[level] = write_ome_level(mask_pyramid[level], store, group_name, level=level, cname='lz4')

            if level < args.fixed_pyramid_depth - 1:
                down = da.coarsen(np.mean, mask_pyramid[level], {0: 2, 1: 2, 2: 2}, trim_excess=True).astype(np.uint8)
                down = da.where(down < 255, 0, 255)  # ensure mask is binary
                down = down.rechunk((160, 160, 160))

                mask_pyramid.append(down)

        # from ome_zarr.writer import write_label_metadata
        # write_label_metadata(group, 'LR')

        viz_slices(fixed_mask, [200, 400, 600], save_dir=out_path, title=args.out_name + f"_fixed_mask", axis=0, vmin=0, vmax=255)
        print(f"min = {fixed_mask[100, :, :].min().compute()}, max = {fixed_mask[100, :, :].max().compute()}")

        fixed = da.where(fixed_mask.astype(bool), fixed, 0)  # Avoids promotion to float due to nan

    hist, bins = da.histogram(fixed, bins=65535, range=(0, 65535))
    with ProgressBar(dt=1):
        print("Computing histogram for percentile clipping...")
        hist = hist.compute()  # to numpy
        hist, bins = hist[1:], bins[1:]  # remove zero bin
        cdf = np.cumsum(hist) / np.sum(hist)
        lower, upper = args.fixed_clip_percentiles
        low = int(np.searchsorted(cdf, lower / 100))
        high = int(np.searchsorted(cdf, upper / 100))
        plot_histogram(hist, bins, low=low, high=high, save_dir=out_path, title=args.out_name + f"_fixed_histogram")
    print(f"Percentile clipping values: low = {low}, high = {high}")

    # scale and clip
    fixed = da.clip(fixed, low, high, dtype=np.float32)  # clip, and promote to float for scaling
    fixed = (fixed - low) / (high - low)
    fixed = (fixed * 65535).astype(input_dtype)
    fixed = fixed.persist()  # persist in memory

    if fixed_mask is not None:
        fixed = da.where(fixed_mask.astype(bool), fixed, 0)

    fixed = fixed.rechunk((160, 160, 160))

    # Write fixed image ome-zarr level 0
    group_name = "HR"
    # fixed_ome_path = os.path.join(fixed_out_path, args.fixed_out_name + "_ome.zarr")
    store, group = create_ome_group(ome_path, group_name=group_name, pyramid_depth=args.fixed_pyramid_depth)

    fixed_pyramid = [fixed]
    for level in range(args.fixed_pyramid_depth):

        fixed_pyramid[level] = write_ome_level(fixed_pyramid[level], store, group_name, level=level, cname='lz4')

        if level < args.fixed_pyramid_depth - 1:
            down = da.coarsen(np.mean, fixed_pyramid[level], {0: 2, 1: 2, 2: 2}, trim_excess=True).astype(input_dtype)
            down = down.rechunk((160, 160, 160))

            # apply mask
            if mask_pyramid is not None:
                down = da.where(mask_pyramid[level + 1].astype(bool), down, 0)
            fixed_pyramid.append(down)

    for i, image in enumerate(fixed_pyramid):
        print(f"Level {i} shape: {image.shape}, chunks: {image.chunksize}")
        slices = [image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2]
        viz_slices(image, slices[0], save_dir=out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_0", axis=0, vmin=0, vmax=65535)
        viz_slices(image, slices[1], save_dir=out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_1", axis=1, vmin=0, vmax=65535)
        viz_slices(image, slices[2], save_dir=out_path, title=args.fixed_out_name + f"_scale_{2 ** i}_pre_axis_2", axis=2, vmin=0, vmax=65535)

    # Save to nifti for registration here
    if args.write_nifti:
        print("Preparing to write fixed image pyramid...")
        save_image_pyramid(fixed_pyramid, mask_pyramid, fixed_affines, fixed_path, out_path, args.fixed_out_name, start_level=2)

    print("Done")
