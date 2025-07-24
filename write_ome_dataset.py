import os
import numpy as np
import glob
import argparse
from utils.utils_path import write_image_categories, categorize_image_directories, get_orient_transform

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Write OME-Zarr dataset from categorized image directories.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the base directory. Other paths will be relative to this path.")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset directory.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to write the output files.")
    parser.add_argument("--scan_prefix", type=str, required=False, help="Prefix for the scan files.")
    parser.add_argument("--name_format", type=str, required=False, help="Format for the output names.")
    parser.add_argument("--name_prefix", type=str, required=False, help="Prefix for the output files.")
    parser.add_argument("--name_suffix", type=str, required=False, help="Suffix for the output files.")
    parser.add_argument("--slice_axis", type=int, default=0, help="Axis along which to count volume slices (0, 1, or 2).")
    parser.add_argument("--chunk_size", type=int, nargs=3, default=(160, 160, 160), help="Size of each ome-zarr chunk (D, H, W).")
    parser.add_argument("--slice_splits", type=int, nargs='*', default=None, help="List of splits to categorize scans into based on number of slices. If None, all scans will be in one category.")
    parser.add_argument("--slice_shape", type=int, nargs=2, default=(160, 160), help="Shape of each slice (H, W).")
    parser.add_argument("--set_slice_count", type=int, default=0, help="Force scans to have a certain set slice count if greater than zero.")
    parser.add_argument("--orient_axcodes", type=str, default="RAS", help="Transformation axes codes (e.g., 'RAS', 'LPS').")
    parser.add_argument("--orient_transpose_axes", type=int, nargs=3, default=(0, 1, 2), help="Transpose axes for orientation (e.g., (0, 1, 2)).")
    parser.add_argument("--pyramid_levels", type=int, default=3, help="Number of ome pyramid levels to create.")

    parser.add_argument("--remove_first_category", action="store_true", help="If set, remove the first category from the image categories.")
    parser.add_argument("--print_summary_only", action="store_true", help="If set, only print the summary of image categories without writing to OME-Zarr.")


    args = parser.parse_args()
    return args

def mask_volume(image, mask):
    image[mask == 0] = 0  # Set values outside mask to zero

if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    if args.dataset_path is not None:
        dataset_path = os.path.join(args.base_path, args.dataset_path)
        print("dataset path: ", dataset_path)
    if args.out_path is not None:
        out_path = os.path.join(args.base_path, args.out_name)
        print("Output name: ", out_path)

    slice_splits = np.array(args.slice_splits) if args.slice_splits is not None else None

    base_dirs = glob.glob(os.path.join(dataset_path, args.scan_prefix))
    if len(base_dirs) == 0:
        raise ValueError(f"No directories found matching prefix '{args.scan_prefix}' in '{dataset_path}'. Please check the path and prefix.")

    image_categories = categorize_image_directories(base_dirs, slice_splits, args.slice_axis)

    # Print summary
    for i, (category, paths) in enumerate(image_categories.items()):
        if i == 0:
            print(f"1_{category}: {len(paths)} scans")
        elif i == len(image_categories) - 1:
            print(f"{category}_inf: {len(paths)} scans")
        else:
            print(f"{category}: {len(paths)} scans")

    # Remove first category
    if args.remove_first_category:
        print(f"Removing first category: {slice_splits[0]}")
        del image_categories[str(slice_splits[0])]

    if args.print_summary_only:
        print("Summary printed. Exiting without writing OME-Zarr data samples.")
        exit(0)

    # Define orientation transformation
    orient_transform = get_orient_transform(axcodes=args.orient_axcodes, transpose_indices=args.orient_transpose_axes)

    write_image_categories(image_categories,
                           args.slice_shape,
                           orient_transform,
                           args.set_slice_count,
                           args.name_format,
                           args.name_prefix,
                           args.name_suffix,
                           out_path,
                           args.chunk_size,
                           pyramid_levels=args.pyramid_levels,
                           cname='lz4',
                           group_name='HR')

    print(f"Done writing OME-Zarr data samples to {out_path}")