import os
import glob
import argparse
from utils.utils_path import write_image_categories, categorize_image_directories, get_orient_transform

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply mask to volume and save")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset directory.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to write the output files.")
    parser.add_argument("--scan_prefix", type=str, required=False, help="Prefix for the scan files.")
    parser.add_argument("--scan_suffix", type=str, required=False, help="Suffix for the scan files.")
    parser.add_argument("--name_format", type=str, required=False, help="Format for the output names.")
    parser.add_argument("--slice_axis", type=int, default=0, help="Axis along which to count volume slices (0, 1, or 2).")
    parser.add_argument("--chunk_size", type=int, nargs=3, default=(160, 160, 160), help="Size of each ome-zarr chunk (D, H, W).")
    parser.add_argument("--slice_splits", type=int, nargs='*', default=None, help="List of splits to categorize scans into based on number of slices. If None, all scans will be in one category.")
    parser.add_argument("--slice_shape", type=int, nargs=2, default=(160, 160), help="Shape of each slice (H, W).")
    parser.add_argument("--set_slice_count", type=int, default=None, help="Force scans to have a certain set slice count.")
    parser.add_argument("--orient_axcodes", type=str, default="RAS", help="Transformation axes codes (e.g., 'RAS', 'LPS').")
    parser.add_argument("--orient_transpose_axes", type=int, nargs=3, default=(0, 1, 2), help="Transpose axes for orientation (e.g., (0, 1, 2)).")
    parser.add_argument("--pyramid_levels", type=int, default=3, help="Number of ome pyramid levels to create.")

    parser.add_argument("--print_summary_only", default=False, help="If set, only print the summary of image categories without writing to OME-Zarr.")

    args = parser.parse_args()
    return args

def mask_volume(image, mask):
    image[mask == 0] = 0  # Set values outside mask to zero

if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/"
    elif args.run_type == "DTU_HPC":
        project_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/"

    # Assign paths
    if args.dataset_path is not None:
        dataset_path = os.path.join(project_path, args.dataset_path)
        print("dataset path: ", dataset_path)
    if args.out_path is not None:
        out_path = os.path.join(project_path, args.out_name)
        print("Output name: ", out_path)

    base_dirs = glob.glob(os.path.join(dataset_path, args.scan_prefix))
    image_categories = categorize_image_directories(base_dirs, args.slice_splits, args.slice_axis)

    # Print summary
    for category, paths in image_categories.items():
        print(f"{category}: {len(paths)} scans")

    # Remove first category
    del image_categories[str(args.splits[0])]

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
                           args.scan_prefix,
                           args.scan_suffix,
                           out_path,
                           args.chunk_size,
                           pyramid_levels=args.pyramid_levels,
                           cname='lz4',
                           group_name='HR')

    print(f"Done writing OME-Zarr data samples to {out_path}")