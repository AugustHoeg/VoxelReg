import os
import numpy as np
from datetime import datetime
import glob
import argparse
from utils.utils_path import write_image_categories, categorize_image_directories, get_orient_transform
from utils.utils_plot import viz_orthogonal_slices
from utils.utils_image import load_image
from utils.utils_zarr import write_ome_datasample

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the base directory. Other paths will be relative to this path.")
    parser.add_argument("--sample_path", type=str, required=True, help="Path to the sample directory relative to the base path.")
    parser.add_argument("--HR_paths", type=str, nargs='*', required=False, help="Path to the sample directory.")
    parser.add_argument("--LR_paths", type=str, nargs='*', required=False, help="Path to fixed image.")
    parser.add_argument("--REG_paths", type=str, nargs='*', required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")

    parser.add_argument("--HR_chunks", type=int, nargs=3, default=(160, 160, 160), help="Size of largest high-res ome-zarr chunk (D, H, W).")
    parser.add_argument("--LR_chunks", type=int, nargs=3, default=(160, 160, 160), help="Size of largest low-res ome-zarr chunk (D, H, W).")
    parser.add_argument("--REG_chunks", type=int, nargs=3, default=(80, 80, 80), help="Size of largest registred ome-zarr chunk (D, H, W).")

    parser.add_argument("--HR_split_indices", type=int, nargs='*', default=(), help="Indices along the split axis for high-res images.")
    parser.add_argument("--LR_split_indices", type=int, nargs='*', default=(), help="Indices along the split axis for low-res images.")
    parser.add_argument("--REG_split_indices", type=int, nargs='*', default=(), help="Indices along the split axis for registered images.")
    parser.add_argument("--split_axis", type=int, default=0, help="Axis along which to split the image data (0 for depth, 1 for height, 2 for width).")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(args.base_path, args.sample_path)
    if args.HR_paths is not None:
        HR_paths = [os.path.join(sample_path, path) for path in args.HR_paths]
        print("HR paths: ", HR_paths)
    if args.LR_paths is not None:
        LR_paths = [os.path.join(sample_path, path) for path in args.LR_paths]
        print("LR paths: ", LR_paths)
    if args.REG_paths is not None:
        REG_paths = [os.path.join(sample_path, path) for path in args.REG_paths]
        print("REG paths: ", REG_paths)

    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # out_name = os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)
    # if args.mask_path is not None:
    #     mask_path = os.path.join(sample_path, args.mask_path)
    #     print("Mask path: ", mask_path)

    print("HR split indices: ", args.HR_split_indices)
    print("LR split indices: ", args.LR_split_indices)
    print("REG split indices: ", args.REG_split_indices)

    write_ome_datasample(out_path,
                         HR_paths,
                         LR_paths,
                         REG_paths,
                         args.HR_chunks,
                         args.LR_chunks,
                         args.REG_chunks,
                         args.HR_split_indices,
                         args.LR_split_indices,
                         args.REG_split_indices,
                         split_axis=args.split_axis,
                         compression='lz4'
                         )

    print("Done writing OME-Zarr data sample")

