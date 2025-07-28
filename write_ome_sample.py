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
    # parser.add_argument("--mask_path", type=str, required=False, default=None, help="Path to the mask image.")
    # parser.add_argument("--name_format", type=str, required=False, help="Format for the output names.")
    # parser.add_argument("--name_prefix", type=str, required=False, help="Prefix for the output files.")
    # parser.add_argument("--name_suffix", type=str, required=False, help="Suffix for the output files.")

    parser.add_argument("--HR_chunks", type=int, nargs=3, default=(160, 160, 160), help="Size of largest high-res ome-zarr chunk (D, H, W).")
    parser.add_argument("--LR_chunks", type=int, nargs=3, default=(160, 160, 160), help="Size of largest low-res ome-zarr chunk (D, H, W).")
    parser.add_argument("--REG_chunks", type=int, nargs=3, default=(80, 80, 80), help="Size of largest registred ome-zarr chunk (D, H, W).")

    #parser.add_argument("--set_slice_count", type=int, default=0, help="Force scans to have a certain set slice count if greater than zero.")
    #parser.add_argument("--print_summary_only", action="store_true", help="If set, only print the summary of image paths without writing to OME-Zarr.")

    parser.add_argument("--split_axis", type=int, default=0, help="Axis along which to split the image data (0 for depth, 1 for height, 2 for width).")
    parser.add_argument("--num_split_sections", type=int, default=1, help="Number of sections to split the image data into along the specified axis.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(args.base_path, args.sample_path)
    if args.HR_paths is not None:
        HR_paths = [os.path.join(path, sample_path) for path in args.HR_paths]
    if args.LR_paths is not None:
        LR_paths = [os.path.join(path, sample_path) for path in args.LR_paths]
    if args.REG_paths is not None:
        REG_paths = [os.path.join(path, sample_path) for path in args.REG_paths]

    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # out_name = os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)
    # if args.mask_path is not None:
    #     mask_path = os.path.join(sample_path, args.mask_path)
    #     print("Mask path: ", mask_path)

    write_ome_datasample(out_path,
                         HR_paths,
                         LR_paths,
                         REG_paths,
                         args.HR_chunks,
                         args.LR_chunks,
                         args.REG_chunks,
                         split_axis=args.split_axis,
                         num_split_sections=args.num_split_sections,
                         compression='lz4'
                         )

    print("Done writing OME-Zarr data sample")

