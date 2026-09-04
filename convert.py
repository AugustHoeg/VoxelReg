import os
import argparse
import numpy as np
from utils.utils_tiff import write_tiff
from utils.utils_npy import write_npy
from utils.utils_image import load_image
from utils.utils_preprocess import rescale


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert scan.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="path for the output image.")
    parser.add_argument("--out_name", type=str, required=False, default="out", help="Name for the output image.")
    parser.add_argument("--out_format", type=str, default=".tiff", help="data format to save to")
    parser.add_argument("--dtype", type=str, required=True, default=np.uint16, help="input data type")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    if args.scan_path is not None:
        scan_path = os.path.join(args.base_path, args.scan_path)

    out_format = args.out_format

    print(f"Loading {scan_path}")
    image = load_image(scan_path, dtype=args.dtype)

    print(f"Writing {scan_path}")
    out_path = os.path.join(args.base_path, args.out_path, args.out_name + out_format)
    if out_format == ".npy":
        write_npy(image, output_path=out_path, dtype=args.dtype)
    elif out_format == ".tiff" or out_format == ".tif":
        write_tiff(image, output_path=out_path, dtype=args.dtype)
    else:
        raise ValueError(f"Unsupported file format: {out_format}")

