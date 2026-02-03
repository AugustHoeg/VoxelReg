import os
import argparse
import numpy as np
from utils.utils_tiff import tiff2zarr

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert polaris BigTIFF files to zarr store.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the sample directory.")
    parser.add_argument("--tiff_path", type=str, required=False, help="Path to tiff image.")
    parser.add_argument("--out_path", type=str, required=False, help="path for the output image.")
    parser.add_argument("--out_dtype", type=str, required=False, default=np.uint16, help="data format to convert to")

    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for parallel processing")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    scan_path = os.path.join(args.base_path, args.tiff_path)
    out_path = os.path.join(args.out_path, args.tiff_path.replace(".tiff", ".zarr"))

    image = tiff2zarr(scan_path,
                      nworkers=args.num_workers,
                      zarr_path=out_path,
                      group_name="raw",
                      dtype=args.out_dtype,
                      cname="lz4",
                      clevel=3,
                      return_as_dask=False)

    print(f"Conversion complete.")

