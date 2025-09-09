import os
import time
import numpy as np
import argparse
from utils.utils_image import load_image

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split volume and save")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to the base directory. Other paths will be relative to this path.")
    parser.add_argument("--scan_path", type=str, required=True,
                        help="Path to the scan directory relative to the base path.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.sample_path is not None:
        scan_path = os.path.join(args.base_path, args.scan_path)
        print("Scan path: ", scan_path)

    print(f"Loading {scan_path}")

    start = time.time()
    image, metadata = load_image(scan_path, dtype=np.float32, nifti_backend="nibabel", return_metadata=True)
    stop = time.time()
    print("Time elapsed, nibabel:", stop - start)

    start = time.time()
    image, metadata = load_image(scan_path, dtype=np.float32, nifti_backend="antspyx", return_metadata=True)
    stop = time.time()
    print("Time elapsed, antspyx:", stop - start)

    start = time.time()
    image, metadata = load_image(scan_path, dtype=np.float32, nifti_backend="sitk", return_metadata=True)
    stop = time.time()
    print("Time elapsed, sitk:", stop - start)





