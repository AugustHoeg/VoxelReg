import os
import time
import numpy as np
import argparse
from utils.utils_image import load_image

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split volume and save")
    parser.add_argument("--base_path", type=str, required=False,
                        help="Path to the base directory. Other paths will be relative to this path.")
    parser.add_argument("--scan_path", type=str, required=False,
                        help="Path to the scan directory relative to the base path.")

    args = parser.parse_args()
    return args


# import copy
# image_backup = copy.deepcopy(image)
# from utils.utils_preprocess import plot_histogram, mask_with_cylinder
#
# mask_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_B/fixed_scale_4_mask.nii.gz"
# mask = mask_with_cylinder(image, cylinder_radius=205, cylinder_offset=(0,0)) #load_image(mask_path, dtype=np.float32, nifti_backend="nibabel", return_metadata=True)
#
# from utils.utils_plot import viz_slices, viz_multiple_images, viz_orthogonal_slices
# viz_orthogonal_slices(image, [50, 100, 150, 200, 250], savefig=False)
#
# from utils.utils_preprocess import masked_clip_percentile
# image = masked_clip_percentile(image, mask, lower=35.0, upper=99.0, mode='rescale')
# viz_orthogonal_slices(image, [50, 100, 150, 200, 250], savefig=False)
#
# plot_histogram(image, save_fig=False)
# #image[image < 0.20] = 0
# #viz_orthogonal_slices(image, [50, 100, 150, 200, 250], savefig=False)


if __name__ == "__main__":

    args = parse_arguments()

    if args.scan_path is not None:
        scan_path = os.path.join(args.base_path, args.scan_path)
        print("Scan path: ", scan_path)

    scan_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_B/fixed_scale_4.nii.gz"
    mask_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_B/fixed_scale_4_mask.nii.gz"

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





