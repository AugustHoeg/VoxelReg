import os
import re
from typing import List, Dict
import tifffile
import nibabel as nib
import glob
import numpy as np
import zarr
import matplotlib.pyplot as plt

from utils.utils_zarr import write_ome_pyramid
from zarr.storage import DirectoryStore
from skimage.transform import downscale_local_mean

from utils.utils_image import load_image, normalize_std
from utils.utils_plot import viz_slices

def get_dicom_slice_count(directory: str) -> int:
    """
    Counts the number of DICOM (.dcm) files in a directory.
    """
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith(".dcm") and os.path.isfile(os.path.join(directory, f))
    ])

def get_tiff_slice_count(directory: str) -> int:
    """
    Counts the number of tiff files in a directory.
    """
    if len(glob.glob(os.path.join(directory, ".tiff"))) > 1:
        return len([
            f for f in os.listdir(directory)
            if f.lower().endswith(".tiff") and os.path.isfile(os.path.join(directory, f))
        ])
    else:
        img = tifffile.imread(directory)  # If the directory is a single TIFF file, read it to get the number of slices
        return img.shape[0]  # Assuming the first dimension is the number of slices



def get_nifti_slice_count(file_path: str) -> int:
    """
    Returns the number of slices (z-dimension) in a NIfTI file.
    """
    try:
        img = nib.load(file_path)
        shape = img.shape
        if len(shape) < 3:
            return 0
        return shape[2]
    except Exception as e:
        print(f"Warning: Failed to load NIfTI file '{file_path}': {e}")
        return 0


def get_path_and_slices(file_path):

    # Determine extension of file types in directory
    try:
        filename, file_extension = os.path.basename(file_path).split('.', 1)
    except Exception as e:
        # if file_path is a directory, we assume it contains multiple files
        if glob.glob(os.path.join(file_path, "*.dcm")):
            return file_path, get_dicom_slice_count(file_path)

        if glob.glob(os.path.join(file_path, "*.tiff")):
            return file_path, get_tiff_slice_count(file_path)

    # Otherwise, check for extensions
    if file_extension == "nii" or file_extension == "nii.gz":
        return file_path, get_nifti_slice_count(file_path)


def categorize_image_directories(base_dirs, slice_splits) -> Dict[str, List[str]]:
    """
    Walks through a base directory and categorizes DICOM and NIfTI scans
    into bins based on their number of slices.

    Returns a dictionary mapping each bin to a list of scan paths.
    """
    # Define bin labels
    bins = []
    bins.append(f"{slice_splits[0]}")
    for i in range(len(slice_splits) - 1):
        bins.append(f"{slice_splits[i]}_{slice_splits[i+1]}")
    bins.append(f"{slice_splits[-1]}")
    categorized_images = {label: [] for label in bins}

    for dir in base_dirs:
        # DICOM: check if directory contains .dcm files

        scan_path, slice_count = get_path_and_slices(dir)

        if slice_count == 0:
            continue

        # Assign to bin
        if slice_count <= slice_splits[0]:
            categorized_images[bins[0]].append(scan_path)
        elif slice_count > slice_splits[-1]:
            categorized_images[bins[-1]].append(scan_path)
        else:
            for i in range(len(slice_splits) - 1):
                if slice_splits[i] < slice_count <= slice_splits[i+1]:
                    categorized_images[bins[i+1]].append(scan_path)
                    break

    return categorized_images


def write_image_categories(image_categories, name_format, save_dir, chunk_size, pyramid_levels=3, cname='lz4', group_name='HR'):

    for category in image_categories:

        if len(image_categories[category]) == 0:
            continue

        # Crop each scan in each category to the minimum number of slices in that category
        min_slices = int(category.split('_')[0])
        max_slices = int(category.split('_')[-1])

        for image_path in image_categories[category]:
            image_name = re.compile(name_format).search(image_path).group(0)
            image = load_image(image_path, dtype=np.float32)

            print(f"Processing scan {os.path.basename(image_path)} with shape {image.shape}")
            image = image[:, :, :min_slices]

            image = normalize_std(image, standard_deviations=3, mode='rescale')

            slices = [image.shape[2] // 2, image.shape[2] // 3, image.shape[2] // 4]
            viz_slices(image, slice_indices=slices, savefig=False)

            # Convert to C-order
            image = np.ascontiguousarray(image)

            # Create image pyramid using downscale_local_mean
            image_pyramid = [image]
            for i in range(pyramid_levels - 1):
                image_pyramid.append(downscale_local_mean(image_pyramid[i], (2, 2, 2)))

            # Create Zarr store
            zarr_path = os.path.join(save_dir, f"{image_name}.zarr")
            store = DirectoryStore(zarr_path)
            root = zarr.group(store=store)

            # Create image group for the volume
            image_group = root.create_group(group_name)

            write_ome_pyramid(
                image_group=image_group,
                image_pyramid=image_pyramid,
                label_pyramid=None,  # No labels for MRI
                chunk_size=chunk_size,
                cname=cname  # Compression codec
            )

            print(f"Done writing {image_path} to OME-Zarr format at {zarr_path}")

            print("Done")


# Example usage
if __name__ == "__main__":

    chunk_size = (128, 128, 80)

    splits = np.array([160 + i*chunk_size[-1] for i in range(15)])  # Customizable
    #root = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/"

    dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/LIDC_IDRI/train/"
    save_dir = "../../Vedrana_master_project/3D_datasets/datasets/LIDC_IDRI/ome/train/"
    scan_prefix = "*/*/*/"
    name_format = "LIDC-IDRI-...."

    # dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/CTSpine1K/train/"
    # save_dir = "../../Vedrana_master_project/3D_datasets/datasets/CTSpine1K/ome/train/"
    # scan_prefix = "*/*/*/image*"
    # name_format = "...._CT"

    # dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/LITS/train/"
    # save_dir = "../../Vedrana_master_project/3D_datasets/datasets/LITS/ome/train/"
    # scan_prefix = "volume*"
    # name_format = "volume-..."

    base_dirs = glob.glob(os.path.join(dataset_dir, scan_prefix))
    image_categories = categorize_image_directories(base_dirs, splits)

    # Print summary
    for category, paths in image_categories.items():
        print(f"{category}: {len(paths)} scans")

    # Remove first category
    del image_categories[str(splits[0])]

    write_image_categories(image_categories, name_format, save_dir, chunk_size, pyramid_levels=3, cname='lz4', group_name='HR')

    print("Done")
