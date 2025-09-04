import os
import re
from typing import List, Dict
import tifffile
import nibabel as nib
import glob
import numpy as np
import zarr
import matplotlib.pyplot as plt
import monai.transforms as mt

from utils.utils_zarr import write_ome_pyramid
from zarr.storage import LocalStore
from skimage.transform import downscale_local_mean

from utils.utils_image import load_image, plot_histogram, compare_histograms, match_histogram_3d_continuous
from utils.utils_plot import viz_slices
from utils.utils_preprocess import image_crop_pad, clip_percentile


def get_orient_transform(axcodes="RAS", transpose_indices=(0, 1, 2)):

    if axcodes == "" or axcodes is None:
        orient_transform = mt.Identity()
    else:
        orient_transform = mt.Orientation(axcodes=axcodes)

    if transpose_indices is None or len(transpose_indices) != 3:
        transpose_indices = (0, 1, 2)

    transform = mt.Compose([
        orient_transform,
        mt.Transpose(indices=transpose_indices)
    ])

    return transform


def get_dicom_slice_count(directory: str) -> int:
    """
    Counts the number of DICOM (.dcm) files in a directory.
    """
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith(".dcm") and os.path.isfile(os.path.join(directory, f))
    ])


def get_tiff_slice_count(directory: str, axis=0) -> int:
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
        return img.shape[axis]  # Assuming the first dimension is the number of slices


def get_nifti_slice_count(file_path: str, axis=0) -> int:
    """
    Returns the number of slices in a NIfTI file.
    """
    try:
        img = nib.load(file_path)
        shape = img.shape
        if len(shape) < 3:
            return 0
        return shape[axis]
    except Exception as e:
        print(f"Warning: Failed to load NIfTI file '{file_path}': {e}")
        return 0


def get_path_and_slices(file_path, axis=0):

    # Determine extension of file types in directory
    try:
        filename, file_extension = os.path.basename(file_path).split('.', 1)
    except Exception as e:
        # if file_path is a directory, we assume it contains multiple files
        if glob.glob(os.path.join(file_path, "*.dcm")):
            return file_path, get_dicom_slice_count(file_path)

        if glob.glob(os.path.join(file_path, "*.tiff")):
            return file_path, get_tiff_slice_count(file_path, axis)

    # Otherwise, check for extensions
    if file_extension == "nii" or file_extension == "nii.gz":
        return file_path, get_nifti_slice_count(file_path, axis)


def categorize_image_directories(base_dirs, slice_splits=None, axis=0) -> Dict[str, List[str]]:
    """
    Walks through a base directory and categorizes DICOM and NIfTI scans
    into bins based on their number of slices.

    Returns a dictionary mapping each bin to a list of scan paths.
    """
    if slice_splits is None:
        # If no splits are provided, categorize all scans into one bin
        slice_splits = [np.inf]

    # Define bin labels
    bins = []
    bins.append(f"{slice_splits[0] - 1}")
    for i in range(len(slice_splits) - 1):
        bins.append(f"{slice_splits[i]}_{slice_splits[i+1] - 1}")
    bins.append(f"{slice_splits[-1]}")
    categorized_images = {label: [] for label in bins}

    for dir in base_dirs:
        # DICOM: check if directory contains .dcm files

        scan_path, slice_count = get_path_and_slices(dir, axis)

        if slice_count == 0:
            continue

        # Assign to bin
        if slice_count < slice_splits[0]:
            categorized_images[bins[0]].append(scan_path)
        elif slice_count >= slice_splits[-1]:
            categorized_images[bins[-1]].append(scan_path)
        else:
            for i in range(len(slice_splits) - 1):
                if slice_splits[i] <= slice_count < slice_splits[i+1]:
                    categorized_images[bins[i+1]].append(scan_path)
                    break

    return categorized_images


def write_image_categories(image_categories,
                           slice_shape,
                           orient_transform=mt.Identity(),
                           set_slice_count=0,
                           name_format="",
                           name_prefix="",
                           name_suffix="",
                           save_dir=None,
                           chunk_size=None,
                           pyramid_levels=3,
                           cname='lz4',
                           group_name='HR'):

    for category in image_categories:

        if len(image_categories[category]) == 0:
            continue

        # Crop each scan in each category to the minimum number of slices in that category
        if set_slice_count > 0:
            min_slices = set_slice_count
        else:
            min_slices = int(category.split('_')[0])
        max_slices = int(category.split('_')[-1])

        for image_path in image_categories[category]:
            image_name = f"{name_prefix}" + list(re.finditer(name_format, image_path))[-1].group(0) + f"{name_suffix}"
            #image_name = f"{name_prefix}" + re.compile(name_format).search(image_path).group(0) + f"{name_suffix}"
            image = load_image(image_path, dtype=np.float32)

            # Ensure image is oriented with slice dimension first.
            image = orient_transform(image).numpy()

            print(f"Processing scan {os.path.basename(image_path)} with shape {image.shape}")
            if slice_shape is None:
                image = image[:min_slices, :, :]
            else:
                image, start_coords, end_coords = image_crop_pad(image, roi=(min_slices, *slice_shape), top_index='first')
            print(f"After cropping, scan shape is {image.shape}")

            # image = normalize_std(image, standard_deviations=3, mode='rescale')
            # plot_histogram(image)
            image = clip_percentile(image, lower=1.0, upper=99.0, mode='rescale')

            slices = [image.shape[0] // 2, image.shape[0] // 3, image.shape[0] // 4]
            viz_slices(image, slice_indices=slices, savefig=False, title=os.path.join(os.path.dirname(image_path), f"{image_name}_slices"))

            # Convert to C-order
            image = np.ascontiguousarray(image)

            # from utils.utils_nifti import write_nifti
            # write_nifti(image, output_path="test.nii")

            # Create image pyramid using downscale_local_mean
            image_pyramid = [image]
            for i in range(pyramid_levels - 1):
                down = downscale_local_mean(image_pyramid[i], (2, 2, 2))
                matched = match_histogram_3d_continuous(source=down, reference=image)
                image_pyramid.append(matched)

            # Create Zarr store
            zarr_path = os.path.join(save_dir, f"{image_name}.zarr")
            store = LocalStore(zarr_path)
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


def write_ome_dataset(dataset_name):

    name_prefix = ""
    name_suffix = ""
    axis = 0
    orient_transform = mt.Identity()

    if dataset_name == "HCP_1200":
        dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/HCP_1200/train/"
        save_dir = "../../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/train/"
        scan_prefix = "*/T1w/T1w_acpc_dc.nii"
        name_format = r"(\d{6})"  # look for six digit number in image path (regex format)
        chunk_size = (128, 128, 160)
        splits = np.array([260])
        name_prefix = "volume_"
        slice_shape = (256, 320)
        set_slice_count = 256
        axis = 2
        orient_transform = mt.Compose([
            mt.Orientation(axcodes="LPS"),
            mt.Transpose(indices=(2, 0, 1))
        ])  # Transform is axcode "RAS" -> transpose (2,0,1)

    elif dataset_name == "IXI":
        dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/IXI/train/"
        save_dir = "../../Vedrana_master_project/3D_datasets/datasets/IXI/ome/train/"
        scan_prefix = "*T1.nii"
        name_format = r"IXI(\d{3})"  # look for IXI path (regex format)
        chunk_size = (128, 72, 128)
        splits = np.array([256])
        slice_shape = (144, 256)
        set_slice_count = 256  # set number of slices, ignoring slice categories.
        axis = 1
        orient_transform = mt.Compose([
            mt.Orientation(axcodes="LPS"),
            mt.Transpose(indices=(1, 2, 0))
        ])  # Transform is axcode "LPS" -> transpose (1,2,0)

    elif dataset_name == "LITS":
        dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/LITS/train/"
        save_dir = "../../Vedrana_master_project/3D_datasets/datasets/LITS/ome/train/"
        scan_prefix = "volume*"
        name_format = "volume-..."
        slice_shape = (512, 512)
        chunk_size = (80, 128, 128)
        set_slice_count = None
        splits = np.array([80 + i * chunk_size[0] for i in range(15)])  # Customizable slice count splits
        axis = 2
        orient_transform = mt.Compose([
            mt.Orientation(axcodes="RAS"),
            mt.Transpose(indices=(2, 0, 1))
        ])  # Transform is axcode "RAS" -> transpose (2,0,1)

    elif dataset_name == "CTSpine1K":
        dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/CTSpine1K/train/"
        save_dir = "../../Vedrana_master_project/3D_datasets/datasets/CTSpine1K/ome/train/"
        scan_prefix = "*/*/*/image*"
        name_format = "...._CT"
        slice_shape = (512, 512)
        chunk_size = (80, 128, 128)
        set_slice_count = None
        splits = np.array([80 + i * chunk_size[0] for i in range(15)])  # Customizable slice count splits
        axis = 2
        orient_transform = mt.Compose([
            mt.Orientation(axcodes="RAS"),
            mt.Transpose(indices=(2, 0, 1))
        ])  # Transform is axcode "RAS" -> transpose (2,0,1)

    elif dataset_name == "LIDC-IDRI":
        dataset_dir = "../../Vedrana_master_project/3D_datasets/datasets/LIDC_IDRI/train/"
        save_dir = "../../Vedrana_master_project/3D_datasets/datasets/LIDC_IDRI/ome/train/"
        scan_prefix = "*/*/*/"
        name_format = "LIDC-IDRI-...."
        slice_shape = (512, 512)
        chunk_size = (80, 128, 128)
        set_slice_count = None
        splits = np.array([80 + i * chunk_size[0] for i in range(15)])  # Customizable slice count splits
        axis = 2
        orient_transform = mt.Compose([
            mt.Orientation(axcodes="RAS"),
            mt.Transpose(indices=(2, 0, 1))
        ])  # Transform is axcode "RAS" -> transpose (2,0,1)

    else:
        raise NotImplementedError('Dataset %s not implemented.' % dataset_name)

    base_dirs = glob.glob(os.path.join(dataset_dir, scan_prefix))
    image_categories = categorize_image_directories(base_dirs, splits, axis)

    # Print summary
    for i, (category, paths) in enumerate(image_categories.items()):
        if i == 0:
            print(f"1_{category}: {len(paths)} scans")
        elif i == len(image_categories) - 1:
            print(f"{category}_inf: {len(paths)} scans")
        else:
            print(f"{category}: {len(paths)} scans")

    # Remove first category
    # del image_categories[str(splits[0])]

    write_image_categories(image_categories,
                           slice_shape,
                           orient_transform,
                           set_slice_count,
                           name_format,
                           name_prefix,
                           name_suffix,
                           save_dir,
                           chunk_size,
                           pyramid_levels=3,
                           cname='lz4',
                           group_name='HR')

# Example usage
if __name__ == "__main__":

    #root = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/"

    datasets = ["HCP_1200", "IXI", "LITS", "CTSpine1K", "LIDC-IDRI"]  # "HCP_1200", "IXI", "LITS", "CTSpine1K", "LIDC-IDRI"

    for dataset_name in datasets:
        print(f"Writing OME-Zarr dataset for {dataset_name}...")

        write_ome_dataset(dataset_name)

    print("Done")



# import monai.transforms as mt
# import matplotlib.pyplot as plt
# orientation_transform = mt.Orientation(axcodes="RAS")
# image_trans = np.transpose(orientation_transform(image), (2,0,1))
# plt.figure()
# image = load_image(image_path, dtype=np.float32)
# plt.imshow(image_trans[100, :, :])
# plt.show()