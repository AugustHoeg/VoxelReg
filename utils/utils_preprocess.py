import os
import numpy as np
import torch
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_otsu

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_tiff import load_tiff, write_tiff, center_crop
from utils.utils_nifti import write_nifti



def crop_to_roi(image, roi_factor, margin_percent=0.50, divis_factor=2, minimum_size=(2000, 2000, 2000), maximum_size=(2000, 2000, 2000)):

    # Define roi
    roi = np.array(image.shape)
    roi[1:] = ((roi[1:] // roi_factor) * (1 + margin_percent)).astype(int)  # Reduce size
    roi = (roi // divis_factor * divis_factor).astype(int)  # Ensure shape is divisible by d

    minimum_size = [np.minimum(val1, val2) for val1, val2 in zip(roi, minimum_size)]

    if minimum_size is not None:
        roi = np.maximum(roi, minimum_size)  # Ensure minimum size
    if maximum_size is not None:
        roi = np.minimum(roi, maximum_size)  # Ensure maximum size

    # Center crop to roi
    crop_image = center_crop(image, roi)
    print("Final crop shape: ", crop_image.shape)

    return crop_image


def preprocess(scan_path, out_name, f, margin_percent, divis_factor, min_size, max_size, save_downscaled=False, mask_threshold=None):

    filename, file_extension = os.path.splitext(os.path.basename(scan_path))
    if out_name is None:
        out_name = filename

    if file_extension == ".tiff" or file_extension == ".tif":
        image = load_tiff(scan_path)
    else:
        assert False, "Unsupported file format."

    image = crop_to_roi(image, roi_factor=f, margin_percent=margin_percent, divis_factor=divis_factor,
                         minimum_size=min_size, maximum_size=max_size)  # Reduce size

    # convert to float
    image = image.astype(np.float32)

    # Normalize to [0, 1]
    image_min = np.min(image)
    image_max = np.max(image)
    image -= image_min
    image /= (image_max - image_min)

    out_base_path = os.path.join(os.path.dirname(scan_path), "processed")
    os.makedirs(out_base_path, exist_ok=True)

    down = None
    if save_downscaled:
        down = downscale_local_mean(image, (2, 2, 2)).astype(np.float32)
        down = downscale_local_mean(down, (2, 2, 2)).astype(np.float32)

        # Ensure range is between [0, 1]
        down = (down - np.min(down)) / (np.max(down) - np.min(down))

        # Save downscaled images
        np.save(os.path.join(out_base_path, out_name + "_down_4.npy"), down)
        #write_tiff(down, os.path.join(sample_path, filename + "_down_4.tiff"))
        write_nifti(down, os.path.join(out_base_path, out_name + "_down_4.nii.gz"))  # for itk-snap

    if mask_threshold is not None:
        mask_image = down if save_downscaled else image
        if mask_threshold == "otsu":
            mask_threshold = threshold_otsu(mask_image)
            print("Otsu threshold: ", mask_threshold)
        else:
            mask_threshold = float(mask_threshold)
            print("Custom threshold: ", mask_threshold)
        mask = np.zeros_like(mask_image)
        mask[mask_image > mask_threshold] = 1
        mask = mask.astype(np.uint8)
        # Save mask
        np.save(os.path.join(out_base_path, out_name + "_mask.npy"), mask)
        #write_tiff(mask, os.path.join(sample_path, filename + "_mask.tiff"))
        write_nifti(mask, os.path.join(out_base_path, out_name + "_mask.nii.gz"))

    np.save(os.path.join(out_base_path, out_name + ".npy"), image)
    #write_tiff(image, os.path.join(sample_path, "processed", filename + ".tiff"))
    write_nifti(image, os.path.join(out_base_path, out_name + ".nii.gz"))  # for itk-snap

    return image, down