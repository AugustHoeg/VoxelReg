import os
import argparse
import numpy as np
from skimage import filters, morphology, measure, segmentation, transform
from scipy import ndimage
from utils.utils_preprocess import rescale, plot_histogram
from utils.utils_image import load_image, normalize_std
from utils.utils_plot import viz_slices, viz_multiple_images, viz_orthogonal_slices
from utils.utils_nifti import write_nifti
from utils.utils_label import threshold_image
from utils.utils_npy import write_npy
from utils.utils_tiff import write_tiff

# def make_diamond_3d(radius):
#     size = 2 * radius + 1
#     Z, Y, X = np.meshgrid(
#         np.arange(-radius, radius + 1),
#         np.arange(-radius, radius + 1),
#         np.arange(-radius, radius + 1),
#         indexing='ij'
#     )
#     manhattan_distance = np.abs(X) + np.abs(Y) + np.abs(Z)
#     return manhattan_distance <= radius
#
#
# def create_mask(img, closing_ite=10, dilation_ite=3):
#     """
#     Create a binary mask from the input image using morphological operations.
#
#     :param image: Input 3D image array.
#     :param structure: Structuring element for morphological operations.
#     :param iterations: Number of iterations for morphological closing.
#     :return: Binary mask of the input image.
#     """
#
#     # borderpad
#     padding = 30
#     img = np.pad(img, ((padding, padding), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
#
#     # gaussian smoothing
#     img = filters.gaussian(img, sigma=0.8, preserve_range=True)
#     viz_slices(img, slices, savefig=False)
#
#     # Histogram
#     counts, bins = np.histogram(img, bins=512, density=True)
#     plt.figure(figsize=(10, 5))
#     plt.stairs(counts, bins)
#     plt.title('Histogram of Image Intensities')
#     plt.xlabel('Intensity Value')
#     plt.ylabel('Frequency')
#     plt.grid()
#     # plt.xlim([0.1, 0.4])
#
#     peaks, _ = find_peaks(counts, distance=2, threshold=3)
#     minima, _ = find_peaks(-counts, distance=2, threshold=3)
#
#     plt.plot(bins[peaks], counts[peaks], "ro", label="Peaks")
#     plt.plot(bins[minima], counts[minima], "go", label="Minima")
#     plt.show()
#
#     # import qim3d
#     # qim3d.viz.threshold(img, cmap_image='magma', vmin=None, vmax=None)
#
#     # Apply Otsu threshold
#     thresh = 0.22  # filters.threshold_otsu(img)
#     img[img < thresh] = 0
#     viz_slices(img, slices, savefig=False)
#
#     # structure = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
#     #                       [0, 1, 0, 1, 1, 1, 0, 1, 0],
#     #                       [0, 0, 0, 0, 1, 0, 0, 0, 0]]).reshape((3,3,3))
#
#     structure = make_diamond_3d(radius=1)
#     # structure = np.ones((2, 2, 2))
#
#     # Apply binary closing to fill holes
#     img = ndimage.binary_closing(img, structure=structure, iterations=closing_ite)
#     viz_slices(img, slices, savefig=False)
#
#     # Dilate a little, since its better to have mask a bit larger than the actual vertebra
#     img = ndimage.binary_dilation(img, structure=structure, iterations=dilation_ite)
#     viz_slices(img, slices, savefig=False)
#
#     # slight smoothing and binarization again
#     img = filters.gaussian(img, sigma=1.5, preserve_range=True)
#     img[img < 0.5] = 0
#     img[img >= 0.5] = 1.0
#     viz_slices(img, slices, savefig=False)
#
#     # Remove padding again
#     img = img[padding:-padding, padding:-padding, padding:-padding]
#     viz_slices(img, slices, savefig=False)
#
#     # Label connected regions
#     labels = measure.label(img)
#
#     # Keep only the largest connected component (assumed to be vertebra)
#     regions = measure.regionprops(labels)
#     if regions:
#         largest_region = max(regions, key=lambda r: r.area)
#         mask = labels == largest_region.label
#     else:
#         mask = np.zeros_like(labels, dtype=bool)
#
#     viz_slices(mask, slices, savefig=False)
#
#     return mask


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="create mask.")
    parser.add_argument("--base_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="path for the output image.")
    parser.add_argument("--out_name", type=str, required=False, default="out", help="Name for the output image.")
    parser.add_argument("--out_format", type=str, default=".tif", help="data format to save to")
    parser.add_argument("--out_dataformat", type=str, required=False, default="UINT8", help="data format to convert to")
    parser.add_argument("--gaussian_sigma_img", type=float, required=False, default=5)
    parser.add_argument("--gaussian_sigma_mask", type=float, required=False, default=3.5)
    parser.add_argument("--threshold_value", type=float, required=False, default=0.14)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    # Assign paths
    if args.scan_path is not None:
        scan_path = os.path.join(args.base_path, args.scan_path)

    out_format = args.out_format

    # Params
    gaussian_sigma_img = args.gaussian_sigma_img  # 5 for A, 5 for B, 5 for C
    gaussian_sigma_mask = args.gaussian_sigma_mask  # 2.0 for A, 3.0 for B, 3.0 for C, 3.5 for D
    threshold_value = args.threshold_value  # 0.120  # 0.160 for A, 0.120 for B, 0.082 for C, 0.14 for D
    struct_fill_holes = (5, 5, 5)
    erosion_structure = (3, 1)  # (dimensions, connectivity)
    erosion_iterations = 1
    scale_factor = 4  # nearest-neighbor

    # scan_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_D/Vertebrae_D_80kV_registered.nii.gz"
    filename, file_extension = os.path.basename(scan_path).split('.', 1)

    print(f"Loading {scan_path}")
    if file_extension == ".nii" or file_extension == ".nii.gz":
        img, metadata = load_image(scan_path, dtype=np.float32, as_contiguous=True, return_metadata=True)
    else:
        img = load_image(scan_path, dtype=np.float32)

    # Rescale
    img = rescale(img)

    # Gaussian filter
    mask = filters.gaussian(img, sigma=gaussian_sigma_img)

    # Thresholding
    plot_histogram(mask, save_fig=True, title=filename + "_filt_histogram")
    mask = threshold_image(mask, threshold=threshold_value)

    # Fill holes
    struct5 = np.ones(struct_fill_holes)
    mask = ndimage.binary_fill_holes(mask, structure=struct5)

    # Smooth mask
    mask = filters.gaussian(mask, sigma=gaussian_sigma_mask)

    # Fill holes again
    mask = ndimage.binary_fill_holes(mask)

    # Morphological erosion
    struct3 = ndimage.generate_binary_structure(*erosion_structure)
    mask = ndimage.binary_erosion(mask, structure=struct3, iterations=erosion_iterations)

    # --- Final visualization ---
    viz_orthogonal_slices(mask + img, range(10, min(img.shape), 25), savefig=True, title=filename + "_mask_viz")

    # --- Upscaling (nearest neighbor) ---
    if args.out_dataformat == "UINT8":
        out_dataformat = np.uint8
        mask = mask.astype(out_dataformat) * 255
    else:
        pass
        #out_dataformat = np.float32

    # Write nifti for viz
    print(f"Writing {scan_path}")
    out_path = os.path.join(args.base_path, args.out_path, args.out_name + "_down.nii.gz")
    write_nifti(mask, affine=np.eye(4), output_path=out_path, dtype=np.float32)

    img_upscaled = transform.resize(
        mask,
        np.array(img.shape) * scale_factor,
        order=0,  # nearest neighbor
        anti_aliasing=False,
        preserve_range=True
    ).astype(out_dataformat)

    if out_format == ".npy":
        write_npy(img_upscaled, output_path=out_path, dtype=out_dataformat)
    elif out_format == ".tiff" or out_format == ".tif":
        write_tiff(img_upscaled, output_path=out_path, dtype=out_dataformat)
    elif out_format == ".nii" or out_format == ".nii.gz":
        write_nifti(img_upscaled, affine=np.eye(4), output_path=out_path, dtype=out_dataformat)
    else:
        raise ValueError(f"Unsupported file format: {out_format}")

    # img = mask_cylinder(img, cylinder_radius=230)
    # mask = create_cylinder_mask(img.shape, cylinder_radius=230*4)
    #mask = create_mask(img, closing_ite=10, dilation_ite=3)

    # out_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_A/mask_scale_1.nii.gz"
    # write_nifti(mask.astype(np.uint8), nifti_data.affine, out_path, dtype=np.uint8, ret=False)

    print("Done")