import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage
from scipy.signal import find_peaks
from utils.utils_image import load_image, normalize_std
from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_nifti import write_nifti

def make_diamond_3d(radius):
    size = 2 * radius + 1
    Z, Y, X = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing='ij'
    )
    manhattan_distance = np.abs(X) + np.abs(Y) + np.abs(Z)
    return manhattan_distance <= radius

def mask_cylinder(img, pixel_radius):

    D, H, W = img.shape

    # For every slice, any voxels outside the pixel radius will be set to 0

    slice_center = (H / 2, W / 2)

    for slice_idx in range(D):
        Y, X = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((Y - slice_center[0])**2 + (X - slice_center[1])**2)
        mask = distance_from_center <= pixel_radius

        img[slice_idx, :, :] *= mask

    return img


def create_cylinder_mask(shape, pixel_radius):

    D, H, W = shape

    # For every slice, any voxels outside the pixel radius will be set to 0
    slice_center = (H / 2, W / 2)
    mask = np.zeroes((D, H, W), dtype=np.uint8)

    for slice_idx in range(D):
        Y, X = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((Y - slice_center[0])**2 + (X - slice_center[1])**2)
        mask[D] = distance_from_center <= pixel_radius

    return mask


def create_mask(img, closing_ite=10, dilation_ite=3):
    """
    Create a binary mask from the input image using morphological operations.

    :param image: Input 3D image array.
    :param structure: Structuring element for morphological operations.
    :param iterations: Number of iterations for morphological closing.
    :return: Binary mask of the input image.
    """

    # borderpad
    padding = 30
    img = np.pad(img, ((padding, padding), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # gaussian smoothing
    img = filters.gaussian(img, sigma=0.8, preserve_range=True)
    viz_slices(img, slices, savefig=False)

    # Histogram
    counts, bins = np.histogram(img, bins=512, density=True)
    plt.figure(figsize=(10, 5))
    plt.stairs(counts, bins)
    plt.title('Histogram of Image Intensities')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid()
    # plt.xlim([0.1, 0.4])

    peaks, _ = find_peaks(counts, distance=2, threshold=3)
    minima, _ = find_peaks(-counts, distance=2, threshold=3)

    plt.plot(bins[peaks], counts[peaks], "ro", label="Peaks")
    plt.plot(bins[minima], counts[minima], "go", label="Minima")
    plt.show()

    # import qim3d
    # qim3d.viz.threshold(img, cmap_image='magma', vmin=None, vmax=None)

    # Apply Otsu threshold
    thresh = 0.22  # filters.threshold_otsu(img)
    img[img < thresh] = 0
    viz_slices(img, slices, savefig=False)

    # structure = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                       [0, 1, 0, 1, 1, 1, 0, 1, 0],
    #                       [0, 0, 0, 0, 1, 0, 0, 0, 0]]).reshape((3,3,3))

    structure = make_diamond_3d(radius=1)
    # structure = np.ones((2, 2, 2))

    # Apply binary closing to fill holes
    img = ndimage.binary_closing(img, structure=structure, iterations=closing_ite)
    viz_slices(img, slices, savefig=False)

    # Dilate a little, since its better to have mask a bit larger than the actual vertebra
    img = ndimage.binary_dilation(img, structure=structure, iterations=dilation_ite)
    viz_slices(img, slices, savefig=False)

    # slight smoothing and binarization again
    img = filters.gaussian(img, sigma=1.5, preserve_range=True)
    img[img < 0.5] = 0
    img[img >= 0.5] = 1.0
    viz_slices(img, slices, savefig=False)

    # Remove padding again
    img = img[padding:-padding, padding:-padding, padding:-padding]
    viz_slices(img, slices, savefig=False)

    # Label connected regions
    labels = measure.label(img)

    # Keep only the largest connected component (assumed to be vertebra)
    regions = measure.regionprops(labels)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        mask = labels == largest_region.label
    else:
        mask = np.zeros_like(labels, dtype=bool)

    viz_slices(mask, slices, savefig=False)

    return mask



if __name__ == "__main__":

    # Load image
    #file_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_A/Vertebrae_A_80kV_registered_scale_2.nii.gz"
    file_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_A/fixed_scale_1.nii.gz"
    slices = [50, 100, 150]

    img, nifti_data = load_image(file_path, dtype=np.float32, return_metadata=True)
    viz_slices(img, slices, savefig=False)

    # img = mask_cylinder(img, pixel_radius=230)
    mask = create_cylinder_mask(img.shape, pixel_radius=230*4)

    #mask = create_mask(img, closing_ite=10, dilation_ite=3)

    out_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_A/mask_scale_1.nii.gz"
    write_nifti(mask.astype(np.uint8), nifti_data.affine, out_path, dtype=np.uint8, ret=False)

    print("Done")