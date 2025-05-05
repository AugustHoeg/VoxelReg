import numpy as np
import qim3d as qim

def load_txm(input_path, dtype=np.float32):
    print(f"Reading input file: {input_path}")
    image, metadata = qim.io.load(input_path, virtual_stack=False, progress_bar=True, dtype=dtype, return_metadata=True)
    print(f"txm shape: {image.shape}")
    return image, metadata


def get_affine_txm(metadata):

    pixel_size = metadata['pixel_size']  # in microns

    # Convert pixel size to millimeters
    pixel_size_mm = pixel_size / 1000.0

    voxel_spacing_mm = [pixel_size_mm] * 3  # Isotropic assumed

    # Create the affine transformation matrix, assuming Nifti (Z,Y,X) order
    affine = np.diag([voxel_spacing_mm[2], voxel_spacing_mm[1], voxel_spacing_mm[0], 1])

    # Set the translation part of the affine matrix
    affine[0:3, 3] = 0  # we assume the origin is at (0, 0, 0)

    return affine


def get_itk_txm(metadata):

    # Convert pixel size from microns to millimeters
    pixel_size_mm = metadata['pixel_size'] / 1000.0

    spacing = np.array([pixel_size_mm] * 3)  # Isotropic assumed
    origin = np.array([0.0] * 3)  # Assuming the origin is at (0, 0, 0)
    size = np.array([metadata['number_of_images'], metadata['image_height'], metadata['image_width']])  # Assuming the size is in (Z, Y, X) order
    direction = np.eye(3)  # Identity matrix for direction cosines

    return origin, spacing, direction, size