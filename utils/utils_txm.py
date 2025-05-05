import numpy as np
import qim3d as qim

def load_txm(input_path, dtype=np.float32):
    print(f"Reading input file: {input_path}")
    image, metadata = qim.io.load(input_path, virtual_stack=False, progress_bar=True, dtype=dtype, return_metadata=True)
    print(f"txm shape: {image.shape}")
    return image, metadata


def get_affine_txm(metadata, custom_origin=None):

    pixel_size = metadata['pixel_size']  # in microns

    # Convert pixel size to millimeters
    pixel_size_mm = pixel_size / 1000.0

    voxel_spacing_mm = [pixel_size_mm] * 3  # Isotropic assumed

    # Create the affine transformation matrix, assuming Nifti (D,H,W) order
    affine = np.diag([voxel_spacing_mm[0], voxel_spacing_mm[1], voxel_spacing_mm[2], 1])

    x_offset = metadata['x_positions'][0] / 1000.0
    y_offset = metadata['y_positions'][0] / 1000.0
    z_offset = metadata['z_positions'][0] / 1000.0  # Assuming the first position is the reference

    # Set the translation part of the affine matrix
    if custom_origin is not None:
        affine[0, 3] = custom_origin[0]
        affine[1, 3] = custom_origin[1]
        affine[2, 3] = custom_origin[2]
    else:
        affine[0, 3] = x_offset
        affine[1, 3] = y_offset
        affine[2, 3] = z_offset

    return affine


def get_itk_txm(metadata):

    # Convert pixel size from microns to millimeters
    pixel_size_mm = metadata['pixel_size'] / 1000.0

    spacing = np.array([pixel_size_mm] * 3)  # Isotropic assumed
    origin = np.array([0.0] * 3)  # Assuming the origin is at (0, 0, 0)
    size = np.array([metadata['number_of_images'], metadata['image_height'], metadata['image_width']])  # Assuming the size is in (Z, Y, X) order
    direction = np.eye(3)  # Identity matrix for direction cosines

    return origin, spacing, direction, size