import nibabel as nib
import numpy as np
import SimpleITK as sitk
import ants

def write_nifti(image, affine=np.eye(4), output_path="", dtype=np.float32, ret=False):
    """
    Write a numpy array to a NIfTI file.
    """

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(image, affine=affine, dtype=dtype)

    # Save the NIfTI image
    nib.save(nifti_image, output_path)
    print(f"Saved nifti image to: {output_path}")

    if ret:
        return image


def write_nifti_ants(image, affine=np.eye(4), output_path="", dtype=np.float32, ret=False):
    """
    Write a numpy array to a NIfTI file, preserving affine semantics exactly.
    Uses ANTsPy (ITK backend) for faster writing.
    """

    # Ensure dtype + contiguity
    image = np.ascontiguousarray(image.astype(dtype))

    # Convert numpy array + affine to ANTs image
    ants_img = ants.from_numpy(image, origin=list(affine[:3, 3]), spacing=list(np.linalg.norm(affine[:3, :3], axis=0)))

    # Direction matrix is normalized affine rotation/scaling
    dir_matrix = (affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0))
    ants_img.set_direction(dir_matrix)

    # Save image
    ants.image_write(ants_img, output_path)
    print(f"Saved nifti image to: {output_path}")

    if ret:
        return image

def load_nifti(input_path):
    """
    Read a NIfTI file and return the image data.
    """
    nifti_image = nib.load(input_path)
    image_data = nifti_image.get_fdata()
    print(f"Read nifti image from: {input_path}")
    return image_data, nifti_image, nifti_image.affine

def crop_nifti(data, affine, crop_start, crop_end):

    # Define crop indices (example)
    x0, y0, z0 = crop_start  # starting voxel indices
    x1, y1, z1 = crop_end  # ending voxel indices (exclusive)

    # Crop the image data
    cropped_data = data[x0:x1, y0:y1, z0:z1]

    # Compute new affine
    # The new origin in world coordinates = affine @ [x0, y0, z0, 1]
    new_origin = affine @ [x0, y0, z0, 1]
    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin[:3]

    return cropped_data, new_affine

def compute_affine_crop(affine, crop_start):
    """
    Compute the new affine matrix based on the crop region.
    :param affine: original affine matrix
    :param crop_start: starting voxel indices for cropping
    :return: new affine matrix
    """

    # The new origin in world coordinates = affine @ [x0, y0, z0, 1]
    new_origin = get_crop_origin(affine, crop_start)

    # Set the translation part of the affine matrix to the new origin
    new_affine = set_origin(affine.copy(), new_origin)

    return new_affine

def compute_affine_scale(affine, scale):
    """
    Compute the new affine matrix based on the scale.
    :param affine: original affine matrix
    :param scale: scaling factor
    :return: new affine matrix
    """

    # Set the scaling part of the affine matrix
    new_affine = set_affine_scale(affine.copy(), scale)

    return new_affine

def get_crop_origin(affine, crop_start):

    # The new origin in world coordinates = affine @ [x0, y0, z0, 1]
    new_origin = voxel2world(affine, crop_start)

    return new_origin


def set_origin(affine, new_origin):

    # Set the translation part of the affine matrix to the new origin
    affine[:3, 3] = new_origin

    return affine


def set_affine_scale(affine, scale):

    affine[:3, :3] *= scale

    return affine


def modify_nifti_origin(nifti_path, new_origin):

    """
    Modify the origin of a NIfTI file to a new specified origin.

    # Example usage
    sample_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/stitched/processed/Bamboo_A_bin1x1/"
    nifti_path = sample_path + "Bamboo_A_bin1x1.nii.gz"
    modify_nifti_origin(nifti_path, new_origin=(0, 0, 0))

    :param nifti_path:
    :param new_origin:
    :return:
    """

    img = nib.load(nifti_path)

    # Modify the affine: keep voxel spacing and orientation, but move origin to (0, 0, 0)
    new_affine = img.affine.copy()
    new_affine[:3, 3] = new_origin  # Set the translation (origin) to (0, 0, 0)

    # Create a new NIfTI image with the modified affine
    new_img = nib.Nifti1Image(img.dataobj, affine=new_affine, header=img.header)

    # Save the modified image
    nib.save(new_img, nifti_path)


def get_affine_from_itk_image(image):

    origin = np.array(image.GetOrigin())  # (x0, y0, z0)
    spacing = np.array(image.GetSpacing())  # (sx, sy, sz)
    direction = np.array(image.GetDirection())  # 9 elements (row-major)

    # Reshape direction to 3x3
    direction = direction.reshape(3, 3)

    # Compute nifti affine
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def voxel2world(affine, point):

    """
    Convert a point in pixel coordinates to world coordinates using the affine transformation matrix.
    :param affine: 4x4 affine transformation matrix
    :param point: 3D point in pixel coordinates (x, y, z)
    :return: 3D point in world coordinates (x_w, y_w, z_w)
    """
    point_homogeneous = np.array([point[0], point[1], point[2], 1])
    world_point = affine @ point_homogeneous
    return world_point[:3]


def compute_world_size(affine, image_shape):

    size = np.abs(affine[:3, :3]) @ (np.array(image_shape) - 1)
    return size




if __name__ == "__main__":

    import time
    image_size = 512

    image = np.random.rand(image_size, image_size, image_size).astype(np.float32)
    affine = np.array([[1, 0, 0, 10],
                       [0, 1, 0, 20],
                       [0, 0, 1, 30],
                       [0, 0, 0, 1]])

    start = time.time()
    write_nifti(image, affine, "output_baseline.nii")
    stop = time.time()
    print("Time elapsed:", stop - start)


    image = np.random.rand(image_size, image_size, image_size).astype(np.float32)
    affine = np.array([[1, 0, 0, 10],
                       [0, 1, 0, 20],
                       [0, 0, 1, 30],
                       [0, 0, 0, 1]])

    start = time.time()
    write_nifti_ants(image, affine, "output_fast.nii")
    stop = time.time()
    print("Time elapsed:", stop - start)