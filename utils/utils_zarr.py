
import os
import glob
import numpy as np
import zarr
from zarr.storage import DirectoryStore
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from utils.utils_image import load_image, normalize_std

def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), cname='lz4'):

    # Define the chunk sizes for each level
    chunk_sizes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    print("Chunk sizes: ", chunk_sizes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {"chunks": chunk_sizes[i], "compression": Blosc(cname=cname, clevel=3, shuffle=Blosc.BITSHUFFLE)}
        for i in range(len(image_pyramid))
    ]

    # Write the image data to the Zarr group
    write_multiscale(
            image_pyramid,
            group=image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    if label_pyramid is not None:
        # Now write the label pyramid under /volume/labels/mask/
        write_multiscale_labels(
            label_pyramid,
            group=image_group,
            name="mask",
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    print("Done writing multiscale data to OME-Zarr group")

def write_ome_datasample(out_path, HR_paths, LR_paths, REG_paths, HR_chunks, LR_chunks, REG_chunks):

    """
    We need these images:
        ome.zarr
        - HR
            - fixed_scale_1.nii.gz
            - fixed_scale_2.nii.gz
            - fixed_scale_4.nii.gz
            - fixed_scale_8.nii.gz
        - LR
            - moving_scale_1.nii.gz
            - moving_scale_2.nii.gz
            - moving_scale_4.nii.gz
        - REG
            - <sample_name>.nii.gz
            - <sample_name>_scale_2.nii.gz
    """

    if HR_paths is None:
        raise ValueError("HR image paths are required and cannot be empty.")

    # Create/open a Zarr array in write mode
    file_path = f"{out_path}.zarr"

    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping...")
        return -1

    store = parse_url(file_path, mode="w").store
    #store = DirectoryStore(file_path)
    root = zarr.group(store=store)

    write_ome_group(root,
                    group_name='HR',
                    image_paths=HR_paths,
                    mask_paths=None,
                    chunk_size=HR_chunks,
                    cname='lz4')

    if LR_paths is None:
        print("No LR image paths provided, skipping LR group.")
    else:
        write_ome_group(root,
                        group_name='LR',
                        image_paths=LR_paths,
                        mask_paths=None,
                        chunk_size=LR_chunks,
                        cname='lz4')

    if REG_paths is None:
        print("No REG image paths provided, skipping REG group.")
    else:
        write_ome_group(root,
                        group_name='REG',
                        image_paths=REG_paths,
                        mask_paths=None,
                        chunk_size=REG_chunks,
                        cname='lz4')

    print(f"Done writing OME-Zarr data sample to {file_path}")



def write_ome_group(root,
                    group_name,
                    image_paths,
                    mask_paths=None,
                    chunk_size=(160, 160, 160),
                    cname='lz4'):

    # Load HR image pyramid
    pyramid = []
    for i in range(len(image_paths)):
        # load image
        image_path = image_paths[i]
        image = load_image(image_path, dtype=np.float32)
        image = np.ascontiguousarray(image)

        # Masking
        if mask_paths is not None:
            mask_path = mask_paths[i]
            mask = load_image(mask_path, dtype=np.float32)

            # processing using mask
            pass

        # Cropping
        image = image[:, :, :]

        # Normalize to +-3 standard deviations, rescaled to 0-1
        image = normalize_std(image, standard_deviations=3, mode='rescale')

        pyramid.append(image)

    # Create image group for the volume
    image_group = root.create_group(group_name)

    write_ome_pyramid(
        image_group=image_group,
        image_pyramid=pyramid,
        label_pyramid=None,  # No labels
        chunk_size=chunk_size,
        cname=cname  # Compression codec
    )



if __name__ == "__main__":

    sample_name = "Femur_74_80kV"
    project_path = "../../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_74/"

    HR_paths = glob.glob(os.path.join(project_path, "fixed_scale_*.nii.gz" ))
    LR_paths = glob.glob(os.path.join(project_path, "moving_scale_*.nii.gz"))
    REG_paths = glob.glob(os.path.join(project_path, f"{sample_name}*.nii.gz"))

    out_path = os.path.join(project_path, "Femur_74_ome")

    write_ome_datasample(out_path=out_path,
                         HR_paths=HR_paths,
                         LR_paths=LR_paths,
                         REG_paths=REG_paths,
                         HR_chunks=(80, 80, 80),
                         LR_chunks=(80, 80, 80),
                         REG_chunks=(40, 40, 40)
                         )

    print("Done writing OME-Zarr data sample")