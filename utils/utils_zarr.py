import math
import os
import glob
import numpy as np
from ome_zarr.format import CurrentFormat, Format, FormatV04
import zarr
from zarr.storage import LocalStore
from skimage.exposure import match_histograms
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
#from numcodecs import Zstd, Blosc, LZ4
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle
from utils.utils_image import load_image, normalize, normalize_std, normalize_std_dask, match_histogram_3d_continuous_sampled, compare_histograms
from utils.utils_preprocess import image_crop_pad
from dask.diagnostics import ProgressBar
from utils.utils_plot import viz_slices, viz_orthogonal_slices, viz_multiple_images


def checkpoint_as_zarr(arr, path, chunks=None, shards=None):

    store = zarr.storage.LocalStore(path)

    z = zarr.create_array(
        store=store,
        shape=arr.shape,
        chunks=arr.chunksize if chunks is None else chunks,
        dtype=arr.dtype,
        shards=arr.chunksize if shards is None else shards,
        overwrite=True
    )

    # Store the Dask array into the Zarr array
    da.store(arr, z, compute=True)

    # Reload with a clean graph
    #arr = da.from_zarr(store, chunks=chunks)
    arr = zarr.open(store, chunks=chunks, shards=shards)

    return da.array(arr)


def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), shard_size=None, cname=None, coordinate_transforms=None, format=None):

    if format is None:
        format = CurrentFormat()  # Use the latest OME-Zarr format by default
    elif format == 'V04':
        format = FormatV04()
    else:
        raise ValueError(f"Unsupported OME-Zarr format: {format}")

    # Define the chunk sizes for each level
    chunk_shapes = [np.array(chunk_size) // (2 ** i) for i in range(len(image_pyramid))]
    print("Chunk shapes: ", chunk_shapes)

    # Define the shard sizes for each level
    shard_shapes = None
    if shard_size is not None:
        shard_shapes = [chunk_shapes[i] * np.array(shard_size) * (2 ** i) for i in range(len(image_pyramid))]
        print("Shard shapes: ", shard_shapes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {
            "chunks": chunk_shapes[i].tolist(),
            # "compressors": BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle),
            # "shards": shard_shapes[i].tolist()
        }
        for i in range(len(image_pyramid))
    ]
    if shard_shapes is not None:
        for i in range(len(storage_opts)):
            storage_opts[i]["shards"] = shard_shapes[i].tolist()

    if cname is not None:
        if format == CurrentFormat():
            for i in range(len(storage_opts)):
                storage_opts[i]["compressors"] = BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle)
        elif format == "V04":
            for i in range(len(storage_opts)):
                storage_opts[i]["compressor"] = BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle)

    with ProgressBar(dt=1.0):
        # Write the image data to the Zarr group
        write_multiscale(
                image_pyramid,
                group=image_group,
                axes=["z", "y", "x"],
                storage_options=storage_opts,
                coordinate_transform=coordinate_transforms,
                fmt=format
            )

    with ProgressBar(dt=1.0):
        if label_pyramid is not None:
            # Now write the label pyramid under /volume/labels/mask/
            write_multiscale_labels(
                label_pyramid,
                group=image_group,
                name="mask",
                axes=["z", "y", "x"],
                storage_options=storage_opts,
                coordinate_transform=coordinate_transforms,
                fmt=format
            )

    print("Done writing multiscale data to OME-Zarr group")


def write_ome_datasample(out_name,
                         HR_paths,
                         HR_mask_paths,
                         LR_paths,
                         LR_mask_paths,
                         REG_paths,
                         REG_mask_paths,
                         HR_chunks,
                         LR_chunks,
                         REG_chunks,
                         HR_split_indices=(),
                         LR_split_indices=(),
                         REG_split_indices=(),
                         split_axis=0,
                         compression='lz4'):

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

    if len(HR_paths) == 0:
        raise ValueError("HR image paths are required and cannot be empty.")

    hr_vals = write_ome_group_resmatch(image_paths=HR_paths,
                                          mask_paths=HR_mask_paths,
                                          out_name=out_name,
                                          group_name='HR',
                                          split_axis=split_axis,
                                          split_indices=HR_split_indices,
                                          chunks=HR_chunks,
                                          compression=compression)

    if len(LR_paths) == 0:
        print("No LR image paths provided, skipping LR group.")
    else:
        write_ome_group_resmatch(image_paths=LR_paths,
                        mask_paths=LR_mask_paths,
                        out_name=out_name,
                        group_name='LR',
                        split_axis=split_axis,
                        split_indices=LR_split_indices,
                        chunks=LR_chunks,
                        compression=compression)

    if len(REG_paths) == 0:
        print("No REG image paths provided, skipping REG group.")
    else:
        match_indices = {0: 2, 1: 3}  # match REG/0 with HR/2, and REG/1 with HR/3
        write_ome_group_resmatch(image_paths=REG_paths,
                        mask_paths=REG_mask_paths,
                        out_name=out_name,
                        group_name='REG',
                        split_axis=split_axis,
                        split_indices=REG_split_indices,
                        chunks=REG_chunks,
                        compression=compression,
                        reference_val_list=hr_vals,
                        match_indices=match_indices)

    return 0


def write_ome_group(image_paths, mask_paths=None, out_name="", group_name='HR', split_axis=0, split_indices=(), chunks=(160, 160, 160), compression='lz4', reference_vals=None):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid = load_image_pyramid(image_paths, dtype=np.float32)
    print("Image pyramid shapes:", [img.shape for img in pyramid])

    mask_pyramid = None
    if mask_paths is not None:
        mask_pyramid = load_image_pyramid(mask_paths, dtype=np.uint8)
        print("Mask pyramid shapes:", [img.shape for img in mask_pyramid]) # ref_mask = load_image("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/fixed_scale_1_mask.nii.gz")

    # match histograms
    start_idx = 0
    if reference_vals is None:
        reference_vals = pyramid[0]  # use first image as reference
        start_idx = 1  # normalize remaining images based on first image
        if mask_pyramid is not None:
            reference_mask = mask_pyramid[0].astype(bool)
            reference_vals = reference_vals[reference_mask]

    for i in range(start_idx, len(pyramid)):
        print(f"Matching histogram pyramid level {i}...")
        source_vals = pyramid[i]

        # viz_slices(pyramid[i], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)

        if mask_pyramid is not None:
            source_mask = mask_pyramid[i].astype(bool)
            matched_vals = match_histograms(source_vals[source_mask], reference_vals)
            source_vals[source_mask] = matched_vals
            pyramid[i] = source_vals
        else:
            matched_vals = match_histograms(source_vals, reference_vals)
            pyramid[i] = matched_vals

        # viz_slices(pyramid[i], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)

    ######
    # matched = match_histograms(pyramid[2], reference_image)
    # viz_slices(pyramid[2], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(matched, [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(reference_image, [10 * 4, 20 * 4, 30 * 4], savefig=False, vmin=0, vmax=1, axis=0)
    # compare_histograms(pyramid[2], reference_image)
    ######

    # Split pyramid (if needed)
    pyramid_splits = split_pyramid(pyramid, split_axis, split_indices)

    out_path = out_name

    if split_indices:
        # Create file name for each split index
        if ".zarr" in out_path:
            out_paths = [out_path.replace(".zarr", f"_{i}.zarr") for i in range(len(split_indices) + 1)]
        else:
            out_paths = [f"{out_path}_{i}.zarr" for i in range(len(split_indices) + 1)]
    else:
        if ".zarr" not in out_path:
            out_path = f"{out_path}.zarr"
        out_paths = [out_path]

    for i in range(len(split_indices) + 1):

        out_path = out_paths[i]
        print(f"Writing OME-Zarr data sample to {out_path}, split index {i}/{len(split_indices)}")

        # Create/open a Zarr array in write mode
        store = parse_url(out_path, mode="w").store
        # store = DirectoryStore(file_path)
        root = zarr.group(store=store)

        if os.path.exists(os.path.join(out_path, group_name)):
            print(f"Group {group_name} already exists in {out_path}. Skipping...")
            continue
        else:
            # Create image group for the volume
            image_group = root.create_group(group_name)

        write_ome_pyramid(
            image_group=image_group,
            image_pyramid=pyramid_splits[i],
            label_pyramid=None,  # No labels
            chunk_size=chunks,
            cname=compression  # Compression codec
        )

        print(f"Done writing OME-Zarr data to {out_path}/{group_name}")

    return reference_vals


def write_ome_group_resmatch(image_paths, mask_paths=None, out_name="", group_name='HR', split_axis=0, split_indices=(), chunks=(160, 160, 160), compression='lz4', reference_val_list=None, match_indices=None):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid = load_image_pyramid(image_paths, dtype=np.uint16)
    print("Image pyramid shapes:", [img.shape for img in pyramid])

    if group_name != "REG":
        viz_slices(pyramid[2], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_raw")

    mask_pyramid = None
    if mask_paths is not None:
        mask_pyramid = load_image_pyramid(mask_paths, dtype=np.uint8)
        print("Mask pyramid shapes:", [img.shape for img in mask_pyramid]) # ref_mask = load_image("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/fixed_scale_1_mask.nii.gz")

    source_vals_list = []
    for i in range(len(pyramid)):
        source_vals = pyramid[i]
        if mask_pyramid is not None:
            source_mask = mask_pyramid[i].astype(bool)
            source_vals = source_vals[source_mask]
        source_vals_list.append(source_vals)

    # match histograms
    for i in range(len(source_vals_list)):
        if reference_val_list is None:
            break
        elif match_indices is None:
            break
        elif i in match_indices:
            source_vals = source_vals_list[i]
            reference_vals = reference_val_list[match_indices[i]]
            print(f"Matching histogram level {i} with reference level {match_indices[i]}...")

            viz_slices(pyramid[i], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_raw")

            if mask_pyramid is not None:
                source_mask = mask_pyramid[i].astype(bool)
                matched_vals = match_histograms(source_vals, reference_vals)  # Do this slice-wise, or N-slice-wise!
                pyramid[i][source_mask] = matched_vals
            else:
                matched_vals = match_histograms(source_vals, reference_vals)
                pyramid[i] = matched_vals

            viz_slices(pyramid[i], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_matched")

    ######
    # matched = match_histograms(pyramid[2], reference_image)
    # viz_slices(pyramid[2], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(matched, [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(reference_image, [10 * 4, 20 * 4, 30 * 4], savefig=False, vmin=0, vmax=1, axis=0)
    # compare_histograms(pyramid[2], reference_vals)
    ######

    # Split pyramid (if needed)
    pyramid_splits = split_pyramid(pyramid, split_axis, split_indices)

    out_path = out_name

    if split_indices:
        # Create file name for each split index
        if ".zarr" in out_path:
            out_paths = [out_path.replace(".zarr", f"_{i}.zarr") for i in range(len(split_indices) + 1)]
        else:
            out_paths = [f"{out_path}_{i}.zarr" for i in range(len(split_indices) + 1)]
    else:
        if ".zarr" not in out_path:
            out_path = f"{out_path}.zarr"
        out_paths = [out_path]

    for i in range(len(split_indices) + 1):

        out_path = out_paths[i]
        print(f"Writing OME-Zarr data sample to {out_path}, split index {i}/{len(split_indices)}")

        # Create/open a Zarr array in write mode
        store = parse_url(out_path, mode="w").store
        # store = DirectoryStore(file_path)
        root = zarr.group(store=store)

        if os.path.exists(os.path.join(out_path, group_name)):
            print(f"Group {group_name} already exists in {out_path}. Skipping...")
            continue
        else:
            # Create image group for the volume
            image_group = root.create_group(group_name)

        pyramid_splits[i] = [da.array(img) for img in pyramid_splits[i]] # REMOVE THIS

        write_ome_pyramid(
            image_group=image_group,
            image_pyramid=pyramid_splits[i],
            label_pyramid=None,  # No labels
            chunk_size=chunks,
            cname=compression,  # Compression codec
            format=None
        )

        print(f"Done writing OME-Zarr data to {out_path}/{group_name}")

    return source_vals_list


def load_image_pyramid(image_paths, dtype=np.uint16, nifti_backend="nibabel"):
    """
    Load a pyramid of images from given paths.
    If mask_paths are provided, apply the masks to the images.
    """
    pyramid = []
    for i, image_path in enumerate(image_paths):
        print(f"Loading pyramid level {i}...")
        image = load_image(image_path, dtype=dtype, as_contiguous=True, nifti_backend=nifti_backend)
        pyramid.append(image)

    return pyramid


def split_pyramid(pyramid, split_axis=0, split_indices=()):

    if len(split_indices) == 0:
        # If no splitting is required, return the pyramid as is
        return [pyramid]

    # Split the pyramid images into num_splits along the specified axis
    pyramid_splits = [[] for _ in range(len(split_indices) + 1)]
    for i, image in enumerate(pyramid):
        # Check if the image is a Dask array
        if isinstance(image, da.Array):
            # TODO: Handle Dask arrays
            raise NotImplementedError("Dask array splitting is not implemented yet.")
        elif isinstance(image, np.ndarray):
            print(f"Splitting pyramid image: {i} along axis {split_axis} with indices {list(np.array(split_indices) // 2**i)}")
            splits = np.array_split(image, np.array(split_indices) // 2**i, axis=split_axis)

        for i, split in enumerate(splits):
            pyramid_splits[i].append(split)

    return pyramid_splits

def load_image_pyramid_splits(image_paths, split_axis=0, split_indices=(), dtype=np.float32, norm_method=None):

    # Load image pyramid
    pyramid = []
    for i, image_path in enumerate(image_paths):
        # Load image
        print(f"Loading image: {os.path.basename(image_path)}")
        image = load_image(image_path, dtype=dtype, as_contiguous=True, as_dask_array=False)

        if norm_method is None:
            print(f"Skipping normalization for image: {os.path.basename(image_path)}")
        elif norm_method == "min_max":
            print(f"Normalizing image: {os.path.basename(image_path)} using min-max normalization")
            image = normalize(image, global_min=None, global_max=None, dtype=image.dtype)
        elif norm_method == "std":
            print(f"Normalizing image: {os.path.basename(image_path)} to +/- 3 standard deviations")
            image = normalize_std(image, standard_deviations=3, mode='rescale')
        else:
            raise ValueError(f"Unsupported normalization method: {norm_method}")

        pyramid.append(image)

    if len(split_indices) == 0:
        # If no splitting is required, return the pyramid as is
        return [pyramid]

    # Split the pyramid images into num_splits along the specified axis
    pyramid_splits = [[] for _ in range(len(split_indices) + 1)]
    for i, image in enumerate(pyramid):
        # Check if the image is a Dask array
        if isinstance(image, da.Array):
            # TODO: Handle Dask arrays
            raise NotImplementedError("Dask array splitting is not implemented yet.")
        elif isinstance(image, np.ndarray):
            print(f"Splitting pyramid image: {i} along axis {split_axis} with indices {list(np.array(split_indices) // 2**i)}")
            splits = np.array_split(image, np.array(split_indices) // 2**i, axis=split_axis)

        for i, split in enumerate(splits):
            pyramid_splits[i].append(split)

    return pyramid_splits


if __name__ == "__main__":

    sample_name = "Femur_01_80kV"
    project_path = "../../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_01"

    HR_paths = glob.glob(os.path.join(project_path, "fixed_scale_*.nii.gz"))
    LR_paths = glob.glob(os.path.join(project_path, "moving_scale_*.nii.gz"))
    REG_paths = glob.glob(os.path.join(project_path, f"{sample_name}*.nii.gz"))

    out_name = os.path.join(project_path, "Femur_01_ome")

    write_ome_datasample(out_name=out_name,
                         HR_paths=HR_paths,
                         LR_paths=LR_paths,
                         REG_paths=REG_paths,
                         HR_chunks=(160, 160, 160),
                         LR_chunks=(120, 120, 120),
                         REG_chunks=(40, 40, 40),
                         HR_split_indices=(320, ),
                         LR_split_indices=(480, ),
                         REG_split_indices=(480, ),
                         split_axis=0,
                         compression='lz4')

    print("Done writing OME-Zarr data sample")