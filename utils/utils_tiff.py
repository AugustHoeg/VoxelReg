import os
import glob
import datetime
import numpy as np
import tifffile
import argparse
import zarr
from dask import delayed
import dask.array as da
from scipy.ndimage import zoom
import multiprocessing as mp
from natsort import natsorted
from zarr.codecs import BytesCodec, BloscCodec, BloscCname, BloscShuffle

def crop_tiff_slice(tiff_slice, start_row, end_row, start_col, end_col):

    return tiff_slice[start_row:end_row, start_col:end_col]


def parallel_crop_tiff(tiff_path, start_row, end_row, start_col, end_col, start_depth, end_depth, n_proc=8):
    """Estimate percentiles using multiprocessing."""

    with tifffile.TiffFile(tiff_path) as tif:

        depth = len(tif.pages)
        print(f"Number of slices: {depth}")

        num_read = 0
        num_write = 0

        # Read N slices
        image_stack = ...

        # Create multiprocessing pool
        with mp.Pool(n_proc) as pool:

            # Start N workers
            results_async = [
                pool.apply_async(crop_tiff_slice, args=(image_stack[idx], start_row, end_row, start_col, end_col))
                for idx in range(len(image_stack))
            ]

            for idx in range(start_depth, end_depth):

                cropped_slice = results_async[0].get()
                results_async.pop()

                next_image = tif.pages[idx].asarray()
                results_async.append(pool.apply_async(crop_tiff_slice, args=(next_image, start_row, end_row, start_col, end_col)))

                num_read += 1

                # Save cropped slice


def load_tiff(input_path, dtype=np.float32, image_sequence=False):
    print(f"Reading input file: {input_path}")
    if image_sequence:
        file_list = glob.glob(input_path)
        file_list = natsorted(file_list)
        image = tifffile.imread(file_list).astype(dtype)
    else:
        image = tifffile.imread(input_path).astype(dtype)
    print(f"tiff shape: {image.shape}")
    return image

def bigtiff2dask(scan_path):

    with tifffile.TiffFile(scan_path) as tif:
        page = tif.pages[0]
        dtype = page.dtype
        n_frames = len(tif.pages)
        vol_shape = (n_frames, *page.shape)
        print(vol_shape)

    def read_single_frame(path, page_idx):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[page_idx]
            return page.asarray()

    lazy_frames = []
    for i in range(n_frames):
        d = delayed(read_single_frame)(scan_path, i)
        frame = da.from_delayed(d, shape=vol_shape[1:], dtype=dtype)
        lazy_frames.append(frame)

    dask_stack = da.stack(lazy_frames, axis=0)
    return dask_stack

def center_crop(image, target_shape):

    """
    Center crop a 3D image to the target shape.

    Args:
        image (ndarray): Input 3D image.
        target_shape (tuple): Target shape for cropping.

    Returns:
        ndarray: Cropped image.
    """

    if image.shape == tuple(target_shape):
        return image

    D, H, W = image.shape
    target_shape = [image.shape[i] if target_shape[i] == -1 else target_shape[i] for i in range(3)]

    center = (D // 2, H // 2, W // 2)

    crop_start = [max(0, center[i] - target_shape[i] // 2) for i in range(3)]
    crop_end = [min(image.shape[i], center[i] + target_shape[i] // 2) for i in range(3)]

    cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
    return cropped_image, crop_start, crop_end

def top_center_crop(image, target_shape, top_index="last"):
    """
    Top-center crop a 3D image to the target shape.
    Use -1 in any dimension to keep the original size in that dimension.

    Args:
        image (ndarray): Input 3D image with shape (D, H, W).
        target_shape (tuple of int): Desired shape (D, H, W) for cropping. Use -1 to keep original size.

    Returns:
        tuple: (cropped_image, crop_start, crop_end)
    """
    if image.shape == tuple(target_shape):
        return image, (0, 0, 0), image.shape

    # Resolve -1 entries in target_shape to use original dimensions
    target_shape = tuple(image.shape[i] if target_shape[i] == -1 else target_shape[i] for i in range(3))

    # Top-center position
    if top_index == "first":
        top_center = (0, image.shape[1] // 2, image.shape[2] // 2)
    elif top_index == "last":
        top_center = (max(0, image.shape[0] - target_shape[0]), image.shape[1] // 2, image.shape[2] // 2)
    else:
        raise ValueError("top_index must be 'first' or 'last'")

    # Compute start and end indices for cropping
    crop_start = (
        top_center[0],
        max(0, top_center[1] - target_shape[1] // 2),
        max(0, top_center[2] - target_shape[2] // 2),
    )
    crop_end = (
        min(image.shape[0], crop_start[0] + target_shape[0]),
        min(image.shape[1], crop_start[1] + target_shape[1]),
        min(image.shape[2], crop_start[2] + target_shape[2]),
    )

    # Perform cropping
    cropped_image = image[
        int(crop_start[0]):int(crop_end[0]),
        int(crop_start[1]):int(crop_end[1]),
        int(crop_start[2]):int(crop_end[2]),
    ]

    return cropped_image, crop_start, crop_end


def write_downsampled_tiff(image, output_path, factor, ret=False):
    """
    Downsamples a 3D TIFF image by the given factor and saves the result.

    Args:
        input_path (str): Path to the input 3D TIFF file.
        output_path (str): Path to save the downsampled TIFF file.
        factor (float or tuple of 3 floats): Downsampling factor(s) for (Z, Y, X).
    """
    image_dtype = image.dtype

    if isinstance(factor, (int, float)):
        factor = (factor, factor, factor)


    ########
    if False:
        import matplotlib.pyplot as plt
        from scipy import ndimage

        D, H, W = image.shape
        c = image[:, H // 2 - 250:H // 2 + 250, W // 2 - 250:W // 2 + 250]

        plt.figure(figsize=(20, 20))
        plt.imshow(c[1976 // 2, :, :])
        plt.show()

        c_smooth = ndimage.gaussian_filter(c, sigma=3)  # type: ignore
        plt.figure(figsize=(20, 20))
        plt.imshow(c_smooth[1976 // 2, :, :])
        plt.show()

        c_ds = zoom(c, zoom=1 / np.array(factor), order=1)  # linear interpolation
        plt.figure(figsize=(20, 20))
        plt.imshow(c_ds[247 // 2, :, :])
        plt.show()

        c_smooth_ds = zoom(c_smooth, zoom=1 / np.array(factor), order=1)  # linear interpolation
        plt.figure(figsize=(20, 20))
        plt.imshow(c_smooth_ds[247 // 2, :, :])
        plt.show()
    ########

    print(f"Downsampling with factor: {factor}")
    image = zoom(image, zoom=1/np.array(factor), order=1)  # linear interpolation
    print("New shape: ", image.shape)

    print(f"Downsampled shape: {image.shape}")
    tifffile.imwrite(output_path, image.astype(image_dtype))
    print(f"Saved downsampled image to: {output_path}")

    if ret:
        return image

def write_tiff(image, output_path, dtype=None, ret=False):

    if dtype is not None:
        image = image.astype(dtype)

    tifffile.imwrite(output_path, image)
    print(f"Saved tiff to: {output_path}")

    if ret:
        return image


def read_and_write_slice_tiff(
    tifffile_path,
    frame_idx,
    read_window,
    zarr_path,
    group_name,
    dtype=np.uint16
):

    # read tiff slice
    with tifffile.TiffFile(tifffile_path) as tif:
        raw = tif.pages[frame_idx].asarray()

    sl = raw[read_window].astype(dtype)

    # write to zarr chunk (ONLY THREADSAFE FOR CHUNKS = SLICES)
    z = zarr.open(os.path.join(zarr_path, group_name), mode='a')
    z[frame_idx, :, :] = sl

    return frame_idx


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tiff2zarr(
        tiff_path,
        num_workers=16,
        read_window=np.s_[:, :],
        zarr_path="data.zarr",
        group_name="raw",
        slice_shape=None,
        dtype=np.uint16,
        cname="lz4",
        clevel=3,
        return_as_dask=True):

    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]
        dtype = page.dtype
        n_frames = len(tif.pages)
        slice_shape = page.shape

    print("n_frames", n_frames)

    # ---- create zarr array ----
    store = zarr.storage.LocalStore(zarr_path)
    group = zarr.group(store=store)

    if os.path.exists(os.path.join(store.root, group_name)):
        print(f"OME level {group_name} already exists, skipping write.")

    else:
        group.create_dataset(
            name=group_name,
            shape=(n_frames, slice_shape[0], slice_shape[1]),
            chunks=(1, slice_shape[0], slice_shape[1]),
            dtype=dtype,
            compressors=BloscCodec(cname=BloscCname[cname], clevel=clevel, shuffle=BloscShuffle.bitshuffle)
        )

        print(f"Created zarr store: {zarr_path}")
        print(f"Shape: {(n_frames, slice_shape[0], slice_shape[1])}")

        # ---- multiprocessing ----
        pool = mp.Pool(num_workers)
        results = []

        for frame_idx in range(n_frames):
            args = (tiff_path, frame_idx, read_window, zarr_path, group_name)
            results.append(pool.apply_async(read_and_write_slice_tiff, args))

        pool.close()

        # ---- progress ----
        write_count = 0
        for r in results:
            frame_idx = r.get()
            write_count += 1
            print(f"{timestamp()} – Frame {frame_idx + 1}/{n_frames}", end="\r")

        pool.join()

        print("\nwrite_count", write_count)

    if return_as_dask:
        image = da.from_zarr(store, component=group_name)
        return image
    else:
        return store, group


def tiff2zarr_dask(
        tiff_path,
        num_workers=16,
        read_window=np.s_[:, :],
        zarr_path="data.zarr",
        group_name="raw",
        dtype=None,
        rescale=True,
        in_range=None,
        cname="lz4",
        clevel=3,
        return_as_dask=True):
    """
    Convert a TIFF stack to a chunked, Blosc-compressed zarr array.

    Unlike ``tiff2zarr``, this variant does NOT rely on ``tif.pages`` exposing
    every frame. It opens the whole series through tifffile's zarr store
    (``imread(..., aszarr=True)``), wraps it in a dask array, and streams it to
    disk slice-by-slice via ``da.store`` so the full volume is never held in RAM.

    Args:
        tiff_path (str): Path to the (multi-page / series) TIFF file.
        read_window (tuple of slices): Spatial crop applied to each frame,
            e.g. ``np.s_[100:900, :]``. Defaults to the full slice.
        zarr_path (str): Output zarr store path.
        group_name (str): Name of the array within the zarr group.
        dtype: Cast the data to this dtype before writing. ``None`` keeps the
            source dtype (e.g. float32).
        rescale (bool): If True (and ``dtype`` is given), linearly rescale the
            input range to the dtype's natural range before casting: integer
            dtypes map to ``[iinfo.min, iinfo.max]`` (e.g. ``[0, 65535]`` for
            uint16), floats map to ``[0, 1]``. Ignored when ``dtype`` is None.
        in_range (tuple or None): ``(in_min, in_max)`` source range to map from.
            If None, the global min/max are computed with a dask reduction
            (one extra streaming pass over the whole volume).
        cname (str): Blosc compressor name (key into ``BloscCname``).
        clevel (int): Blosc compression level.
        num_workers (int or None): Threads for the dask write. ``None`` lets dask
            decide. Reads go through a single open TIFF handle, so keep this
            modest if you hit tifffile thread-safety issues (see note below).
        return_as_dask (bool): If True, return a dask array backed by the new
            store; otherwise return ``(store, group)``.
    """

    # ---- open the whole TIFF series as a (lazy) zarr-backed dask array ----
    # This is the key difference vs. tiff2zarr: it works even when tif.pages
    # only exposes the first frame, because tifffile resolves the full series.
    tiff_store = tifffile.imread(tiff_path, aszarr=True)
    try:
        darr = da.from_zarr(tiff_store)  # shape (n_frames, H, W), chunks (1, H, W)

        # Apply the spatial read window to the (Y, X) dims of every frame.
        darr = darr[(slice(None),) + tuple(read_window)]

        if dtype is not None:
            out_dtype = np.dtype(dtype)

            if rescale:
                # ---- target range from the output dtype ----
                if np.issubdtype(out_dtype, np.integer):
                    out_min, out_max = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
                else:
                    out_min, out_max = 0.0, 1.0

                # ---- source range (auto min/max = one extra streaming pass) ----
                if in_range is None:
                    print(f"{timestamp()} – computing input min/max ...")
                    in_min, in_max = da.compute(darr.min(), darr.max())
                    in_min, in_max = float(in_min), float(in_max)
                else:
                    in_min, in_max = float(in_range[0]), float(in_range[1])
                print(f"rescaling [{in_min}, {in_max}] -> [{out_min}, {out_max}]")

                # ---- linear rescale in float, then clip/round/cast (all lazy) ----
                span = in_max - in_min
                scale = (out_max - out_min) / span if span > 0 else 0.0
                darr = (darr.astype(np.float32) - in_min) * scale + out_min
                darr = da.clip(darr, out_min, out_max)
                if np.issubdtype(out_dtype, np.integer):
                    darr = da.round(darr)

            darr = darr.astype(out_dtype)

        n_frames, H, W = darr.shape
        print("n_frames", n_frames)

        # Chunk one slice per chunk to match the write pattern below.
        chunks = (1, H, W)
        darr = darr.rechunk(chunks)

        # ---- create the output zarr array (same setup as tiff2zarr) ----
        store = zarr.storage.LocalStore(zarr_path)
        group = zarr.group(store=store)

        if os.path.exists(os.path.join(store.root, group_name)):
            print(f"OME level {group_name} already exists, skipping write.")
        else:
            z = group.create_dataset(
                name=group_name,
                shape=(n_frames, H, W),
                chunks=chunks,
                dtype=darr.dtype,
                compressors=BloscCodec(cname=BloscCname[cname], clevel=clevel, shuffle=BloscShuffle.bitshuffle)
            )

            print(f"Created zarr store: {zarr_path}")
            print(f"Shape: {(n_frames, H, W)}")

            # Stream the write. Each dask block is exactly one output chunk
            # (a single Z-slice), so lock=False is safe for the writes and
            # memory stays bounded by (num_workers * slice size).
            print(f"{timestamp()} – writing {n_frames} frames ...")
            da.store(darr, z, lock=False, num_workers=num_workers)
            print(f"{timestamp()} – done.")
    finally:
        # dask ran eagerly above, so the source handle is safe to close now.
        tiff_store.close()

    if return_as_dask:
        image = da.from_zarr(store, component=group_name)
        return image
    else:
        return store, group


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a 3D TIFF image.")
    parser.add_argument("input_path", help="Path to input TIFF stack")
    parser.add_argument("output_path", help="Path to save downsampled TIFF")
    parser.add_argument("--factor", type=float, nargs='+', default=[2.0],
                        help="Downsampling factor (single value or 3 values for Z, Y, X)")

    args = parser.parse_args()

    # Handle single factor or list of 3
    factor = args.factor
    if len(factor) == 1:
        factor = factor[0]
    elif len(factor) == 3:
        factor = tuple(factor)
    else:
        raise ValueError("Provide either one factor or three values for Z, Y, X")

    write_downsampled_tiff(args.input_path, args.output_path, factor)
