import os
import dask
import dask.array as da
import numpy as np
import h5py
import multiprocessing as mp
import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _task(h5file, frame_idx, crop_window, data_path):
    """
    Runs in each worker process:
    - opens the HDF5,
    - reads frame frame_idx,
    - applies window and dead-pixel fix,
    - runs geometry_correction,
    - returns corrected slice.
    """

    h_start, h_end, w_start, w_end = crop_window
    with h5py.File(h5file, 'r') as f:
        frame = f[data_path][frame_idx, h_start:h_end, w_start:w_end]
    min_val = np.min(frame)
    max_val = np.max(frame)
    return (min_val, max_val, frame_idx)


def get_min_max_hdf5(
        h5file,
        nworkers,
        worker_task=_task,
        crop_bounds=(0, 100, 0, 100, 0, 100),
        data_path='exchange/data',
):
    # Step 0: Get HDF5 shape
    with h5py.File(h5file, 'r') as f:
        D, H, W = f[data_path].shape
        print(f"HDF5 shape: (D={D}, H={H}, W={W})")

    # Calculate slice shape based on crop_window
    d_start, d_end, h_start, h_end, w_start, w_end = crop_bounds
    slice_shape = (min(H, h_end - h_start), min(W, w_end - w_start))
    no_slices = min(D, d_end - d_start)
    print("Slice shape:", slice_shape)
    print("No. of slices:", no_slices)

    pool = mp.Pool(nworkers)
    results = []

    # submit tasks for all frames
    for frame_idx in range(no_slices):
        args = (h5file, frame_idx + d_start, crop_bounds[2:], data_path)
        results.append(pool.apply_async(worker_task, args))

    read_count = nworkers
    write_count = 0

    global_min = np.inf
    global_max = -np.inf

    for frame_idx in range(no_slices):
        min_val, max_val, frame_idx = results[0].get()
        results.pop(0)
        print(f"{timestamp()} – Frame {frame_idx + 1}/{no_slices}, Global min/max {global_min}/{global_max}", end='\r')

        # update global min/max
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)

    print("Number of slices", no_slices)
    print(f"Global min: {global_min}, Global max: {global_max}")

    return global_min, global_max


def downscale():
    import numpy as np
    import h5py
    import os
    import dask.array as da
    from dask.diagnostics import ProgressBar

    dataset_name = 'exchange/data'

    data = h5py.File(scan_path, 'r')[dataset_name]
    d, h, w = data.shape
    print(f"HDF5 shape: (D={d}, H={h}, W={w})")
    image = da.from_array(data, chunks=(1, h, w))

    use_dask_cluster = False
    if use_dask_cluster:
        print("Using Dask cluster for multiprocessing...")
        # Step 1: Start Dask cluster (multiprocessing)
        cluster = LocalCluster(processes=True)
        client = Client(cluster)

    reduce_factor = 2
    axes = {0: reduce_factor, 1: reduce_factor, 2: reduce_factor}  # reduce by factor in (x,y,z)

    for i in range(3):
        f = reduce_factor ** (i + 1)
        print(f"Reduce factor: {f}")

        scan_out = f"{out_name}_down_{f}.h5"
        write_file = os.path.join(root, sample, scan_out)
        print(write_file)

        image = da.coarsen(np.mean, image, axes, trim_excess=True)

        with ProgressBar():
            da.to_hdf5(write_file, '/exchange/data', image, chunks=(1, h // f, w // f))  # Create h5

    print("Done")


if __name__ == "__main__":


    # _, global_min, global_max = get_min_max_hdf5(scan_path,
    #                                              nworkers=64,
    #                                              worker_task=_task,
    #                                              crop_bounds=(0, 100000, 0, 100000, 0, 100000),  # bin1x1
    #                                              data_path='/exchange/data',
    #                                              )

    project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/"

    # Define paths
    sample_path = project_path
    out_name = "test"  # Name of the output file

    moving_path = sample_path + "Cardboard_A_LFOV_80kV_7W_air_4s_8mu_bin1_pos1_Stitch_scale_4.tif"
    fixed_path = sample_path + "Cardboard_A_4X_80kV_7W_air_3s_2mu_bin1_pos1_Stitch_scale_4.tif"

    print("test")


