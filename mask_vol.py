import os
import nibabel as nib
import numpy as np
import argparse

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply mask to volume and save")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--mask_path", type=str, required=False, help="Size of each chunk (D, H, W).")

    args = parser.parse_args()
    return args

def mask_volume(image, mask):
    image[mask == 0] = 0  # Set values outside mask to zero

if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU_HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path)
        print("Sample path: ", sample_path)
    if args.scan_path is not None:
        scan_path = os.path.join(sample_path, args.scan_path)
        print("Scan path: ", scan_path)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)

    print(f"Loading {scan_path}")
    image = np.load(scan_path).astype(np.float32)
    # If you must open large images, consider using memory-mapped arrays:
    # image = np.load("image.npy", mmap_mode='r')  # won't load entire array into RAM

    mask = np.load(args.mask_path).astype(np.uint8)

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Apply mask to the image
    mask_volume(image, mask)

    # Save / process the chunk
    np.save(os.path.join(out_path, f"{out_name}_masked.npy"), image)


