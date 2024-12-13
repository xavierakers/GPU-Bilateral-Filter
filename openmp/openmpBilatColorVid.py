import sys
import os
import subprocess
import numpy as np
import cv2
import ctypes as ct
import time


def precompute_spatial_weight(radius, spatial_sigma):
    grid = np.arange(-radius, radius + 1)  # 1D grid for both x and y
    x, y = np.meshgrid(grid, grid)

    # Compute the squared Euclidean distance for the grid
    spatial_dist_sq = np.power(x, 2, dtype=np.float32) + np.power(
        y, 2, dtype=np.float32
    )

    # Compute Gaussian weights
    spatial_weights = np.exp(
        -spatial_dist_sq / (2 * np.power(spatial_sigma, 2, dtype=np.float32))
    )

    return spatial_weights


if len(sys.argv) < 6:
    print(
        "command usage :",
        sys.argv[0],
        "input.mp4",
        "radius",
        "intensity sigma",
        "spatial sigma",
        "num threads"
    )
    exit(1)


# Filepaths
lib_path = "./openmpBilatColorVid.so"
vid_path = sys.argv[1]

# Filtering Variables
radius = int(sys.argv[2])
if radius > 16:
    print("error : max radius is 16")
    exit(1)
intensity_sigma = int(sys.argv[3])  # sigma r
intensity_factor = 1 / (2 * intensity_sigma * intensity_sigma)  # precompute
spatial_sigma = int(sys.argv[4])
spatial_weights = precompute_spatial_weight(radius, spatial_sigma)
num_threads = int(sys.argv[5])
# For timing and image output
output_prefix = f"{vid_path.split('/')[-1].split('.')[0]}_{radius}_{intensity_sigma}_{spatial_sigma}"
so_version = lib_path.split("_")[-1].split(".")[0]
outfile = f"outputs/{output_prefix}_{so_version}.mp4"
timing_path = f"outputs/timing_{output_prefix}_{so_version}.txt"
if not os.path.exists("./outputs/"):
    os.makedirs("./outputs")

# Load shared object library
lib = ct.cdll.LoadLibrary(lib_path)
lib.filter.argtypes = [
    ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_uint8),
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_float,
    ct.POINTER(ct.c_float),
    ct.c_int,
]
lib.filter.restype = None

# Load in video
start_time = time.perf_counter()

cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    print(f"error : unable to open {vid_path}")
    exit(1)

# Some video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)
colorspace = 4
print(frame_size)

# Create Pinned memory
frames = np.empty(
    shape=(num_frames, frame_height + 2 * radius, frame_width + 2 * radius, colorspace),
    dtype=np.uint8,
)
filtered_frames = np.empty(shape=(num_frames, frame_height, frame_width, colorspace), dtype=np.uint8)

# Read in all frames into pinned memory
idx = 0
for idx in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame_pad = np.pad(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        pad_width=((radius, radius), (radius, radius), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    frames[idx] = frame_pad
    print('-', end='', flush=True)
# Close video reader
cap.release()
print()

# Set up video writer object
out = cv2.VideoWriter(
    "output_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
)

load_time = time.perf_counter() - start_time
print(f"{num_frames}\t Number of Frames", flush=True)
print(f"{load_time:.6f}\t Load time", flush=True)

# Filter video
start_time = time.perf_counter()

frames_cptr = np.ascontiguousarray(frames, dtype=np.uint8).ctypes.data_as(
    ct.POINTER(ct.c_uint8)
)

filtered_frames_cptr = np.ascontiguousarray(
    filtered_frames, dtype=np.uint8
).ctypes.data_as(ct.POINTER(ct.c_uint8))

spatial_weights_cptr = np.ascontiguousarray(
    spatial_weights, dtype=np.float32
).ctypes.data_as(ct.POINTER(ct.c_float))

lib.filter(
    frames_cptr,
    filtered_frames_cptr,
    ct.c_int(num_frames),
    ct.c_int(frame_height),
    ct.c_int(frame_width),
    ct.c_int(colorspace),
    ct.c_int(radius),
    ct.c_float(intensity_factor),
    spatial_weights_cptr,
    ct.c_int(num_threads),
)

filter_time = time.perf_counter() - start_time
print(f"{filter_time:.6f}\t Filter time", flush=True)

# Save video
start_time = time.perf_counter()
# Write to video object
try:
    for frame in filtered_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
finally:
    out.release()
save_time = time.perf_counter() - start_time
print(f"{save_time:.6f}\t Save time")

# Properly encode the video
try:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-i",
            "output_raw.mp4",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            outfile,
            "-y",
        ],
        check=True,
    )
    subprocess.run(["rm", "output_raw.mp4"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error during processing: {e}")
    sys.exit(1)


total_time = load_time + filter_time + save_time
print(f"{save_time:.6f}\t Save time")
print(f"{total_time:.6f}\t Total time")
print("------------------------------------------------------")
print(f"Video successfully saved")
print("------------------------------------------------------")

with open("timing.data", "w") as fptr:
    fptr.write("load_time,filter_time,save_time,total_time\n")
    fptr.write(f"{load_time},{filter_time},{save_time},{total_time}")
