import os
import subprocess
import tempfile
import json
import cv2
import re
import numpy as np
from scipy import interpolate

# Quality thresholds (can be adjusted as needed)
VMAF_THRESHOLD = 90.0
PSNR_THRESHOLD = 40.0
SSIM_THRESHOLD = 0.95

# Function to calculate quality metrics using FFmpeg
def calculate_metrics(original_video, compressed_video, vmaf_model_path=None):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as vmaf_log_file, \
         tempfile.NamedTemporaryFile(suffix='.log', delete=False) as psnr_log_file, \
         tempfile.NamedTemporaryFile(suffix='.log', delete=False) as ssim_log_file:
        
        vmaf_log_path = vmaf_log_file.name
        psnr_log_path = psnr_log_file.name
        ssim_log_path = ssim_log_file.name

    # Build FFmpeg command to calculate metrics
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', compressed_video,
        '-i', original_video,
        '-filter_complex',
        f"[0:v][1:v]psnr=stats_file={psnr_log_path};"
        f"[0:v][1:v]ssim=stats_file={ssim_log_path};"
        f"[0:v][1:v]libvmaf="
        f"log_path={vmaf_log_path}:"
        f"log_fmt=json"
        + (f":model_path={vmaf_model_path}" if vmaf_model_path else ""),
        '-f', 'null', '-'
    ]

    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_output = result.stderr.decode('utf-8')

    if result.returncode != 0:
        print("FFmpeg command failed:")
        print(stderr_output)
        return None, None, None, None

    # Parse PSNR score
    psnr_score = None
    with open(psnr_log_path, 'r') as f:
        content = f.read()
        match = re.search(r'psnr_avg:(\s*\d+\.\d+)', content)
        if match:
            psnr_score = float(match.group(1))

    # Parse SSIM score
    ssim_score = None
    with open(ssim_log_path, 'r') as f:
        content = f.read()
        match = re.search(r'All:(\s*\d+\.\d+)', content)
        if match:
            ssim_score = float(match.group(1))

    # Parse VMAF score
    vmaf_score = None
    with open(vmaf_log_path, 'r') as f:
        vmaf_data = json.load(f)
        if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
            vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']

    # Clean up temporary files
    os.remove(psnr_log_path)
    os.remove(ssim_log_path)
    os.remove(vmaf_log_path)

    # Get bitrate of the compressed video
    bitrate = get_bitrate(compressed_video)

    return vmaf_score, psnr_score, ssim_score, bitrate

# Function to get the bitrate of the video using ffprobe
def get_bitrate(video_path):
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=bit_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("ffprobe command failed:")
        print(result.stderr.decode('utf-8'))
        return None
    bitrate = result.stdout.decode('utf-8').strip()
    try:
        bitrate_kbps = int(bitrate) / 1000  # Convert to kbps
    except ValueError:
        bitrate_kbps = None
    return bitrate_kbps

# Function to compress video at a specified CRF with the chosen codec and preset
def compress_video(original_video, crf_value, output_path, codec, preset, encoder_settings=None):
    encode_cmd = [
        'ffmpeg',
        '-i', original_video,
        '-c:v', codec,  # Use the codec specified by the user
        '-crf', str(crf_value),  # Set CRF value
        '-preset', preset,  # Use the preset specified by the user
    ]

    # Add VBR and CQ options if using GPU-based codecs like h264_nvenc or hevc_nvenc
    if codec in ['h264_nvenc', 'hevc_nvenc']:
        encode_cmd.extend([
            '-rc:v', 'vbr',  # Variable Bit Rate mode for efficient compression
            '-cq:v', str(crf_value)  # Constant Quality mode for NVENC
        ])

    if encoder_settings:
        encode_cmd.extend(encoder_settings)
    encode_cmd.extend([
        '-y',  # Overwrite output file if it exists
        output_path
    ])
    result = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg compression failed:")
        print(result.stderr.decode('utf-8'))
        return False
    return True

# Function to calculate automatic bitrate threshold based on resolution and frame rate
def calculate_dynamic_bitrate_threshold(video_path, quality_factor=24):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    video.release()

    if frame_rate == 0:
        print("Error: Unable to retrieve frame rate from the video.")
        return None

    target_bitrate_kbps = (width * height * frame_rate) / (1000 * quality_factor)
    return target_bitrate_kbps

# Adjust CRF iteratively with bitrate capping using binary search
def optimize_crf_with_bitrate_cap(original_video, initial_crf=23, min_crf=0, max_crf=51, max_iterations=10, quality_factor=24, vmaf_model_path=None):
    lower_bound = min_crf
    upper_bound = max_crf
    crf = initial_crf
    optimal_crf = None

    # Calculate the dynamic bitrate threshold based on video characteristics
    dynamic_bitrate_threshold = calculate_dynamic_bitrate_threshold(original_video, quality_factor)
    if dynamic_bitrate_threshold is None:
        print("Failed to calculate dynamic bitrate threshold.")
        return initial_crf  # Fallback to initial CRF
    print(f"Dynamic Bitrate Threshold: {dynamic_bitrate_threshold:.2f} kbps")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Testing CRF = {crf}")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video_file:
            compressed_video = tmp_video_file.name

        # Compress the video at the current CRF
        success = compress_video(original_video, crf, compressed_video)
        if not success:
            os.remove(compressed_video)
            break  # Exit if compression fails

        # Calculate the quality metrics and bitrate
        vmaf_score, psnr_score, ssim_score, bitrate = calculate_metrics(original_video, compressed_video, vmaf_model_path)

        # Remove the compressed video
        os.remove(compressed_video)

        # Check if metrics are valid
        if vmaf_score is None or psnr_score is None or ssim_score is None or bitrate is None:
            print("Failed to calculate one or more quality metrics or bitrate.")
            break

        # Print the evaluation results
        print(f"VMAF: {vmaf_score:.2f}, PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, Bitrate: {bitrate:.2f} kbps")

        # Use dynamic bitrate threshold instead of BITRATE_THRESHOLD
        if (vmaf_score >= VMAF_THRESHOLD and psnr_score >= PSNR_THRESHOLD and
            ssim_score >= SSIM_THRESHOLD and bitrate <= dynamic_bitrate_threshold):
            optimal_crf = crf
            # Try a higher CRF to see if we can reduce file size further
            lower_bound = crf + 1
        else:
            # Quality not sufficient or bitrate too high, need lower CRF
            upper_bound = crf - 1

        if lower_bound > upper_bound:
            break  # No more CRF values to test

        # Update CRF for next iteration
        crf = (lower_bound + upper_bound) // 2

    if optimal_crf is not None:
        print(f"Optimal CRF found: {optimal_crf} with bitrate under {dynamic_bitrate_threshold:.2f} kbps")
    else:
        print("Failed to find an optimal CRF that meets the quality thresholds and bitrate cap.")
        optimal_crf = initial_crf  # Fallback to initial CRF

    return optimal_crf

# BD-Rate calculation function
def calculate_bd_rate(ref_bitrates, ref_metrics, test_bitrates, test_metrics):
    # Convert lists to numpy arrays
    ref_bitrates = np.array(ref_bitrates)
    ref_metrics = np.array(ref_metrics)
    test_bitrates = np.array(test_bitrates)
    test_metrics = np.array(test_metrics)

    # Ensure that metrics are in the same range
    min_metric = max(min(ref_metrics), min(test_metrics))
    max_metric = min(max(ref_metrics), max(test_metrics))

    # If the curves do not overlap, BD-Rate cannot be calculated
    if min_metric >= max_metric:
        print("Error: No overlapping quality range between reference and test data.")
        return None

    # Interpolate RD points
    metric_range = np.linspace(min_metric, max_metric, num=100)
    interp_ref = interpolate.pchip_interpolate(ref_metrics, np.log(ref_bitrates), metric_range)
    interp_test = interpolate.pchip_interpolate(test_metrics, np.log(test_bitrates), metric_range)

    # Calculate the integral over the metric range
    avg_diff = np.trapz(interp_test - interp_ref, metric_range) / (max_metric - min_metric)

    # Compute BD-Rate
    bd_rate = (np.exp(avg_diff) - 1) * 100  # Percentage difference

    return bd_rate

# Function to collect RD points
def collect_rd_points(original_video, crf_values, encoder_settings=None, vmaf_model_path=None):
    bitrates = []
    metrics = []
    for crf in crf_values:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video_file:
            compressed_video = tmp_video_file.name

        # Compress the video at the current CRF
        success = compress_video(original_video, crf, compressed_video, encoder_settings)
        if not success:
            os.remove(compressed_video)
            continue  # Skip to the next CRF value

        # Calculate the quality metrics and bitrate
        vmaf_score, psnr_score, ssim_score, bitrate = calculate_metrics(original_video, compressed_video, vmaf_model_path)

        # Remove the compressed video
        os.remove(compressed_video)

        # Check if metrics are valid
        if vmaf_score is None or bitrate is None:
            print(f"Failed to calculate metrics for CRF {crf}.")
            continue

        # Collect data
        bitrates.append(bitrate)
        metrics.append(vmaf_score)  # You can choose PSNR or SSIM instead

        print(f"CRF: {crf}, VMAF: {vmaf_score:.2f}, Bitrate: {bitrate:.2f} kbps")

    return bitrates, metrics

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize CRF for video compression and compute BD-Rate.")
    parser.add_argument('original_video', type=str, help="Path to the original video file.")
    parser.add_argument('--initial_crf', type=int, default=23, help="Initial CRF value to start optimization.")
    parser.add_argument('--vmaf_model_path', type=str, default=None, help="Path to VMAF model file.")
    parser.add_argument('--output_video', type=str, default='output_video.mp4', help="Path to the output compressed video.")
    parser.add_argument('--crf_values', type=str, default='18,23,28,33,38', help="Comma-separated list of CRF values to test.")
    parser.add_argument('--quality_factor', type=int, default=24, help="Quality factor for dynamic bitrate threshold calculation.")
    parser.add_argument('--codec', type=str, default='libx264', help="Codec to use for encoding (e.g., libx264, h264_nvenc).")
    parser.add_argument('--preset', type=str, default='medium', help="Preset to use for encoding (e.g., slow, fast, medium).")

    args = parser.parse_args()

    # Optimize the CRF while capping the bitrate
    optimal_crf = optimize_crf_with_bitrate_cap(
        args.original_video,
        initial_crf=args.initial_crf,
        vmaf_model_path=args.vmaf_model_path,
        quality_factor=args.quality_factor,
        codec=args.codec,
        preset=args.preset
    )

    # Use the optimized CRF to compress the video
    success = compress_video(args.original_video, optimal_crf, args.output_video, args.codec, args.preset)
    if success:
        print(f"Video compressed with optimal CRF: {optimal_crf}")
    else:
        print("Failed to compress the video with the optimal CRF.")

    # Parse CRF values
    crf_values = [int(crf.strip()) for crf in args.crf_values.split(',')]

    # Collect RD points for the reference encoding (e.g., default settings)
    print("\nCollecting RD points for reference encoding...")
    ref_bitrates, ref_metrics = collect_rd_points(
        args.original_video,
        crf_values,
        encoder_settings=None,  # Default encoder settings
        vmaf_model_path=args.vmaf_model_path
    )

    # Collect RD points for the test encoding (e.g., with your optimizations)
    print("\nCollecting RD points for test encoding...")
    test_encoder_settings = ['-tune', 'zerolatency']  # Example of test settings
    test_bitrates, test_metrics = collect_rd_points(
        args.original_video,
        crf_values,
        encoder_settings=test_encoder_settings,
        vmaf_model_path=args.vmaf_model_path
    )

    # Compute BD-Rate
    bd_rate = calculate_bd_rate(ref_bitrates, ref_metrics, test_bitrates, test_metrics)

    if bd_rate is not None:
        print(f"\nBD-Rate between test and reference encodings: {bd_rate:.2f}%")
        if bd_rate < 0:
            print("Test encoding is better (lower bitrate for same quality).")
        else:
            print("Reference encoding is better (lower bitrate for same quality).")
    else:
        print("BD-Rate could not be calculated due to insufficient data.")