import os
import subprocess
import tempfile
import json
import cv2
import re

# Quality thresholds
VMAF_THRESHOLD = 90.0
PSNR_THRESHOLD = 40.0
SSIM_THRESHOLD = 0.95

# Bitrate threshold (in kbps)
BITRATE_THRESHOLD = 5000  # Set your target bitrate cap

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

# Function to compress video at a specified CRF
def compress_video(original_video, crf_value, output_path):
    encode_cmd = [
        'ffmpeg',
        '-i', original_video,
        '-c:v', 'libx264',
        '-crf', str(crf_value),
        '-preset', 'medium',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
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
    target_bitrate_kbps = (width * height * frame_rate) / (1000 * quality_factor)
    video.release()
    return target_bitrate_kbps

# Adjust CRF iteratively with bitrate capping using binary search
def optimize_crf_with_bitrate_cap(original_video, initial_crf=23, min_crf=0, max_crf=51, max_iterations=10, quality_factor=24, vmaf_model_path=None):
    lower_bound = min_crf
    upper_bound = max_crf
    crf = initial_crf
    optimal_crf = None

    # Calculate the dynamic bitrate threshold based on video characteristics
    dynamic_bitrate_threshold = calculate_dynamic_bitrate_threshold(original_video, quality_factor)
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize CRF for video compression based on quality metrics and bitrate cap.")
    parser.add_argument('original_video', type=str, help="Path to the original video file.")
    parser.add_argument('--initial_crf', type=int, default=23, help="Initial CRF value to start optimization.")
    parser.add_argument('--vmaf_model_path', type=str, default=None, help="Path to VMAF model file.")
    parser.add_argument('--output_video', type=str, default='output_video.mp4', help="Path to the output compressed video.")
    parser.add_argument('--bitrate_threshold', type=int, default=5000, help="Maximum allowed bitrate in kbps.")
    parser.add_argument('--quality_factor', type=int, default=24, help="Quality factor for dynamic bitrate threshold calculation.")
    args = parser.parse_args()

    # Update the bitrate threshold if provided via command line
    BITRATE_THRESHOLD = args.bitrate_threshold

    # Optimize the CRF while capping the bitrate
    optimal_crf = optimize_crf_with_bitrate_cap(
        args.original_video,
        initial_crf=args.initial_crf,
        vmaf_model_path=args.vmaf_model_path,
        quality_factor=args.quality_factor
    )

    # Use the optimized CRF to compress the video
    success = compress_video(args.original_video, optimal_crf, args.output_video)
    if success:
        print(f"Video compressed with optimal CRF: {optimal_crf}")
    else:
        print("Failed to compress the video with the optimal CRF.")