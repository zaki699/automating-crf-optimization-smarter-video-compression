import os
import subprocess
import tempfile
import json

# Quality thresholds
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
        return None, None, None

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

    return vmaf_score, psnr_score, ssim_score

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

# Iteratively adjust CRF using binary search to meet quality thresholds
def optimize_crf(original_video, initial_crf=23, min_crf=0, max_crf=51, max_iterations=10, vmaf_model_path=None):
    lower_bound = min_crf
    upper_bound = max_crf
    crf = initial_crf
    optimal_crf = None

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Testing CRF = {crf}")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video_file:
            compressed_video = tmp_video_file.name

        # Compress the video at the current CRF
        success = compress_video(original_video, crf, compressed_video)
        if not success:
            os.remove(compressed_video)
            break  # Exit if compression fails

        # Calculate the quality metrics
        vmaf_score, psnr_score, ssim_score = calculate_metrics(original_video, compressed_video, vmaf_model_path)

        # Remove the compressed video
        os.remove(compressed_video)

        # Check if metrics are valid
        if vmaf_score is None or psnr_score is None or ssim_score is None:
            print("Failed to calculate one or more quality metrics.")
            break

        # Print the evaluation results
        print(f"VMAF Score: {vmaf_score:.2f}, PSNR Score: {psnr_score:.2f}, SSIM Score: {ssim_score:.4f}")

        # Check if the quality metrics meet the thresholds
        if vmaf_score >= VMAF_THRESHOLD and psnr_score >= PSNR_THRESHOLD and ssim_score >= SSIM_THRESHOLD:
            optimal_crf = crf
            # Try a higher CRF to see if we can reduce file size further
            lower_bound = crf + 1
        else:
            # Quality not sufficient, need lower CRF
            upper_bound = crf - 1

        if lower_bound > upper_bound:
            break  # No more CRF values to test

        # Update CRF for next iteration
        crf = (lower_bound + upper_bound) // 2

    if optimal_crf is not None:
        print(f"Optimal CRF found: {optimal_crf}")
    else:
        print("Failed to find an optimal CRF that meets the quality thresholds.")
        optimal_crf = initial_crf  # Fallback to initial CRF

    return optimal_crf

# Example usage
if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Optimize CRF for video compression based on quality metrics.")
    parser.add_argument('original_video', type=str, help="Path to the original video file.")
    parser.add_argument('--initial_crf', type=int, default=23, help="Initial CRF value to start optimization.")
    parser.add_argument('--vmaf_model_path', type=str, default=None, help="Path to VMAF model file.")
    parser.add_argument('--output_video', type=str, default='output_video.mp4', help="Path to the output compressed video.")
    args = parser.parse_args()

    # Optimize the CRF for this video
    optimal_crf = optimize_crf(args.original_video, initial_crf=args.initial_crf, vmaf_model_path=args.vmaf_model_path)

    # Use the optimized CRF to compress the video
    success = compress_video(args.original_video, optimal_crf, args.output_video)
    if success:
        print(f"Video compressed with optimal CRF: {optimal_crf}")
    else:
        print("Failed to compress the video with the optimal CRF.")