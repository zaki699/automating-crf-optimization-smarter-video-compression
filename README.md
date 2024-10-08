# CRF Optimization for Video Compression

This project provides a collection of Python scripts that optimize Constant Rate Factor (CRF) for video compression, allowing efficient encoding with quality metrics like VMAF, PSNR, and SSIM. The scripts also include functionality for dynamically adjusting bitrate and computing BD-Rate to compare different encoding settings.

## Files

### 1. `CRF-Optimization-with-bitrate-cap.py`
This script performs CRF optimization with a predefined bitrate cap. It adjusts the CRF while maintaining the target bitrate, ensuring an efficient balance between video quality and file size.

- **Features**:
    - Optimizes CRF based on quality metrics like VMAF, PSNR, and SSIM.
    - Applies a bitrate cap to prevent excessive bitrate usage.
    - Reports the optimal CRF that meets the desired quality and bitrate limits.

### 2. `CRF-Optimization-with-dynamic-bitrate-cap.py`
This script dynamically adjusts the bitrate cap during CRF optimization, allowing more flexibility in meeting video quality thresholds while controlling file size.

- **Features**:
    - Dynamically adjusts bitrate based on scene complexity.
    - Optimizes CRF while meeting the bitrate cap and quality thresholds.
    - Suitable for videos with varying levels of complexity across scenes.

### 3. `CRF-Optimization-with-dynamic-bitrate-cap-bdrate.py`
This version adds BD-Rate computation to the dynamic bitrate optimization process, providing a quantitative comparison of different encoding settings across various quality levels.

- **Features**:
    - Dynamically adjusts bitrate during CRF optimization.
    - Computes BD-Rate to compare the efficiency of different encoding settings.
    - Helps identify the optimal trade-off between quality and file size.

### 4. `CRF-Optimization-without-bitrate-cap.py`
This script optimizes CRF without enforcing a bitrate cap, allowing full freedom to explore the trade-offs between video quality and file size.

- **Features**:
    - Optimizes CRF purely based on quality metrics like VMAF, PSNR, and SSIM.
    - No restrictions on bitrate, allowing the encoder to maximize quality.
    - Useful for scenarios where file size is less of a concern compared to video quality.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/crf-optimization-tool.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. CRF Optimization with Bitrate Cap
To optimize CRF while capping the bitrate:

```bash
python CRF-Optimization-with-bitrate-cap.py --original_video /path/to/video --output_video /path/to/output.mp4 --vmaf_model_path /path/to/vmaf/model
```

###  2. CRF Optimization with Dynamic Bitrate Cap
To dynamically adjust the bitrate cap during CRF optimization:
```bash
python CRF-Optimization-with-dynamic-bitrate-cap.py --original_video /path/to/video --output_video /path/to/output.mp4 --vmaf_model_path /path/to/vmaf/model
```

###  3. CRF Optimization with Dynamic Bitrate Cap and BD-Rate
To include BD-Rate computation in the dynamic bitrate cap optimization:
```bash
python CRF-Optimization-with-dynamic-bitrate-cap-bdrate.py --original_video /path/to/video --output_video /path/to/output.mp4 --vmaf_model_path /path/to/vmaf/model
```

###  4. CRF Optimization without Bitrate Cap
To perform CRF optimization without any bitrate restrictions:
```bash
python CRF-Optimization-without-bitrate-cap.py --original_video /path/to/video --output_video /path/to/output.mp4 --vmaf_model_path /path/to/vmaf/model
```

###  Key Features
- VMAF, PSNR, SSIM: These quality metrics are used to guide CRF optimization.
- BD-Rate: Quantitative comparison between different encoding settings.
- Bitrate Cap: Option to enforce a bitrate limit or dynamically adjust it based on scene complexity.
- Scene Complexity: Accounts for scene changes to better control bitrate and quality.

### Contributions
Feel free to contribute to this repository by submitting a pull request or opening an issue!
