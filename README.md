# Image_Processing

This Python script automates the detection, measurement, and classification of cracks in images. It extracts crack length, width, area, angle, and type, visualizes results, and exports all data to Excel for further analysis.

## Features

- **Image Preprocessing**
  - Resize images and convert to grayscale.
  - Apply Otsu’s thresholding to create binary crack masks.
  - Exclude margins to remove noise.

- **Crack Segmentation**
  - Detect contours corresponding to cracks.
  - Fit bounding boxes and filter by aspect ratio.
  - Extract crack segments for analysis.

- **Crack Measurement**
  - Compute crack length, area, and maximum width.
  - Calculate orientation angle and classify cracks (shear, tensile, diagonal, vertical).
  - Derive additional metrics like imaginary length and ratio T.

- **Visualization**
  - Overlay bounding boxes, angles, and width markers on images.
  - Highlight the widest crack point.

- **Data Export**
  - Save all crack properties (length, width, area, type, etc.) to an Excel file.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Pandas

## Usage

```bash
python image processing for crack width.py
