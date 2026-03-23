# Road Damage Detection System

> ML project for detecting and quantifying road damage in images and videos.

---

## Overview

This project leverages YOLOv8 Segmentation to automatically detect and measure road damage (potholes and cracks) in both images and video streams. It provides real-time damage percentage calculations with smoothed results for video analysis.

**Key Highlights:**
- Semantic segmentation of road damage areas
- Real-time video processing with frame-by-frame analysis
- Damage percentage calculation based on segmentation masks
- Easy-to-use Python API for both images and videos

---

## Project Structure

```
ML_Project/
├── README.md         
├── main.ipynb
├── image_analysis.py
├── video_analysis.py
|
├── input/                      
│   ├── image3.jpeg
│   └── sample2.mp4
│
├── output/ 
│   ├── img_out3.jpeg
│   └── output2.avi
│
└── dataset/
    ├── data.yaml
    ├── train/
    │   └── images/
    └── valid/
        └── images/
```

---

## Features

### Image Analysis
- **Single Image Processing**: Detect and segment road damage in static images
- **Damage Area Calculation**: Compute percentage of damaged area
- **Annotated Output**: Visual overlay of detected damage regions
- **High Accuracy**: Built on YOLOv8 segmentation backbone

### Video Analysis
- **Real-time Processing**: Frame-by-frame damage detection
- **Moving Average Smoothing**: Smooth damage percentage across frames
- **Video Output**: Save processed video with annotations

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager
- CUDA 11.x (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/utkarshk-iitr/ML_Project.git
   cd ML_Project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install torch torchvision opencv-python ultralytics numpy
   ```

---

## Usage

### Image Analysis

Analyze a single image for road damage:

```bash
python image_analysis.py
```

**Input:** `input/image3.jpeg`  
**Output:** `output/img_out3.jpeg`

### Video Analysis

Process a video stream and generate annotated output:

```bash
python video_analysis.py
```

**Input:** `input/sample2.mp4`  
**Output:** `output/output2.avi`


## Dataset

### Configuration (`dataset/data.yaml`)
```yaml
train: dataset/train/images
val: dataset/valid/images

```

## Output


**Image Analysis:**
```
Input: input/image3.jpeg
Road Damage: 4.32%
Output: output/img_out3.jpeg
```

**Video Analysis:**
```
Input: input/sample2.mp4
Processing Speed: ~30ms per frame
Smoothed Damage Range: 0.5% - 8.7%
Output: output/output2.avi
```

---

This project is part of the CSL-382 course assignment at IIT Roorkee.

---

## Authors

**Utkarsh Kumar**
23114101

**Ankit Kumar**
23114006

**Aadit Kumar Sahoo**

**Kavy Vaghela**

---
