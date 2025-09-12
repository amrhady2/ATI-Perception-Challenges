# ATI Perception Challenges

This repository contains solutions to two computer vision challenges focused on automotive perception tasks. Each challenge demonstrates structured, reproducible pipelines for visual inference, calibration, and metric estimation.

---

## 📍 Challenge 1 – Rim Center Location

**Objective:**  
Estimate the center of a car rim from a static image using classical computer vision techniques.

**Approach:**  
- Preprocessing: grayscale conversion, Gaussian blur  
- Edge detection: Canny  
- Circle detection: Hough Transform  
- Post-filtering: radius constraints and confidence scoring  
- Output: pixel coordinates of rim center

**Highlights:**  
- Modular pipeline for circle detection  
- Tunable parameters for robustness across rim types  
- Easily extendable to multi-frame or video-based input

---

## 🔴 Challenge 2 – Tread Depth Estimation

**Objective:**  
Estimate tire tread depth using structured light (laser line) from a lateral pan video.

**Approach:**  
- Frame extraction from video  
- Laser line detection via HSV thresholding  
- Depth estimation from vertical displacement of laser line  
- Calibration: pixel-to-cm conversion  
- Output: CSV and plot of tread profile over time

**Highlights:**  
- Real and simulated pipelines for validation  
- Sub-millimeter precision via structured light triangulation  
- Robust to low-texture surfaces and lighting variation

---

## 📁 Folder Structure

Challenge 1 - Rim Center Location/
├── Final/                  # Final model, evaluation scripts, overlays
├── Old/                   # Archived experiments
├── rim_center_ready.ipynb # Main Jupyter notebook
├── README.md              # This file
├── .gitignore             # Clean repo setup


Challenge 2 - Tread Depth/
├── tread_depth_video.py       # Real video pipeline
├── tread_depth_simulation.py  # Synthetic validation pipeline
├── results/
│   ├── tread_profile.csv
│   └── tread_depth_plot.png



---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt

