# Challenge 1 - ATI Perception Challenge – Rim Center Identification  
Author: Amr Attia  
Submission Date: September 2025

# Problem Statement  
Automatically identify the center of car wheels from RGB images, even under tilt, occlusion, or absence.  
**Input:** Single RGB image (15 FPS stream)  
**Output:** Pixel coordinates of wheel center or “No wheel” if absent

# Approach Summary  
This project uses a heatmap-based regression model to predict wheel centers.  
- **Model:** Compact UNet-style CNN  
- **Output:** Single-channel heatmap + wheel presence classifier  
- **Loss:** MSE for heatmap + BCE for presence  
- **Inference:** Argmax of heatmap + thresholding for “no wheel” case

# Folder Structure  
```bash
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
