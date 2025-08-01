# Customer Segmentation Analysis

## Overview
This project segments mall customers using K-Means clustering based on `Age`, `Annual Income`, and `Spending Score`. It employs PCA for 3D visualization and SHAP for feature importance, delivering actionable insights through interactive plots and summary statistics.

## Features
- **Data Preprocessing**: Standardizes features using `StandardScaler`.
- **Clustering**: Applies K-Means with k=5, selected via the elbow method.
- **Visualization**: Creates an interactive 3D PCA scatter plot with axes labeled by dominant features and an elbow plot with k=5 annotation.
- **Interpretability**: Uses SHAP to analyze feature importance for Cluster 0.
- **Outputs**: Saves plots (`elbow_plot.png`, `shap_bar_plot.png`, `customer_segmentation_3d.html`, `customer_segmentation_3d.png`), data (`customer_segments.csv`), and PCA loadings (`pca_loadings.csv`).

## Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Setup
1. **Dataset**: Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it in the project directory.
2. **Run the Script**:
   ```bash
   python customer_segmentation.py
   ```
3. **Outputs** (in `outputs` directory):
   - `elbow_plot.png`: Elbow plot with k=5 annotation.
   - `shap_bar_plot.png`: SHAP feature importance for Cluster 0.
   - `customer_segmentation_3d.html`: Interactive 3D scatter plot.
   - `customer_segmentation_3d.png`: Static 3D plot (requires `kaleido`).
   - `pca_loadings.csv`: PCA feature contributions.
   - `cluster_summary.csv`: Cluster center statistics.
   - `customer_segments.csv`: DataFrame with cluster labels.

## Notes
- Install `kaleido` (`pip install kaleido`) for PNG export of the 3D plot.
- The 3D plot axes are labeled with dominant features (e.g., “Mostly Income”) based on PCA loadings for clarity.